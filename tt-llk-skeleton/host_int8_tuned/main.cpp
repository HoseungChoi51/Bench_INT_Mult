// Tensix tuned INT8 matmul benchmark.
//
// Adapts the upstream programming example
//   tt_metal/programming_examples/matmul/matmul_multicore_reuse/matmul_multicore_reuse.cpp
// from BF16 to INT8 by switching CB DataFormat to Int8, enabling
// fp32_dest_acc_en for the INT32 destination accumulator, and converting
// inputs to sign-magnitude (the on-card representation Tensix's INT8
// matrix path expects — see v1
// `tt-llk-skeleton/host/main.cpp::convert_to_sign_mag`).
//
// The compute kernel reused unchanged is the upstream block-tiled kernel
//   `tt_metal/programming_examples/matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp`
// — same `mm_init` + `matmul_tiles` LLK calls as our v1 INT8 reference,
// just with block / sub-block orchestration via 12 compile-time args.
//
// **No multicast.** This is the mid-tier "tuned" path: block reuse on,
// operand multicast off. Per the v2 plan that's enough to land within a
// few factors of BF16/HiFi4 (~140 TFLOPS on this card); full mcast adds
// another 2–3× on top and is left for future work.
//
// CLI:
//   --M, --K, --N      shape (multiples of TILE=32, factor cleanly into
//                       the 11×10 grid; 256/512/1024/2048 work)
//   --fidelity         HiFi4 (default) | HiFi2 | LoFi
//   --warmup, --iters  timing samples
//   --in0-block-w      inner-K block width in tiles (default 2)
//
// Stdout (one CSV line):
//
//     median_ms,p10_ms,p90_ms,arch,n_cores,gate,err
//
// gate is "skipped" — INT8 packed output saturates with random inputs at
// any non-trivial K, so we don't validate post-hoc; throughput is the
// only honest measurement.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tile.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// --- bmm_op helpers (inlined from
//     tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp,
//     trimmed to the bits the non-mcast block-tiled path needs) ---------

constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
}};

std::vector<uint32_t> get_prime_factors(uint32_t n) {
    uint32_t i = 2;
    std::vector<uint32_t> pf;
    while (i * i <= n) {
        if (n % i != 0) ++i;
        else { n /= i; pf.push_back(i); }
    }
    if (n > 1) pf.push_back(n);
    return pf;
}

std::vector<uint32_t> get_possible_products(const std::vector<uint32_t>& factors) {
    if (factors.empty()) return {1};
    std::vector<uint32_t> products;
    for (uint32_t fac : factors) {
        std::vector<uint32_t> new_products;
        if (!std::count(products.begin(), products.end(), fac)) new_products.push_back(fac);
        for (uint32_t prod : products) {
            if (!std::count(products.begin(), products.end(), fac * prod))
                new_products.push_back(fac * prod);
        }
        products.reserve(products.size() + new_products.size());
        products.insert(products.end(), new_products.begin(), new_products.end());
    }
    std::sort(products.begin(), products.end());
    return products;
}

uint32_t get_maximum_block_dim(int32_t block_dim, int32_t in0_block_w) {
    int32_t other = (400 - 2 * in0_block_w * block_dim) / (2 * in0_block_w + block_dim);
    return other > 0 ? static_cast<uint32_t>(other) : 0u;
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> get_large_matmul_params(
    uint32_t Mt, uint32_t Nt, uint32_t num_cores_y, uint32_t num_cores_x,
    uint32_t in0_block_w) {
    auto Nt_fac = get_prime_factors(Nt);
    auto Mt_fac = get_prime_factors(Mt);
    uint32_t Npc_min = 1, Mpc_min = 1;

    for (auto it = Nt_fac.begin(); it != Nt_fac.end(); ++it) {
        if (*it > num_cores_x) { Npc_min *= *it; Nt_fac.erase(it); --it; }
    }
    for (auto it = Mt_fac.begin(); it != Mt_fac.end(); ++it) {
        if (*it > num_cores_y) { Mpc_min *= *it; Mt_fac.erase(it); --it; }
    }

    if (Npc_min > get_maximum_block_dim(Mpc_min, in0_block_w)) return {0, 0, 0, 0};

    uint32_t Mpc = Mpc_min, Npc = Npc_min;
    auto search_subblock = [](uint32_t mpc, uint32_t npc)
        -> std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> {
        for (auto& sub : SUBBLOCK_HW_CHOICES) {
            auto sh = std::get<0>(sub), sw = std::get<1>(sub);
            if (mpc % sh == 0 && npc % sw == 0) return {mpc, npc, sh, sw};
        }
        return {0, 0, 0, 0};
    };

    if (Mpc_min > 1) {
        auto Npc_choices = get_possible_products(Nt_fac);
        auto Npc_max = get_maximum_block_dim(Mpc_min, in0_block_w);
        for (auto e : Npc_choices) {
            if (e * Npc_min <= Npc_max) Npc = e * Npc_min;
            else break;
        }
        if (Mt / Mpc > num_cores_y || Nt / Npc > num_cores_x) return {0, 0, 0, 0};
        return search_subblock(Mpc, Npc);
    } else if (Npc_min > 1) {
        auto Mpc_choices = get_possible_products(Mt_fac);
        auto Mpc_max = get_maximum_block_dim(Npc_min, in0_block_w);
        for (auto e : Mpc_choices) {
            if (e * Mpc_min <= Mpc_max) Mpc = e * Mpc_min;
            else break;
        }
        if (Mt / Mpc > num_cores_y || Nt / Npc > num_cores_x) return {0, 0, 0, 0};
        return search_subblock(Mpc, Npc);
    } else {
        auto Mpc_choices = get_possible_products(Mt_fac);
        auto Npc_choices = get_possible_products(Nt_fac);
        for (auto cur_npc : Npc_choices) {
            uint32_t cur_mpc = 1;
            auto Mpc_max = get_maximum_block_dim(cur_npc, in0_block_w);
            for (auto e : Mpc_choices) if (e <= Mpc_max) cur_mpc = e;
            if (Mt / cur_mpc > num_cores_y || Nt / cur_npc > num_cores_x) continue;
            auto r = search_subblock(cur_mpc, cur_npc);
            if (std::get<0>(r) != 0) return r;
        }
    }
    return {0, 0, 0, 0};
}

// --- INT8 helpers (lifted from our v1 INT8 path: host/main.cpp) --------

std::vector<int8_t> tilize_int8_nfaces(
    const std::vector<int8_t>& src, uint32_t M, uint32_t K) {
    constexpr uint32_t kTile = 32;
    constexpr uint32_t kFace = 16;
    constexpr uint32_t kFaceArea = kFace * kFace;
    const uint32_t Mt = M / kTile;
    const uint32_t Kt = K / kTile;
    std::vector<int8_t> dst(static_cast<size_t>(M) * K);
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            int8_t* tile = dst.data() + (static_cast<size_t>(mt) * Kt + kt) * (kTile * kTile);
            for (uint32_t face_id = 0; face_id < 4; ++face_id) {
                const uint32_t face_row_off = (face_id / 2) * kFace;
                const uint32_t face_col_off = (face_id % 2) * kFace;
                int8_t* face_dst = tile + face_id * kFaceArea;
                for (uint32_t fr = 0; fr < kFace; ++fr) {
                    for (uint32_t fc = 0; fc < kFace; ++fc) {
                        const uint32_t src_row = mt * kTile + face_row_off + fr;
                        const uint32_t src_col = kt * kTile + face_col_off + fc;
                        face_dst[fr * kFace + fc] = src[static_cast<size_t>(src_row) * K + src_col];
                    }
                }
            }
        }
    }
    return dst;
}

void convert_to_sign_mag(std::vector<int8_t>& v) {
    for (auto& x : v) {
        if (x < 0) {
            const uint8_t mag = static_cast<uint8_t>(-static_cast<int>(x));
            x = static_cast<int8_t>(0x80 | mag);
        }
    }
}

// --- CSV output ----------------------------------------------------------

void emit_csv(double median_ms, double p10_ms, double p90_ms,
              const char* arch, int n_cores, const char* gate, const char* err) {
    if (err && *err) {
        std::printf("null,null,null,%s,%d,%s,%s\n",
                    arch ? arch : "?", n_cores, gate ? gate : "skipped", err);
    } else {
        std::printf("%.6f,%.6f,%.6f,%s,%d,%s,\n",
                    median_ms, p10_ms, p90_ms, arch ? arch : "?", n_cores,
                    gate ? gate : "skipped");
    }
}

void emit_skip(const char* err) { emit_csv(0, 0, 0, "?", 0, "skipped", err); }

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    size_t idx = static_cast<size_t>(q * (v.size() - 1));
    return v[idx];
}

// --- Args ---------------------------------------------------------------

struct Args {
    int M = 256, K = 256, N = 256;
    int warmup = 5, iters = 30;
    int in0_block_w = 2;
    std::string fidelity = "HiFi4";  // HiFi4 | HiFi2 | LoFi
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto take = [&](const char* k, std::string& dst) {
            if (s == k && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        std::string tmp;
        if (take("--M", tmp))            { a.M = std::stoi(tmp); continue; }
        if (take("--K", tmp))            { a.K = std::stoi(tmp); continue; }
        if (take("--N", tmp))            { a.N = std::stoi(tmp); continue; }
        if (take("--warmup", tmp))       { a.warmup = std::stoi(tmp); continue; }
        if (take("--iters", tmp))        { a.iters = std::stoi(tmp); continue; }
        if (take("--in0-block-w", tmp))  { a.in0_block_w = std::stoi(tmp); continue; }
        if (take("--fidelity", a.fidelity)) continue;
    }
    return a;
}

MathFidelity parse_fidelity(const std::string& s) {
    if (s == "HiFi4") return MathFidelity::HiFi4;
    if (s == "HiFi3") return MathFidelity::HiFi3;
    if (s == "HiFi2") return MathFidelity::HiFi2;
    if (s == "LoFi")  return MathFidelity::LoFi;
    return MathFidelity::HiFi4;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.M % TILE_HEIGHT != 0 || args.K % TILE_WIDTH != 0 || args.N % TILE_WIDTH != 0) {
        emit_skip("M/K/N must be multiples of TILE=32"); return 3;
    }

    const char* tt_metal_home = std::getenv("TT_METAL_HOME");
    if (!tt_metal_home || !*tt_metal_home) {
        emit_skip("TT_METAL_HOME not set"); return 3;
    }

    std::shared_ptr<distributed::MeshDevice> mesh_device;
    try {
        mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    } catch (const std::exception& e) { emit_skip(e.what()); return 2; }

    int n_cores_total = 0;
    CoreCoord grid;
    try {
        grid = mesh_device->compute_with_storage_grid_size();
        n_cores_total = static_cast<int>(grid.x * grid.y);
    } catch (const std::exception& e) { emit_skip(e.what()); return 2; }

    try {
        const uint32_t Mt = args.M / TILE_HEIGHT;
        const uint32_t Kt = args.K / TILE_WIDTH;
        const uint32_t Nt = args.N / TILE_WIDTH;
        const uint32_t num_cores_x = grid.x;
        const uint32_t num_cores_y = grid.y;

        if (args.in0_block_w <= 0 || Kt % args.in0_block_w != 0) {
            emit_skip("Kt not divisible by --in0-block-w"); return 3;
        }

        // Block selection: mirror the *upstream* test_matmul_2d_host_perf
        // recipe (line 300-301: per_core_M = Mt / grid_y, per_core_N =
        // Nt / grid_x), NOT the auto-tuner in get_large_matmul_params —
        // that auto-tuner often picks tiny blocks that under-fill the
        // grid (e.g. only 22 cores out of 110), which is why upstream
        // hardcodes the grid-filling values.
        if (Mt % num_cores_y != 0) { emit_skip("Mt not divisible by grid.y"); return 3; }
        if (Nt % num_cores_x != 0) { emit_skip("Nt not divisible by grid.x"); return 3; }
        const uint32_t per_core_M = Mt / num_cores_y;
        const uint32_t per_core_N = Nt / num_cores_x;
        // Sub-block search: pick the first (h, w) ∈ SUBBLOCK_HW_CHOICES
        // that divides both per_core_M and per_core_N.
        uint32_t out_subblock_h = 1, out_subblock_w = 1;
        for (auto& sub : SUBBLOCK_HW_CHOICES) {
            const uint32_t sh = std::get<0>(sub), sw = std::get<1>(sub);
            if (per_core_M % sh == 0 && per_core_N % sw == 0) {
                out_subblock_h = sh; out_subblock_w = sw; break;
            }
        }
        // Discard the unused helpers so -Wunused-function doesn't fire.
        (void)get_large_matmul_params;
        (void)get_prime_factors;
        (void)get_possible_products;
        (void)get_maximum_block_dim;
        std::fprintf(stderr,
            "[int8 tuned] grid=%ux%u Mt=%u Kt=%u Nt=%u "
            "per_core_M=%u per_core_N=%u out_sub=%ux%u in0_block_w=%d fidelity=%s\n",
            num_cores_x, num_cores_y, Mt, Kt, Nt, per_core_M, per_core_N,
            out_subblock_h, out_subblock_w, args.in0_block_w, args.fidelity.c_str());

        const tt::DataFormat fmt = tt::DataFormat::Int8;
        const uint32_t single_tile_size = tt::tile_size(fmt);  // 1024 for INT8
        const uint32_t in0_CB_size  = (per_core_M * args.in0_block_w * 2) * single_tile_size;
        const uint32_t in1_CB_size  = (per_core_N * args.in0_block_w * 2) * single_tile_size;
        const uint32_t out_CB_size  = (per_core_M * per_core_N) * single_tile_size;

        // --- Compute kernel compile-time args (matches upstream order)
        const uint32_t num_blocks            = Kt / args.in0_block_w;
        const uint32_t in0_num_subblocks     = per_core_M / out_subblock_h;
        const uint32_t in0_block_num_tiles   = out_subblock_h * args.in0_block_w * in0_num_subblocks;
        const uint32_t in0_subblock_num_tiles= out_subblock_h * args.in0_block_w;
        const uint32_t in1_num_subblocks     = per_core_N / out_subblock_w;
        const uint32_t in1_block_num_tiles   = out_subblock_w * args.in0_block_w * in1_num_subblocks;
        const uint32_t in1_per_core_w        = out_subblock_w * in1_num_subblocks;
        const uint32_t out_subblock_num_tiles= out_subblock_h * out_subblock_w;
        const uint32_t batch                 = 1;

        std::vector<uint32_t> compute_args = {
            args.in0_block_w, in0_num_subblocks, in0_block_num_tiles, in0_subblock_num_tiles,
            in1_num_subblocks, in1_block_num_tiles, in1_per_core_w,
            num_blocks, out_subblock_h, out_subblock_w, out_subblock_num_tiles,
            batch,
        };

        // --- Host data: random INT8, sign-magnitude, tilized
        const size_t a_elems = static_cast<size_t>(args.M) * args.K;
        const size_t b_elems = static_cast<size_t>(args.K) * args.N;
        std::vector<int8_t> a_host(a_elems), b_host(b_elems);
        std::mt19937 rng(20260429u);
        std::uniform_int_distribution<int> dist(-127, 127);
        for (auto& v : a_host) v = static_cast<int8_t>(dist(rng));
        for (auto& v : b_host) v = static_cast<int8_t>(dist(rng));
        a_host = tilize_int8_nfaces(a_host, args.M, args.K);
        b_host = tilize_int8_nfaces(b_host, args.K, args.N);
        convert_to_sign_mag(a_host);
        convert_to_sign_mag(b_host);

        auto& cq = mesh_device->mesh_command_queue();
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig a_cfg{.size = single_tile_size * Mt * Kt};
        distributed::ReplicatedBufferConfig b_cfg{.size = single_tile_size * Kt * Nt};
        distributed::ReplicatedBufferConfig c_cfg{.size = single_tile_size * Mt * Nt};
        auto a_buf = distributed::MeshBuffer::create(a_cfg, dram_cfg, mesh_device.get());
        auto b_buf = distributed::MeshBuffer::create(b_cfg, dram_cfg, mesh_device.get());
        auto c_buf = distributed::MeshBuffer::create(c_cfg, dram_cfg, mesh_device.get());
        distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);

        // --- Build program
        Program program{};
        const uint32_t num_blocks_y = Mt / per_core_M;
        const uint32_t num_blocks_x = Nt / per_core_N;
        const uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
        if (num_blocks_total > num_cores_x * num_cores_y) {
            emit_skip("output blocks exceed core count"); return 3;
        }
        CoreRangeSet all_cores =
            num_cores_to_corerangeset(num_blocks_total, grid, /*row_wise=*/true);
        const int n_cores_used = static_cast<int>(num_blocks_total);

        // Input + output CBs (output uses the c_16/c_24 spill pattern from
        // upstream — interm0_cb is required by bmm_large_block_zm.cpp).
        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(in0_CB_size, {{CBIndex::c_0, fmt}})
                .set_page_size(CBIndex::c_0, single_tile_size));
        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(in1_CB_size, {{CBIndex::c_1, fmt}})
                .set_page_size(CBIndex::c_1, single_tile_size));
        std::map<uint8_t, tt::DataFormat> out_fmt_spec{
            {static_cast<uint8_t>(CBIndex::c_16), fmt},
            {24, fmt},  // interm0_cb
        };
        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(out_CB_size, out_fmt_spec)
                .set_page_size(CBIndex::c_16, single_tile_size)
                .set_page_size(24, single_tile_size));

        std::vector<uint32_t> reader_cta;
        TensorAccessorArgs(*a_buf).append_to(reader_cta);
        TensorAccessorArgs(*b_buf).append_to(reader_cta);
        std::vector<uint32_t> writer_cta;
        TensorAccessorArgs(*c_buf).append_to(writer_cta);

        const std::string up_mc = std::string(tt_metal_home)
            + "/tt_metal/programming_examples/matmul/matmul_common/kernels";
        const std::string reader_path  = up_mc + "/dataflow/reader_bmm_tile_layout.cpp";
        const std::string writer_path  = up_mc + "/dataflow/writer_bmm_tile_layout.cpp";
        const std::string compute_path = up_mc + "/compute/bmm_large_block_zm.cpp";

        auto reader_kid = CreateKernel(program, reader_path, all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_cta});
        auto writer_kid = CreateKernel(program, writer_path, all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_cta});
        CreateKernel(program, compute_path, all_cores,
            ComputeConfig{
                .math_fidelity = parse_fidelity(args.fidelity),
                // INT32 dst slots required for INT8 accumulator.
                .fp32_dest_acc_en = true,
                .compile_args = compute_args,
            });

        // Per-core runtime args (mirror upstream non-mcast walking order).
        uint32_t num_blocks_read = 0;
        for (uint32_t oy = 0; oy < num_blocks_y; ++oy) {
            for (uint32_t ox = 0; ox < num_blocks_x; ++ox) {
                const uint32_t cx = num_blocks_read % num_cores_x;
                const uint32_t cy = num_blocks_read / num_cores_x;
                CoreCoord core{cx, cy};
                const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
                const uint32_t b_addr = static_cast<uint32_t>(b_buf->address());
                const uint32_t c_addr = static_cast<uint32_t>(c_buf->address());
                std::vector<uint32_t> r_args = {
                    a_addr,
                    Kt * per_core_M * oy,         // in0_tensor_start_tile_id
                    1u,                            // in0_stride_w
                    Kt,                            // in0_stride_h
                    static_cast<uint32_t>(args.in0_block_w),     // in0_next_block_stride
                    static_cast<uint32_t>(args.in0_block_w),     // in0_block_w
                    per_core_M,                    // in0_block_h
                    args.in0_block_w * per_core_M, // in0_block_num_tiles
                    b_addr,
                    per_core_N * ox,               // in1_tensor_start_tile_id
                    1u,                            // in1_stride_w
                    Nt,                            // in1_stride_h
                    static_cast<uint32_t>(args.in0_block_w) * Nt, // in1_next_block_stride
                    per_core_N,                    // in1_block_w
                    static_cast<uint32_t>(args.in0_block_w),     // in1_block_h
                    per_core_N * args.in0_block_w, // in1_block_num_tiles
                    Kt / args.in0_block_w,         // num_blocks
                    Mt * Kt, Kt * Nt, batch, /*bcast_B=*/0u,
                };
                std::vector<uint32_t> w_args = {
                    c_addr,
                    ox * per_core_N + oy * per_core_M * Nt,   // out_tensor_start_tile_id
                    1u,                                       // stride_w
                    Nt,                                       // stride_h
                    out_subblock_w,                           // next_subblock_stride_w
                    out_subblock_h * Nt,                      // next_subblock_stride_h
                    out_subblock_w, out_subblock_h,
                    out_subblock_w * out_subblock_h,
                    per_core_N / out_subblock_w,
                    per_core_M / out_subblock_h,
                    Mt * Nt, batch,
                };
                SetRuntimeArgs(program, reader_kid, core, r_args);
                SetRuntimeArgs(program, writer_kid, core, w_args);
                ++num_blocks_read;
            }
        }

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        for (int w = 0; w < args.warmup; ++w) {
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }
        distributed::Finish(cq);

        std::vector<double> times_ms; times_ms.reserve(args.iters);
        for (int it = 0; it < args.iters; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            distributed::Finish(cq);
            auto t1 = std::chrono::steady_clock::now();
            times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (times_ms.empty()) { emit_skip("no samples"); return 4; }
        std::sort(times_ms.begin(), times_ms.end());
        const double median = times_ms[times_ms.size() / 2];
        const double p10 = percentile(times_ms, 0.10);
        const double p90 = percentile(times_ms, 0.90);

        // gate=skipped — INT8 packed output saturates with random values
        // at non-trivial K; verifying correctness needs bounded inputs and
        // is out of scope for the throughput benchmark.
        emit_csv(median, p10, p90, "blackhole", n_cores_used, "skipped", "");
        return 0;
    } catch (const std::exception& e) {
        emit_skip(e.what()); return 5;
    }
}
