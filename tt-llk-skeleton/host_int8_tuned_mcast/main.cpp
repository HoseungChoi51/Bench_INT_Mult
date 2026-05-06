// Tensix tuned INT8 matmul benchmark *with operand multicast*.
//
// Adapts the upstream programming example
//   tt_metal/programming_examples/matmul/matmul_multicore_reuse_mcast/
//   matmul_multicore_reuse_mcast.cpp
// from BF16 to INT8, with the same adaptations as the non-mcast harness
// at tt-llk-skeleton/host_int8_tuned/main.cpp:
//   - DataFormat::Int8 throughout (single_tile_size = 1024 B)
//   - fp32_dest_acc_en = true for INT32 destination accumulator
//   - host inputs converted to sign-magnitude on the byte representation
//   - per_core_M = Mt/grid_y, per_core_N = Nt/grid_x (grid-filling)
//
// The compute kernel and all 4 multicast reader variants + the writer
// are reused **unmodified** from upstream. Multicast NoC routing,
// semaphores, and dataflow are dtype-agnostic — they stream byte tiles.
//
// Stdout: median_ms,p10_ms,p90_ms,arch,n_cores,gate,err  (gate=skipped).

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

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace {

// SUBBLOCK_HW_CHOICES, lifted from upstream
// tt_metal/programming_examples/matmul/matmul_common/bmm_op.hpp:111. Used
// to pick the largest (h, w) pair that divides per_core_M and per_core_N.
constexpr std::array<std::tuple<uint32_t, uint32_t>, 20> SUBBLOCK_HW_CHOICES = {{
    {4, 2}, {2, 4}, {8, 1}, {1, 8}, {7, 1}, {1, 7}, {3, 2}, {2, 3}, {6, 1}, {1, 6},
    {5, 1}, {1, 5}, {2, 2}, {4, 1}, {1, 4}, {3, 1}, {1, 3}, {2, 1}, {1, 2}, {1, 1},
}};

// --- INT8 helpers (lifted from host_int8_tuned/main.cpp) ----------------

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

struct Args {
    int M = 256, K = 256, N = 256;
    int warmup = 5, iters = 30;
    int in0_block_w = 2;
    std::string fidelity = "HiFi4";
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
    if (!tt_metal_home || !*tt_metal_home) { emit_skip("TT_METAL_HOME not set"); return 3; }

    std::shared_ptr<distributed::MeshDevice> mesh_device;
    try {
        mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    } catch (const std::exception& e) { emit_skip(e.what()); return 2; }

    CoreCoord grid;
    int n_cores_total = 0;
    try {
        grid = mesh_device->compute_with_storage_grid_size();
        n_cores_total = static_cast<int>(grid.x * grid.y);
    } catch (const std::exception& e) { emit_skip(e.what()); return 2; }

    const uint32_t Mt = args.M / TILE_HEIGHT;
    const uint32_t Kt = args.K / TILE_WIDTH;
    const uint32_t Nt = args.N / TILE_WIDTH;
    const uint32_t num_cores_x = grid.x;
    const uint32_t num_cores_y = grid.y;

    if (num_cores_x < 2 || num_cores_y < 2) { emit_skip("mcast needs grid >= 2x2"); return 3; }
    if (Mt % num_cores_y != 0) { emit_skip("Mt not divisible by grid.y"); return 3; }
    if (Nt % num_cores_x != 0) { emit_skip("Nt not divisible by grid.x"); return 3; }
    if (args.in0_block_w <= 0 || Kt % args.in0_block_w != 0) {
        emit_skip("Kt not divisible by --in0-block-w"); return 3;
    }
    const uint32_t per_core_M = Mt / num_cores_y;
    const uint32_t per_core_N = Nt / num_cores_x;
    uint32_t out_subblock_h = 1, out_subblock_w = 1;
    for (auto& sub : SUBBLOCK_HW_CHOICES) {
        const uint32_t sh = std::get<0>(sub), sw = std::get<1>(sub);
        if (per_core_M % sh == 0 && per_core_N % sw == 0) {
            out_subblock_h = sh; out_subblock_w = sw; break;
        }
    }
    std::fprintf(stderr,
        "[int8 tuned mcast] grid=%ux%u Mt=%u Kt=%u Nt=%u "
        "per_core_M=%u per_core_N=%u out_sub=%ux%u in0_block_w=%d fidelity=%s\n",
        num_cores_x, num_cores_y, Mt, Kt, Nt, per_core_M, per_core_N,
        out_subblock_h, out_subblock_w, args.in0_block_w, args.fidelity.c_str());

    try {
        // --- DRAM buffers ----------------------------------------------
        const tt::DataFormat fmt = tt::DataFormat::Int8;
        const uint32_t single_tile_size = tt::tile_size(fmt);  // 1024 for INT8
        const uint32_t in0_CB_size = (per_core_M * args.in0_block_w * 2) * single_tile_size;
        const uint32_t in1_CB_size = (per_core_N * args.in0_block_w * 2) * single_tile_size;
        const uint32_t out_CB_size = (per_core_M * per_core_N) * single_tile_size;

        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig a_cfg{.size = single_tile_size * Mt * Kt};
        distributed::ReplicatedBufferConfig b_cfg{.size = single_tile_size * Kt * Nt};
        distributed::ReplicatedBufferConfig c_cfg{.size = single_tile_size * Mt * Nt};
        auto a_buf = distributed::MeshBuffer::create(a_cfg, dram_cfg, mesh_device.get());
        auto b_buf = distributed::MeshBuffer::create(b_cfg, dram_cfg, mesh_device.get());
        auto c_buf = distributed::MeshBuffer::create(c_cfg, dram_cfg, mesh_device.get());

        // --- Host data: random INT8 → sign-magnitude → tilize ----------
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
        distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);

        // --- Build program (mcast scaffolding) -------------------------
        Program program{};

        // Compute kernel compile-time args (12 entries — same as non-mcast).
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
            static_cast<uint32_t>(args.in0_block_w),
            in0_num_subblocks, in0_block_num_tiles, in0_subblock_num_tiles,
            in1_num_subblocks, in1_block_num_tiles, in1_per_core_w,
            num_blocks, out_subblock_h, out_subblock_w, out_subblock_num_tiles,
            batch,
        };

        // --- Core regions (full grid, 2D mcast topology) ---------------
        const uint32_t sx = 0, sy = 0;
        CoreRange all_cores({sx, sy}, {sx + num_cores_x - 1, sy + num_cores_y - 1});
        CoreRange left_column({sx, sy}, {sx, sy + num_cores_y - 1});
        CoreRange all_except_left_column(
            {sx + 1, sy}, {sx + num_cores_x - 1, sy + num_cores_y - 1});
        CoreRange in0_sender_in1_sender({sx, sy}, {sx, sy});
        CoreRange in0_sender_in1_receiver(
            {sx, sy + 1}, {sx, sy + num_cores_y - 1});
        CoreRange in0_receiver_in1_sender(
            {sx + 1, sy}, {sx + num_cores_x - 1, sy});
        CoreRange in0_receiver_in1_receiver(
            {sx + 1, sy + 1}, {sx + num_cores_x - 1, sy + num_cores_y - 1});

        // --- CBs (Int8 in/out + Int8 intermediate c_24) ----------------
        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(in0_CB_size, {{CBIndex::c_0, fmt}})
                .set_page_size(CBIndex::c_0, single_tile_size));
        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(in1_CB_size, {{CBIndex::c_1, fmt}})
                .set_page_size(CBIndex::c_1, single_tile_size));
        std::map<uint8_t, tt::DataFormat> out_fmt_spec{
            {static_cast<uint8_t>(CBIndex::c_16), fmt},
            {24, fmt},
        };
        CreateCircularBuffer(program, all_cores,
            CircularBufferConfig(out_CB_size, out_fmt_spec)
                .set_page_size(CBIndex::c_16, single_tile_size)
                .set_page_size(24, single_tile_size));

        // --- Compile-time args for reader / writer ---------------------
        std::vector<uint32_t> reader_cta;
        TensorAccessorArgs(*a_buf).append_to(reader_cta);
        TensorAccessorArgs(*b_buf).append_to(reader_cta);
        std::vector<uint32_t> writer_cta;
        TensorAccessorArgs(*c_buf).append_to(writer_cta);

        // --- Kernel paths (upstream programming-example tree) ---------
        const std::string up = std::string(tt_metal_home)
            + "/tt_metal/programming_examples/matmul/matmul_common/kernels";
        const std::string r_ss = up + "/dataflow/reader_bmm_tile_layout_in0_sender_in1_sender.cpp";
        const std::string r_sr = up + "/dataflow/reader_bmm_tile_layout_in0_sender_in1_receiver.cpp";
        const std::string r_rs = up + "/dataflow/reader_bmm_tile_layout_in0_receiver_in1_sender.cpp";
        const std::string r_rr = up + "/dataflow/reader_bmm_tile_layout_in0_receiver_in1_receiver.cpp";
        const std::string w_path = up + "/dataflow/writer_bmm_tile_layout.cpp";
        const std::string c_path = up + "/compute/bmm_large_block_zm.cpp";

        // --- Reader / writer / compute kernels (4 + 2 + 1) -------------
        // Note: NoC assignment matches upstream — sender_sender + sender_receiver
        // use RISCV_1+NOC0; receiver_sender + receiver_receiver use RISCV_1+NOC1;
        // writer_noc0 (RISCV_0+NOC0) on all_except_left_column;
        // writer_noc1 (RISCV_0+NOC1) on left_column.
        auto kid_r_ss = CreateKernel(program, r_ss, in0_sender_in1_sender,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = reader_cta});
        auto kid_r_sr = CreateKernel(program, r_sr, in0_sender_in1_receiver,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = reader_cta});
        auto kid_r_rs = CreateKernel(program, r_rs, in0_receiver_in1_sender,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_cta});
        auto kid_r_rr = CreateKernel(program, r_rr, in0_receiver_in1_receiver,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_cta});
        auto kid_w_noc0 = CreateKernel(program, w_path, all_except_left_column,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_cta});
        auto kid_w_noc1 = CreateKernel(program, w_path, left_column,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = writer_cta});
        CreateKernel(program, c_path, all_cores,
            ComputeConfig{
                .math_fidelity = parse_fidelity(args.fidelity),
                .fp32_dest_acc_en = true,  // INT32 dst slots for INT8
                .compile_args = compute_args,
            });

        // --- Semaphores (4: in0/in1 × sender/receiver) ----------------
        const uint32_t sem_in0_send = CreateSemaphore(program, all_cores, INVALID);
        const uint32_t sem_in0_recv = CreateSemaphore(program, all_cores, INVALID);
        const uint32_t sem_in1_send = CreateSemaphore(program, all_cores, INVALID);
        const uint32_t sem_in1_recv = CreateSemaphore(program, all_cores, INVALID);

        // --- Per-core runtime args ------------------------------------
        const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
        const uint32_t b_addr = static_cast<uint32_t>(b_buf->address());
        const uint32_t c_addr = static_cast<uint32_t>(c_buf->address());
        for (uint32_t cy = 0; cy < num_cores_y; ++cy) {
            for (uint32_t cx = 0; cx < num_cores_x; ++cx) {
                CoreCoord core{cx, cy};
                CoreCoord left_core{sx, cy};
                CoreCoord left_core_plus_one{sx + 1, cy};
                CoreCoord right_core{sx + num_cores_x - 1, cy};
                CoreCoord top_core{cx, sy};
                CoreCoord top_core_plus_one{cx, sy + 1};
                CoreCoord bottom_core{cx, sy + num_cores_y - 1};
                auto lp  = mesh_device->worker_core_from_logical_core(left_core);
                auto lp1 = mesh_device->worker_core_from_logical_core(left_core_plus_one);
                auto rp  = mesh_device->worker_core_from_logical_core(right_core);
                auto tp  = mesh_device->worker_core_from_logical_core(top_core);
                auto tp1 = mesh_device->worker_core_from_logical_core(top_core_plus_one);
                auto bp  = mesh_device->worker_core_from_logical_core(bottom_core);

                std::vector<uint32_t> r_args = {
                    a_addr,
                    Kt * per_core_M * cy,                  // in0_buffer_start_tile_id
                    1u, Kt,                                // strides w/h
                    static_cast<uint32_t>(args.in0_block_w),  // next_block_stride
                    static_cast<uint32_t>(args.in0_block_w),  // in0_block_w
                    per_core_M,                            // in0_block_h
                    args.in0_block_w * per_core_M,         // in0_block_num_tiles

                    b_addr,
                    per_core_N * cx,                       // in1_buffer_start_tile_id
                    1u, Nt,                                // strides w/h
                    static_cast<uint32_t>(args.in0_block_w) * Nt,
                    per_core_N,
                    static_cast<uint32_t>(args.in0_block_w),
                    per_core_N * args.in0_block_w,

                    Kt / args.in0_block_w,                 // num_blocks

                    // in0 mcast (along X / row): dest box right_core → left_core_plus_one
                    static_cast<uint32_t>(rp.x),  static_cast<uint32_t>(rp.y),
                    static_cast<uint32_t>(lp1.x), static_cast<uint32_t>(lp1.y),
                    num_cores_x - 1,                       // in0_mcast_num_dests
                    static_cast<uint32_t>(lp.x),  static_cast<uint32_t>(lp.y),
                    sem_in0_send, sem_in0_recv,

                    // in1 mcast (along Y / col): dest box bottom_core → top_core_plus_one
                    static_cast<uint32_t>(bp.x),  static_cast<uint32_t>(bp.y),
                    static_cast<uint32_t>(tp1.x), static_cast<uint32_t>(tp1.y),
                    num_cores_y - 1,                       // in1_mcast_num_dests
                    static_cast<uint32_t>(tp.x),  static_cast<uint32_t>(tp.y),
                    sem_in1_send, sem_in1_recv,

                    Mt * Kt, Kt * Nt, batch, /*bcast_B=*/0u,
                };
                std::vector<uint32_t> w_args = {
                    c_addr,
                    cx * per_core_N + cy * per_core_M * Nt,
                    1u, Nt,
                    out_subblock_w, out_subblock_h * Nt,
                    out_subblock_w, out_subblock_h,
                    out_subblock_w * out_subblock_h,
                    per_core_N / out_subblock_w,
                    per_core_M / out_subblock_h,
                    Mt * Nt, batch,
                };

                if (cx == 0 && cy == 0) {
                    SetRuntimeArgs(program, kid_r_ss, core, r_args);
                    SetRuntimeArgs(program, kid_w_noc1, core, w_args);
                } else if (cx == 0 && cy != 0) {
                    SetRuntimeArgs(program, kid_r_sr, core, r_args);
                    SetRuntimeArgs(program, kid_w_noc1, core, w_args);
                } else if (cx != 0 && cy == 0) {
                    SetRuntimeArgs(program, kid_r_rs, core, r_args);
                    SetRuntimeArgs(program, kid_w_noc0, core, w_args);
                } else {
                    SetRuntimeArgs(program, kid_r_rr, core, r_args);
                    SetRuntimeArgs(program, kid_w_noc0, core, w_args);
                }
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
        emit_csv(median, p10, p90, "blackhole", n_cores_total, "skipped", "");
        return 0;
    } catch (const std::exception& e) {
        emit_skip(e.what()); return 5;
    }
}
