// Tenstorrent Blackhole bench host driver.
//
// Pairs with scripts/bench_nvidia.py. One invocation runs one
// (backend, layer, M, K, N) cell: warmup + iters samples of the dispatch,
// then prints a single CSV line on stdout for tt-llk-skeleton/bench_blackhole.py
// to parse.
//
// Layer A — single matmul, capability probe.
// Layer B — single matmul, raw GEMM throughput.
// Layer C — 25 matmuls (the byte-decomposition recipe of BENCHMARK.md §4),
//           wall-clock includes all 25 dispatches. Cost-proxy: the on-device
//           output is *not* used as the modular result; the recipe's
//           correctness is checked once on the host bigint reference inside
//           bench_blackhole.py. The Blackhole row therefore reports
//           "exact 36-bit modmul throughput estimated from the cost of the
//           25-GEMM cascade", documented in device_detail.note.
//
// Backends:
//   tt_llk_bf16  — Float16_b inputs/outputs, fp16 dst accumulator.
//   tt_llk_int8  — Int8 inputs, Int8 packed output, int32 dst accumulator
//                  (ComputeConfig.fp32_dest_acc_en = true).
//   tt_llk_sfpu_fp32 — TODO; emits a clean "skipped" record.
//
// Stdout (one CSV line, this exact column order):
//
//     median_ms,p10_ms,p90_ms,arch,n_cores,err
//
// On error, median/p10/p90 are the literal string "null" and `err` carries
// the exception message.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

#ifndef KERNEL_DIR
#define KERNEL_DIR ""  // absolute path injected by CMake; see host/CMakeLists.txt
#endif

namespace {

struct Args {
    std::string backend = "tt_llk_bf16";
    std::string layer = "A";
    int M = 256, K = 256, N = 256;
    int warmup = 5, iters = 30;
    uint64_t q36 = 0xFFFF00001ULL;  // matches scripts/_bench_common.py::q36_ntt_friendly_prime
    int n_gemms = 25;               // Layer C dispatches per measured iter
                                    // (5x5 = 25 for q36, 6x6 = 36 for q48)
};

// 32x32 tile.
constexpr uint32_t kTile = TILE_HEIGHT;

// Per-backend dispatch configuration. Keeps the matmul scaffolding identical
// across BF16 / INT8 — only formats, tile bytes, and the int-accumulator flag
// vary.
struct BackendCfg {
    DataFormat in_fmt;
    DataFormat out_fmt;
    uint32_t in_bytes;     // per-tile bytes for input CBs / DRAM buffers
    uint32_t out_bytes;    // per-tile bytes for output CB / DRAM buffer
    bool fp32_dest_acc;    // selects int32 / fp32 wide-dst accumulation
    bool is_int8;          // gates the int8 host data path
};

BackendCfg backend_cfg_for(const std::string& backend) {
    if (backend == "tt_llk_int8") {
        return BackendCfg{
            .in_fmt = DataFormat::Int8,
            .out_fmt = DataFormat::Int8,
            .in_bytes = static_cast<uint32_t>(sizeof(int8_t) * TILE_HW),    // 1024
            .out_bytes = static_cast<uint32_t>(sizeof(int8_t) * TILE_HW),   // 1024
            .fp32_dest_acc = true,
            .is_int8 = true,
        };
    }
    if (backend == "tt_llk_fp32_matrix") {
        // Tensix matrix engine with Float32 inputs. Internally uses TF32-fidelity
        // per Tenstorrent's fp32_accuracy doc — same fidelity as NVIDIA's Tensor
        // Core FP32 (cublaslt_tf32). DataFormat::Tf32 itself throws "unsupported
        // atm" in tt_backend_api_types.hpp:72; Float32 is the supported entry.
        return BackendCfg{
            .in_fmt = DataFormat::Float32,
            .out_fmt = DataFormat::Float32,
            .in_bytes = static_cast<uint32_t>(sizeof(float) * TILE_HW),     // 4096
            .out_bytes = static_cast<uint32_t>(sizeof(float) * TILE_HW),    // 4096
            .fp32_dest_acc = true,
            .is_int8 = false,
        };
    }
    // Default: BF16 (the only other backend with a working compute kernel).
    return BackendCfg{
        .in_fmt = DataFormat::Float16_b,
        .out_fmt = DataFormat::Float16_b,
        .in_bytes = static_cast<uint32_t>(sizeof(bfloat16) * TILE_HW),  // 2048
        .out_bytes = static_cast<uint32_t>(sizeof(bfloat16) * TILE_HW), // 2048
        .fp32_dest_acc = false,
        .is_int8 = false,
    };
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto take_str = [&](const char* k, std::string& dst) {
            if (s == k && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        std::string tmp;
        if (take_str("--backend", a.backend)) continue;
        if (take_str("--layer", a.layer)) continue;
        if (take_str("--M", tmp)) { a.M = std::stoi(tmp); continue; }
        if (take_str("--K", tmp)) { a.K = std::stoi(tmp); continue; }
        if (take_str("--N", tmp)) { a.N = std::stoi(tmp); continue; }
        if (take_str("--warmup", tmp)) { a.warmup = std::stoi(tmp); continue; }
        if (take_str("--iters", tmp)) { a.iters = std::stoi(tmp); continue; }
        if (take_str("--q36", tmp)) { a.q36 = std::stoull(tmp); continue; }
        if (take_str("--n-gemms", tmp)) { a.n_gemms = std::stoi(tmp); continue; }
    }
    return a;
}

const char* compute_kernel_for(const std::string& backend) {
    if (backend == "tt_llk_bf16") return KERNEL_DIR "/compute_bf16_mma.cpp";
    if (backend == "tt_llk_int8") return KERNEL_DIR "/compute_int8_mma.cpp";
    // Matrix-engine FP32 reuses the BF16 compute kernel verbatim — the LLK is
    // dtype-agnostic and the CB DataFormat (Float32) selects the FP32 path.
    if (backend == "tt_llk_fp32_matrix") return KERNEL_DIR "/compute_bf16_mma.cpp";
    if (backend == "tt_llk_sfpu_fp32") return KERNEL_DIR "/compute_sfpu_fp32.cpp";
    return nullptr;
}

void emit_csv(double median_ms, double p10_ms, double p90_ms,
              const char* arch, int n_cores, const char* err) {
    if (err && *err) {
        std::printf("null,null,null,%s,%d,%s\n", arch ? arch : "?", n_cores, err);
    } else {
        std::printf("%.6f,%.6f,%.6f,%s,%d,\n", median_ms, p10_ms, p90_ms,
                    arch ? arch : "?", n_cores);
    }
}

void emit_skip(const char* err) { emit_csv(0, 0, 0, "?", 0, err); }

// Build one matmul Program against the given inputs/outputs. Modeled on
// tt_metal/programming_examples/matmul/matmul_multi_core; the per-backend
// CB DataFormat / page_size and ComputeConfig.fp32_dest_acc_en come from cfg.
Program build_matmul_program(
    const BackendCfg& cfg,
    distributed::MeshDevice* device,
    const std::shared_ptr<distributed::MeshBuffer>& a_buf,
    const std::shared_ptr<distributed::MeshBuffer>& b_buf,
    const std::shared_ptr<distributed::MeshBuffer>& c_buf,
    uint32_t Mt, uint32_t Kt, uint32_t Nt,
    const std::string& compute_path) {
    Program program{};

    auto core_grid = device->compute_with_storage_grid_size();
    uint32_t num_output_tiles_total = Mt * Nt;

    auto [num_cores, all_cores, core_group_1, core_group_2,
          work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_tiles_total);

    constexpr uint32_t cb_depth = 2;  // double-buffered

    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(cb_depth * cfg.in_bytes, {{CBIndex::c_0, cfg.in_fmt}})
            .set_page_size(CBIndex::c_0, cfg.in_bytes));
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(cb_depth * cfg.in_bytes, {{CBIndex::c_1, cfg.in_fmt}})
            .set_page_size(CBIndex::c_1, cfg.in_bytes));
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(cb_depth * cfg.out_bytes, {{CBIndex::c_16, cfg.out_fmt}})
            .set_page_size(CBIndex::c_16, cfg.out_bytes));

    std::vector<uint32_t> reader_cta;
    TensorAccessorArgs(*a_buf).append_to(reader_cta);
    TensorAccessorArgs(*b_buf).append_to(reader_cta);
    auto reader_kid = CreateKernel(
        program, KERNEL_DIR "/reader.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                           .noc = NOC::RISCV_1_default,
                           .compile_args = reader_cta});

    std::vector<uint32_t> writer_cta;
    TensorAccessorArgs(*c_buf).append_to(writer_cta);
    auto writer_kid = CreateKernel(
        program, KERNEL_DIR "/writer.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                           .noc = NOC::RISCV_0_default,
                           .compile_args = writer_cta});

    auto compute_kid = CreateKernel(
        program, compute_path, all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = cfg.fp32_dest_acc,
            .compile_args = {},
        });

    uint32_t work_offset = 0;
    auto work_groups = {
        std::make_pair(core_group_1, work_per_core1),
        std::make_pair(core_group_2, work_per_core2)};
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // DeviceAddr is 64-bit; truncate to the 32-bit runtime-arg
                // slot. Buffers fit comfortably under 4 GiB on this device.
                const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
                const uint32_t b_addr = static_cast<uint32_t>(b_buf->address());
                const uint32_t c_addr = static_cast<uint32_t>(c_buf->address());
                SetRuntimeArgs(program, reader_kid, core,
                               {a_addr, b_addr, Mt, Kt, Nt,
                                work_offset, work_per_core});
                SetRuntimeArgs(program, writer_kid, core,
                               {c_addr, work_per_core, work_offset});
                SetRuntimeArgs(program, compute_kid, core,
                               {work_per_core, Kt});
                work_offset += work_per_core;
            }
        }
    }
    return program;
}

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    size_t idx = static_cast<size_t>(q * (v.size() - 1));
    return v[idx];
}

// Tilize an int8 row-major M×K matrix into the device's tile-row-major layout:
// each 32x32 tile is 4 faces of 16x16, row-major within face, face order
// TL/TR/BL/BR (matches the BF16 path used by tilize_nfaces<bfloat16>, just
// without the standard-library template instantiation that doesn't ship for
// int8_t — see /home/chs/TT/tt-metal/tt_metal/impl/data_format/tilize_utils.cpp:570-573).
std::vector<int8_t> tilize_int8_nfaces(
    const std::vector<int8_t>& src, uint32_t M, uint32_t K) {
    constexpr uint32_t kFace = 16;
    constexpr uint32_t kFaceArea = kFace * kFace;  // 256
    const uint32_t Mt = M / kTile;
    const uint32_t Kt = K / kTile;
    std::vector<int8_t> dst(static_cast<size_t>(M) * K);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            int8_t* tile = dst.data()
                + (static_cast<size_t>(mt) * Kt + kt) * (kTile * kTile);
            for (uint32_t face_id = 0; face_id < 4; ++face_id) {
                const uint32_t face_row_off = (face_id / 2) * kFace;
                const uint32_t face_col_off = (face_id % 2) * kFace;
                int8_t* face_dst = tile + face_id * kFaceArea;
                for (uint32_t fr = 0; fr < kFace; ++fr) {
                    for (uint32_t fc = 0; fc < kFace; ++fc) {
                        const uint32_t src_row = mt * kTile + face_row_off + fr;
                        const uint32_t src_col = kt * kTile + face_col_off + fc;
                        face_dst[fr * kFace + fc] =
                            src[static_cast<size_t>(src_row) * K + src_col];
                    }
                }
            }
        }
    }
    return dst;
}

// Sign-magnitude conversion required by Tensix INT8 inputs (see
// /home/chs/TT/tt-metal/tests/tt_metal/tt_metal/llk/test_single_core_matmul_int8.cpp::convert_to_sign_mag).
// Two's-complement -v with v>0 → bit 7 set + magnitude.
void convert_to_sign_mag(std::vector<int8_t>& v) {
    for (auto& x : v) {
        if (x < 0) {
            const uint8_t mag = static_cast<uint8_t>(-static_cast<int>(x));
            x = static_cast<int8_t>(0x80 | mag);
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const char* compute_path = compute_kernel_for(args.backend);
    if (!compute_path) {
        emit_skip("unknown backend");
        return 1;
    }
    if (args.M % kTile != 0 || args.K % kTile != 0 || args.N % kTile != 0) {
        emit_skip("M/K/N must be multiples of TILE=32");
        return 3;
    }

    // Open device. Anything that goes wrong here is reported as a "skipped"
    // record (all-null perf) so the wrapper sees an honest signal.
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    try {
        mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 2;
    }

    std::string arch_name;
    int n_cores = 0;
    try {
        const auto cg = mesh_device->compute_with_storage_grid_size();
        n_cores = static_cast<int>(cg.x * cg.y);
        arch_name = "blackhole";
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 2;
    }

    const uint32_t Mt = args.M / kTile;
    const uint32_t Kt = args.K / kTile;
    const uint32_t Nt = args.N / kTile;

    // SFPU FP32 backend's compute kernel is still a TODO (see
    // kernels/compute_sfpu_fp32.cpp); surface a clean skip so the wrapper
    // marks the row appropriately. INT8 + BF16 fall through to the dispatch.
    if (args.backend == "tt_llk_sfpu_fp32") {
        emit_skip("backend kernel TODO: see kernels/compute_sfpu_fp32.cpp");
        return 0;
    }

    const BackendCfg cfg = backend_cfg_for(args.backend);

    try {
        // Allocate buffers with backend-specific tile sizes (BF16: 2048 B,
        // INT8: 1024 B for inputs and packed output).
        distributed::DeviceLocalBufferConfig dram_in_cfg{
            .page_size = cfg.in_bytes, .buffer_type = BufferType::DRAM};
        distributed::DeviceLocalBufferConfig dram_out_cfg{
            .page_size = cfg.out_bytes, .buffer_type = BufferType::DRAM};

        auto make_buf_in = [&](uint32_t n_tiles) {
            distributed::ReplicatedBufferConfig bcfg{.size = cfg.in_bytes * n_tiles};
            return distributed::MeshBuffer::create(bcfg, dram_in_cfg, mesh_device.get());
        };
        auto make_buf_out = [&](uint32_t n_tiles) {
            distributed::ReplicatedBufferConfig bcfg{.size = cfg.out_bytes * n_tiles};
            return distributed::MeshBuffer::create(bcfg, dram_out_cfg, mesh_device.get());
        };
        auto a_buf = make_buf_in(Mt * Kt);
        auto b_buf = make_buf_in(Kt * Nt);
        auto c_buf = make_buf_out(Mt * Nt);

        std::mt19937 rng(20260427u);
        auto& cq = mesh_device->mesh_command_queue();

        if (cfg.is_int8) {
            std::uniform_int_distribution<int> dist(-127, 127);
            std::vector<int8_t> a_host(static_cast<size_t>(args.M) * args.K);
            std::vector<int8_t> b_host(static_cast<size_t>(args.K) * args.N);
            for (auto& v : a_host) v = static_cast<int8_t>(dist(rng));
            for (auto& v : b_host) v = static_cast<int8_t>(dist(rng));
            a_host = tilize_int8_nfaces(a_host, args.M, args.K);
            b_host = tilize_int8_nfaces(b_host, args.K, args.N);
            convert_to_sign_mag(a_host);
            convert_to_sign_mag(b_host);
            distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
            distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);
        } else if (cfg.in_fmt == DataFormat::Float32) {
            // Tensix matrix-engine FP32 path. tilize_nfaces<float> is in the
            // standard library instantiation set.
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            std::vector<float> a_host(static_cast<size_t>(args.M) * args.K);
            std::vector<float> b_host(static_cast<size_t>(args.K) * args.N);
            for (auto& v : a_host) v = dist(rng);
            for (auto& v : b_host) v = dist(rng);
            a_host = tilize_nfaces(a_host, args.M, args.K);
            b_host = tilize_nfaces(b_host, args.K, args.N);
            distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
            distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);
        } else {
            std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
            std::vector<bfloat16> a_host(static_cast<size_t>(args.M) * args.K);
            std::vector<bfloat16> b_host(static_cast<size_t>(args.K) * args.N);
            for (auto& v : a_host) v = bfloat16(dist(rng));
            for (auto& v : b_host) v = bfloat16(dist(rng));
            a_host = tilize_nfaces(a_host, args.M, args.K);
            b_host = tilize_nfaces(b_host, args.K, args.N);
            distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
            distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);
        }

        Program program = build_matmul_program(
            cfg, mesh_device.get(), a_buf, b_buf, c_buf, Mt, Kt, Nt, compute_path);
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        // Warmup: untimed.
        for (int w = 0; w < args.warmup; ++w) {
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }
        distributed::Finish(cq);

        // Timed loop. Layer C dispatches the workload args.n_gemms times per
        // sample to model the byte-decomposition recipe (25 for q36, 36 for
        // q48); Layers A/B dispatch once.
        const int dispatches_per_iter = (args.layer == "C") ? args.n_gemms : 1;
        std::vector<double> times_ms;
        times_ms.reserve(args.iters);
        for (int it = 0; it < args.iters; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            for (int d = 0; d < dispatches_per_iter; ++d) {
                distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            }
            distributed::Finish(cq);
            auto t1 = std::chrono::steady_clock::now();
            times_ms.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        if (times_ms.empty()) { emit_skip("no samples"); return 4; }
        std::sort(times_ms.begin(), times_ms.end());
        double median = times_ms[times_ms.size() / 2];
        double p10 = percentile(times_ms, 0.10);
        double p90 = percentile(times_ms, 0.90);
        emit_csv(median, p10, p90, arch_name.c_str(), n_cores, "");
        return 0;
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 5;
    }
}
