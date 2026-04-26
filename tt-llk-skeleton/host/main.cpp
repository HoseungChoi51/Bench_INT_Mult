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
//           wall-clock includes all 25 dispatches. The on-device output is
//           BF16 (tt_llk_bf16) and is *not* used as the modular result; the
//           recipe's correctness is checked once on host bigint reference
//           inside bench_blackhole.py. The Blackhole row therefore reports
//           "exact 36-bit modmul throughput estimated from the cost of the
//           25-GEMM cascade", documented in device_detail.note.
//
// Build via the sibling Makefile / host/CMakeLists.txt against an installed
// tt-metalium (built from /home/chs/TT/tt-metal/build_Release on this host).
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
};

// Number of GEMM dispatches per measured iteration of Layer C — matches the
// 5x5 byte-decomposition recipe used by the NVIDIA path.
constexpr int kLayerCGemmsPerIter = 25;

// 32x32 tile.
constexpr uint32_t kTile = TILE_HEIGHT;

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
    }
    return a;
}

const char* compute_kernel_for(const std::string& backend) {
    if (backend == "tt_llk_bf16") return KERNEL_DIR "/compute_bf16_mma.cpp";
    if (backend == "tt_llk_int8") return KERNEL_DIR "/compute_int8_mma.cpp";
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

// Build one BF16 matmul Program against the given inputs/outputs. Modeled on
// tt_metal/programming_examples/matmul/matmul_multi_core; same kernel
// structure (reader / writer / mm.cpp-style compute) but with our skeleton's
// kernel files (absolute paths via KERNEL_DIR).
Program build_bf16_matmul_program(
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

    constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HW;  // 2 * 32 * 32 = 2048
    constexpr DataFormat fmt = DataFormat::Float16_b;

    constexpr uint32_t cb_depth = 2;  // double-buffered

    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(cb_depth * single_tile_bytes, {{CBIndex::c_0, fmt}})
            .set_page_size(CBIndex::c_0, single_tile_bytes));
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(cb_depth * single_tile_bytes, {{CBIndex::c_1, fmt}})
            .set_page_size(CBIndex::c_1, single_tile_bytes));
    CreateCircularBuffer(
        program, all_cores,
        CircularBufferConfig(cb_depth * single_tile_bytes, {{CBIndex::c_16, fmt}})
            .set_page_size(CBIndex::c_16, single_tile_bytes));

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
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {}});

    uint32_t work_offset = 0;
    auto work_groups = {
        std::make_pair(core_group_1, work_per_core1),
        std::make_pair(core_group_2, work_per_core2)};
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // DeviceAddr is a 64-bit type; explicitly truncate to the
                // 32-bit runtime-arg slot. Buffers fit comfortably under 4 GiB
                // on this device, so the cast is safe for our shapes.
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
        // Capture device metadata for the JSONL device_detail.
        const auto cg = mesh_device->compute_with_storage_grid_size();
        n_cores = static_cast<int>(cg.x * cg.y);
        // Best-effort arch name; ARCH_NAME enum stringification is version-dependent.
        // The wrapper substitutes the friendly "blackhole" if this comes back generic.
        arch_name = "blackhole";
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 2;
    }

    const uint32_t Mt = args.M / kTile;
    const uint32_t Kt = args.K / kTile;
    const uint32_t Nt = args.N / kTile;

    // Currently only the BF16 backend has a working compute kernel; the INT8
    // and SFPU FP32 kernels remain TODO (see kernels/*.cpp). Surface a clean
    // skip for unsupported backends so the wrapper marks the row appropriately.
    if (args.backend == "tt_llk_int8" || args.backend == "tt_llk_sfpu_fp32") {
        // mesh_device closes on shared_ptr destruction.
        emit_skip("backend kernel TODO: see tt-llk-skeleton/kernels/*.cpp");
        return 0;
    }

    try {
        // Allocate buffers + fill with deterministic random data once.
        constexpr uint32_t single_tile_bytes = sizeof(bfloat16) * TILE_HW;
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = single_tile_bytes, .buffer_type = BufferType::DRAM};

        auto make_buf = [&](uint32_t n_tiles) {
            distributed::ReplicatedBufferConfig bcfg{.size = single_tile_bytes * n_tiles};
            return distributed::MeshBuffer::create(bcfg, dram_cfg, mesh_device.get());
        };
        auto a_buf = make_buf(Mt * Kt);
        auto b_buf = make_buf(Kt * Nt);
        auto c_buf = make_buf(Mt * Nt);

        std::mt19937 rng(20260427u);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<bfloat16> a_host(static_cast<size_t>(args.M) * args.K);
        std::vector<bfloat16> b_host(static_cast<size_t>(args.K) * args.N);
        for (auto& v : a_host) v = bfloat16(dist(rng));
        for (auto& v : b_host) v = bfloat16(dist(rng));
        // Tilize: convert from row-major to the device's 32x32 tile-row-major layout.
        a_host = tilize_nfaces(a_host, args.M, args.K);
        b_host = tilize_nfaces(b_host, args.K, args.N);

        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);

        Program program = build_bf16_matmul_program(
            mesh_device.get(), a_buf, b_buf, c_buf, Mt, Kt, Nt, compute_path);
        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        // Warmup: untimed.
        for (int w = 0; w < args.warmup; ++w) {
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }
        distributed::Finish(cq);

        // Timed loop. Layer C dispatches the workload kLayerCGemmsPerIter
        // times per sample to model the 25-GEMM byte-decomposition recipe;
        // Layers A/B dispatch once.
        const int dispatches_per_iter = (args.layer == "C") ? kLayerCGemmsPerIter : 1;
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
