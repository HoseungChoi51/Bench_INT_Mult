// Tensix SFPU INT32 fused-mul-add benchmark host driver.
//
// Pairs with scripts/bench_blackhole_sfpu_int32_fma.py — one invocation
// runs one (n_tiles_per_core) cell: warmup + iters samples, then emits one
// CSV line on stdout.
//
// What it does:
//   - Opens MeshDevice 0, queries the full compute grid (110 cores on a
//     harvested Blackhole p150a — 11x10 after 2 cores are harvested from
//     13x10).
//   - Allocates 3 Int32 input DRAM buffers (A, B, C) and 1 Int32 output
//     buffer (D), each sized n_tiles_total * 4096 B.
//   - Builds a Program with a 3-stream reader (sfpu_reader_three.cpp), the
//     existing single-stream writer (writer.cpp, reads c_16), and the
//     fused mul+add compute kernel (compute_sfpu_int32_fma.cpp).
//   - Warms up, then times `iters` enqueues with chrono::steady_clock,
//     reading back D and validating a sample against host-computed
//     `a*b + c` on bounded inputs.
//
// Useful-ops counting (in the Python wrapper, not here): per-element 1
// SFPU mul + 1 SFPU add → 2 ops. useful_ops = 2 * n_tiles_total * 1024.
//
// Stdout (one CSV line, this exact column order):
//
//     median_ms,p10_ms,p90_ms,arch,n_cores,gate,err
//
// On error: numeric fields = "null", gate = "skipped", err carries the
// exception message.

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

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

#ifndef KERNEL_DIR
#define KERNEL_DIR ""
#endif

namespace {

struct Args {
    uint32_t n_tiles_per_core = 64;  // vector size dial: per-core tiles
    int warmup = 5;
    int iters = 30;
    int gate_sample = 256;  // host validation: # elements to check
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
        if (take("--n-tiles", tmp)) { a.n_tiles_per_core = std::stoul(tmp); continue; }
        if (take("--warmup", tmp))  { a.warmup = std::stoi(tmp); continue; }
        if (take("--iters", tmp))   { a.iters = std::stoi(tmp); continue; }
        if (take("--gate-sample", tmp)) { a.gate_sample = std::stoi(tmp); continue; }
    }
    return a;
}

// Emit one CSV row. gate ∈ {"passed","failed","skipped"}.
void emit_csv(double median_ms, double p10_ms, double p90_ms,
              const char* arch, int n_cores, const char* gate, const char* err) {
    if (err && *err) {
        std::printf("null,null,null,%s,%d,%s,%s\n",
                    arch ? arch : "?", n_cores, gate ? gate : "skipped", err);
    } else {
        std::printf("%.6f,%.6f,%.6f,%s,%d,%s,\n",
                    median_ms, p10_ms, p90_ms, arch ? arch : "?", n_cores,
                    gate ? gate : "passed");
    }
}

void emit_skip(const char* err) { emit_csv(0, 0, 0, "?", 0, "skipped", err); }

double percentile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    if (v.empty()) return 0.0;
    size_t idx = static_cast<size_t>(q * (v.size() - 1));
    return v[idx];
}

// Tile layout: 32x32 = 1024 INT32 lanes laid out as 4 16x16 faces in
// TL/TR/BL/BR row-major order — matches how the matrix engine's tilizer
// arranges BF16 tiles. For pure SFPU eltwise the in-tile order doesn't
// affect correctness (same permutation A → A^perm, B → B^perm, C →
// C^perm ⇒ output is also permuted; readback uses the same permutation
// so element-wise equality holds without explicit untilize). We still
// fill linearly, then read back linearly.

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::shared_ptr<distributed::MeshDevice> mesh_device;
    try {
        mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 2;
    }

    int n_cores = 0;
    CoreCoord grid;
    try {
        grid = mesh_device->compute_with_storage_grid_size();
        n_cores = static_cast<int>(grid.x * grid.y);
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 2;
    }

    if (args.n_tiles_per_core == 0) {
        emit_skip("--n-tiles must be > 0");
        return 3;
    }

    const uint32_t n_tiles_total =
        args.n_tiles_per_core * static_cast<uint32_t>(n_cores);
    constexpr uint32_t kTileBytes = sizeof(int32_t) * TILE_HW;  // 4096
    constexpr uint32_t kElemsPerTile = TILE_HW;                  // 1024

    try {
        // --- DRAM buffers ------------------------------------------------
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = kTileBytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig buf_cfg{
            .size = static_cast<uint32_t>(kTileBytes * n_tiles_total)};

        auto a_buf = distributed::MeshBuffer::create(buf_cfg, dram_cfg, mesh_device.get());
        auto b_buf = distributed::MeshBuffer::create(buf_cfg, dram_cfg, mesh_device.get());
        auto c_buf = distributed::MeshBuffer::create(buf_cfg, dram_cfg, mesh_device.get());
        auto d_buf = distributed::MeshBuffer::create(buf_cfg, dram_cfg, mesh_device.get());

        // --- host data: bounded INT32 so a*b+c can't overflow -----------
        // |a|,|b| ≤ 2^14, |c| ≤ 2^30 ⇒ |a*b+c| < 2^29 + 2^30 < 2^31 (fits int32).
        std::mt19937 rng(20260429u);
        std::uniform_int_distribution<int32_t> dist_ab(-(1 << 14), (1 << 14));
        std::uniform_int_distribution<int32_t> dist_c (-(1 << 30), (1 << 30));

        const size_t n_elems = static_cast<size_t>(n_tiles_total) * kElemsPerTile;
        std::vector<int32_t> a_host(n_elems), b_host(n_elems), c_host(n_elems);
        for (size_t i = 0; i < n_elems; ++i) {
            a_host[i] = dist_ab(rng);
            b_host[i] = dist_ab(rng);
            c_host[i] = dist_c(rng);
        }

        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, c_buf, c_host, /*blocking=*/false);

        // --- build program ----------------------------------------------
        Program program{};
        CoreRange all_cores({0, 0}, {grid.x - 1, grid.y - 1});

        constexpr uint32_t cb_depth = 4;  // quadruple-buffer to cover the
                                          // 3-stream NoC read latency.
        constexpr DataFormat fmt = DataFormat::Int32;

        for (uint32_t cb : {static_cast<uint32_t>(CBIndex::c_0),
                            static_cast<uint32_t>(CBIndex::c_1),
                            static_cast<uint32_t>(CBIndex::c_2),
                            static_cast<uint32_t>(CBIndex::c_16)}) {
            auto cb_idx = static_cast<CBIndex>(cb);
            CreateCircularBuffer(
                program, all_cores,
                CircularBufferConfig(cb_depth * kTileBytes, {{cb_idx, fmt}})
                    .set_page_size(cb_idx, kTileBytes));
        }

        std::vector<uint32_t> reader_cta;
        TensorAccessorArgs(*a_buf).append_to(reader_cta);
        TensorAccessorArgs(*b_buf).append_to(reader_cta);
        TensorAccessorArgs(*c_buf).append_to(reader_cta);
        auto reader_kid = CreateKernel(
            program, KERNEL_DIR "/sfpu_reader_three.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                               .noc = NOC::RISCV_1_default,
                               .compile_args = reader_cta});

        std::vector<uint32_t> writer_cta;
        TensorAccessorArgs(*d_buf).append_to(writer_cta);
        auto writer_kid = CreateKernel(
            program, KERNEL_DIR "/writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                               .noc = NOC::RISCV_0_default,
                               .compile_args = writer_cta});

        auto compute_kid = CreateKernel(
            program, KERNEL_DIR "/compute_sfpu_int32_fma.cpp", all_cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                // Required: INT32 SFPU operands need 32-bit dst slots.
                .fp32_dest_acc_en = true,
                .compile_args = {},
            });

        // Per-core runtime args: each core handles `n_tiles_per_core`
        // contiguous tiles starting at (linear_core_id * tiles_per_core).
        for (uint32_t cy = 0; cy < grid.y; ++cy) {
            for (uint32_t cx = 0; cx < grid.x; ++cx) {
                CoreCoord core{cx, cy};
                const uint32_t linear = cy * grid.x + cx;
                const uint32_t start = linear * args.n_tiles_per_core;
                const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
                const uint32_t b_addr = static_cast<uint32_t>(b_buf->address());
                const uint32_t c_addr = static_cast<uint32_t>(c_buf->address());
                const uint32_t d_addr = static_cast<uint32_t>(d_buf->address());
                SetRuntimeArgs(program, reader_kid, core,
                               {a_addr, b_addr, c_addr, args.n_tiles_per_core, start});
                SetRuntimeArgs(program, writer_kid, core,
                               {d_addr, args.n_tiles_per_core, start});
                SetRuntimeArgs(program, compute_kid, core,
                               {args.n_tiles_per_core});
            }
        }

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        // --- warmup -----------------------------------------------------
        for (int w = 0; w < args.warmup; ++w) {
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }
        distributed::Finish(cq);

        // --- timed loop -------------------------------------------------
        std::vector<double> times_ms;
        times_ms.reserve(args.iters);
        for (int it = 0; it < args.iters; ++it) {
            auto t0 = std::chrono::steady_clock::now();
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
            distributed::Finish(cq);
            auto t1 = std::chrono::steady_clock::now();
            times_ms.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        if (times_ms.empty()) { emit_skip("no samples"); return 4; }
        std::sort(times_ms.begin(), times_ms.end());
        const double median = times_ms[times_ms.size() / 2];
        const double p10 = percentile(times_ms, 0.10);
        const double p90 = percentile(times_ms, 0.90);

        // --- correctness gate (sample read-back) ------------------------
        // The output buffer was filled by the timed loop; readback compares
        // a small sample against host `a*b+c`. Tile in-memory permutation
        // is identical for the linearly-uploaded inputs and outputs (the
        // SFPU is element-wise — no cross-lane shuffles), so a direct
        // index-by-index comparison works.
        const char* gate = "passed";
        try {
            std::vector<int32_t> d_host(n_elems);
            distributed::EnqueueReadMeshBuffer(cq, d_host, d_buf, /*blocking=*/true);
            int n_check = std::min<int>(args.gate_sample, static_cast<int>(n_elems));
            for (int k = 0; k < n_check; ++k) {
                size_t i = (static_cast<size_t>(k) * 1009) % n_elems;
                int32_t expected =
                    static_cast<int32_t>(a_host[i] * b_host[i] + c_host[i]);
                if (d_host[i] != expected) {
                    std::fprintf(stderr,
                                 "gate mismatch at i=%zu: dev=%d host=%d "
                                 "(a=%d b=%d c=%d)\n",
                                 i, d_host[i], expected,
                                 a_host[i], b_host[i], c_host[i]);
                    gate = "failed";
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "gate readback error: %s\n", e.what());
            gate = "skipped";
        }

        emit_csv(median, p10, p90, "blackhole", n_cores, gate, "");
        return 0;
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 5;
    }
}
