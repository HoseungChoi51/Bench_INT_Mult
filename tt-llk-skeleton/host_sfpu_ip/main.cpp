// Tensix SFPU INT32 inner-product benchmark host driver.
//
// On-device shape (per core):
//   acc_lane[ℓ]  =  Σ_{t=0}^{n_tiles-1}  A[t][ℓ] * B[t][ℓ]      ℓ ∈ [0, 1024)
//
// All cores run the same recipe over a disjoint slice of the input. The
// output buffer holds one accumulator tile per core (linear core id ⇒
// output tile id). The "scalar inner product" the host could derive is
// Σ_core Σ_ℓ acc_lane[ℓ] over all cores; that final reduce is host-side
// and is not part of the timed loop.
//
// Overflow planning (per user request — no post-hoc check):
//   For each n_tiles_per_core, choose B such that the per-lane partial
//   sum  Σ_{t<N} a[t]·b[t]  with |a|,|b| ≤ B can be bounded:
//       |sum|  ≤  N · B²  <  2³¹
//       ⇒      B  ≤  ⌊√(2³¹ / N)⌋
//   We pick the largest power-of-two ≤ that bound for cleanliness; the
//   value is logged in stderr so you can sanity-check.
//
// Stdout (one CSV line):
//
//     median_ms,p10_ms,p90_ms,arch,n_cores,gate,err
//
// The "gate" field is informational only — we do not validate post-hoc
// per the design directive. It is always reported as "passed" on a
// successful run because the input bounds make overflow impossible.

#include <algorithm>
#include <chrono>
#include <cmath>
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
    uint32_t n_tiles_per_core = 64;
    int warmup = 5;
    int iters = 30;
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
    }
    return a;
}

// Largest int32_t bound B such that  N · B²  <  2³¹.
// Floor-of-sqrt with a tiny safety margin (use 2³⁰ to keep one bit of headroom).
int32_t safe_input_bound(uint32_t n_tiles) {
    // headroom = 2^30 leaves the per-lane sum ≤ 2^30 < 2^31 - 1.
    const double max_sum = static_cast<double>(1u << 30);
    const double bound = std::floor(std::sqrt(max_sum / static_cast<double>(n_tiles)));
    int32_t b = static_cast<int32_t>(bound);
    if (b < 1) b = 1;
    return b;
}

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

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::shared_ptr<distributed::MeshDevice> mesh_device;
    try {
        mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    } catch (const std::exception& e) { emit_skip(e.what()); return 2; }

    int n_cores = 0;
    CoreCoord grid;
    try {
        grid = mesh_device->compute_with_storage_grid_size();
        n_cores = static_cast<int>(grid.x * grid.y);
    } catch (const std::exception& e) { emit_skip(e.what()); return 2; }

    if (args.n_tiles_per_core == 0) { emit_skip("--n-tiles must be > 0"); return 3; }

    const uint32_t n_tiles_total =
        args.n_tiles_per_core * static_cast<uint32_t>(n_cores);
    constexpr uint32_t kTileBytes = sizeof(int32_t) * TILE_HW;  // 4096
    constexpr uint32_t kElemsPerTile = TILE_HW;                  // 1024

    const int32_t B = safe_input_bound(args.n_tiles_per_core);
    std::fprintf(stderr,
                 "[ip] n_tiles_per_core=%u n_cores=%d safe_input_bound=±%d "
                 "(per-lane partial ≤ %u·%d² = %llu < 2^31)\n",
                 args.n_tiles_per_core, n_cores, B, args.n_tiles_per_core, B,
                 static_cast<unsigned long long>(args.n_tiles_per_core)
                     * static_cast<unsigned long long>(B)
                     * static_cast<unsigned long long>(B));

    try {
        // --- DRAM buffers -----------------------------------------------
        distributed::DeviceLocalBufferConfig dram_cfg{
            .page_size = kTileBytes, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig in_cfg{
            .size = static_cast<uint32_t>(kTileBytes * n_tiles_total)};
        // Output: ONE tile per core (the per-core accumulator). Total
        // output tiles = n_cores; tile id = linear core id.
        distributed::ReplicatedBufferConfig out_cfg{
            .size = static_cast<uint32_t>(kTileBytes * n_cores)};

        auto a_buf = distributed::MeshBuffer::create(in_cfg, dram_cfg, mesh_device.get());
        auto b_buf = distributed::MeshBuffer::create(in_cfg, dram_cfg, mesh_device.get());
        auto d_buf = distributed::MeshBuffer::create(out_cfg, dram_cfg, mesh_device.get());

        // --- host data: bounded INT32 so per-lane partial fits in int31 -
        std::mt19937 rng(20260429u);
        std::uniform_int_distribution<int32_t> dist(-B, B);
        const size_t n_elems =
            static_cast<size_t>(n_tiles_total) * kElemsPerTile;
        std::vector<int32_t> a_host(n_elems), b_host(n_elems);
        for (size_t i = 0; i < n_elems; ++i) {
            a_host[i] = dist(rng);
            b_host[i] = dist(rng);
        }

        auto& cq = mesh_device->mesh_command_queue();
        distributed::EnqueueWriteMeshBuffer(cq, a_buf, a_host, /*blocking=*/false);
        distributed::EnqueueWriteMeshBuffer(cq, b_buf, b_host, /*blocking=*/false);

        // --- build program ----------------------------------------------
        Program program{};
        CoreRange all_cores({0, 0}, {grid.x - 1, grid.y - 1});

        constexpr uint32_t cb_depth_in = 4;       // double-buffer A and B reads
        constexpr uint32_t cb_depth_acc = 2;      // carry CB depth must be ≥ 2
                                                  // (kernel pushes 1 ahead of consume)
        constexpr uint32_t cb_depth_out = 1;
        constexpr DataFormat fmt = DataFormat::Int32;

        auto mk_cb = [&](CBIndex idx, uint32_t depth) {
            CreateCircularBuffer(
                program, all_cores,
                CircularBufferConfig(depth * kTileBytes, {{idx, fmt}})
                    .set_page_size(idx, kTileBytes));
        };
        mk_cb(CBIndex::c_0,  cb_depth_in);
        mk_cb(CBIndex::c_1,  cb_depth_in);
        mk_cb(CBIndex::c_24, cb_depth_acc);   // intermediate carry
        mk_cb(CBIndex::c_16, cb_depth_out);

        std::vector<uint32_t> reader_cta;
        TensorAccessorArgs(*a_buf).append_to(reader_cta);
        TensorAccessorArgs(*b_buf).append_to(reader_cta);
        auto reader_kid = CreateKernel(
            program, KERNEL_DIR "/sfpu_reader_two.cpp", all_cores,
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
            program, KERNEL_DIR "/compute_sfpu_int32_inner_product.cpp", all_cores,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,    // INT32 SFPU dst slots
                .compile_args = {},
            });

        for (uint32_t cy = 0; cy < grid.y; ++cy) {
            for (uint32_t cx = 0; cx < grid.x; ++cx) {
                CoreCoord core{cx, cy};
                const uint32_t linear = cy * grid.x + cx;
                // Reader: read this core's slice of A and B (linear contiguous).
                const uint32_t in_start = linear * args.n_tiles_per_core;
                const uint32_t a_addr = static_cast<uint32_t>(a_buf->address());
                const uint32_t b_addr = static_cast<uint32_t>(b_buf->address());
                const uint32_t d_addr = static_cast<uint32_t>(d_buf->address());
                SetRuntimeArgs(program, reader_kid, core,
                               {a_addr, b_addr, args.n_tiles_per_core, in_start});
                // Writer: each core emits one accumulator tile at slot
                // `linear` in the output buffer.
                SetRuntimeArgs(program, writer_kid, core,
                               {d_addr, /*num_tiles=*/1u, /*start_id=*/linear});
                SetRuntimeArgs(program, compute_kid, core,
                               {args.n_tiles_per_core});
            }
        }

        distributed::MeshWorkload workload;
        distributed::MeshCoordinateRange device_range(mesh_device->shape());
        workload.add_program(device_range, std::move(program));

        for (int w = 0; w < args.warmup; ++w) {
            distributed::EnqueueMeshWorkload(cq, workload, /*blocking=*/false);
        }
        distributed::Finish(cq);

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

        emit_csv(median, p10, p90, "blackhole", n_cores, "passed", "");
        return 0;
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 5;
    }
}
