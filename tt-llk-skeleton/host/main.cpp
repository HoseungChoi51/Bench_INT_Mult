// Tenstorrent Blackhole bench host driver.
//
// Pairs with the NVIDIA-side scripts/bench_nvidia.py. Dispatches one of
// {layer A probe, layer B raw GEMM, layer C q36 INT8 modmul} per
// invocation, runs warmup + iters, and prints a single CSV line on
// stdout that bench_blackhole.py parses and converts to a JSONL record.
//
// Build via the sibling Makefile (which uses host/CMakeLists.txt).
//
// CLI (one shot per dispatch — bench_blackhole.py loops over backend×size):
//
//     bench_blackhole \
//         --backend tt_llk_int8 | tt_llk_bf16 | tt_llk_sfpu_fp32 \
//         --layer A | B | C \
//         --M <int> --K <int> --N <int> \
//         --warmup <n> --iters <n> \
//         [--q36 <decimal>]   // required for layer C
//
// stdout (CSV, one line, in this exact order):
//
//     median_ms,p10_ms,p90_ms,arch,n_cores,err
//
// On error: median_ms is the literal string "null", err is non-empty.
//
// The compute kernel selection is driven by --backend; the modular
// reduction epilogue (Layer C) is only enabled when --layer is "C".

#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>

// TT-Metal includes. Adjust per your installed layout if the bundled
// CMake config does not bring these in.
#if __has_include(<tt-metalium/host_api.hpp>)
    #include <tt-metalium/host_api.hpp>
    #include <tt-metalium/device.hpp>
    #include <tt-metalium/program.hpp>
    #include <tt-metalium/buffer.hpp>
    #include <tt-metalium/circular_buffer.hpp>
#elif __has_include(<tt_metal/host_api.hpp>)
    // Older layout
    #include <tt_metal/host_api.hpp>
    #include <tt_metal/device.hpp>
    #include <tt_metal/program.hpp>
    #include <tt_metal/buffer.hpp>
#else
    #error "TT-Metal headers not found. Adjust include path or update CMakeLists.txt."
#endif

namespace tt_metal = ::tt::tt_metal;

// ---------------------------------------------------------------------------

struct Args {
    std::string backend = "tt_llk_int8";
    std::string layer   = "A";
    int M = 256, K = 256, N = 256;
    int warmup = 5, iters = 30;
    uint64_t q36 = 0xFFFF00001ULL;  // matches scripts/_bench_common.py::q36_ntt_friendly_prime
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto take = [&](const char* k, std::string& dst) {
            if (s == k && i + 1 < argc) { dst = argv[++i]; return true; }
            return false;
        };
        std::string tmp;
        if (take("--backend", a.backend)) continue;
        if (take("--layer",   a.layer))   continue;
        if (take("--M", tmp)) { a.M = std::stoi(tmp); continue; }
        if (take("--K", tmp)) { a.K = std::stoi(tmp); continue; }
        if (take("--N", tmp)) { a.N = std::stoi(tmp); continue; }
        if (take("--warmup", tmp)) { a.warmup = std::stoi(tmp); continue; }
        if (take("--iters",  tmp)) { a.iters  = std::stoi(tmp); continue; }
        if (take("--q36",    tmp)) { a.q36 = std::stoull(tmp); continue; }
    }
    return a;
}

static const char* compute_kernel_for(const std::string& backend) {
    if (backend == "tt_llk_int8")      return "kernels/compute_int8_mma.cpp";
    if (backend == "tt_llk_bf16")      return "kernels/compute_bf16_mma.cpp";
    if (backend == "tt_llk_sfpu_fp32") return "kernels/compute_sfpu_fp32.cpp";
    return nullptr;
}

// ---------------------------------------------------------------------------

static void emit_csv(double median_ms, double p10_ms, double p90_ms,
                     const char* arch, int n_cores, const char* err) {
    if (err && *err) {
        std::printf("null,null,null,%s,%d,%s\n", arch ? arch : "?", n_cores, err);
    } else {
        std::printf("%.6f,%.6f,%.6f,%s,%d,\n", median_ms, p10_ms, p90_ms,
                    arch ? arch : "?", n_cores);
    }
}

static void emit_skip(const char* err) { emit_csv(0, 0, 0, "?", 0, err); }

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const char* compute = compute_kernel_for(args.backend);
    if (!compute) { emit_skip("unknown backend"); return 1; }

    // === Open device ===
    int device_id = 0;
    tt_metal::IDevice* device = nullptr;
    try {
        device = tt_metal::CreateDevice(device_id);
    } catch (const std::exception& e) {
        emit_skip(e.what());
        return 2;
    }

    const auto arch = device->arch();
    const std::string arch_str = std::to_string(static_cast<int>(arch));  // human-readable name varies per version
    const int n_cores = static_cast<int>(device->compute_with_storage_grid_size().x *
                                         device->compute_with_storage_grid_size().y);

    // === Build program ===
    constexpr uint32_t TILE = 32;
    if (args.M % TILE != 0 || args.K % TILE != 0 || args.N % TILE != 0) {
        tt_metal::CloseDevice(device);
        emit_skip("M/K/N must be multiples of TILE=32");
        return 3;
    }
    const uint32_t Mt = args.M / TILE;
    const uint32_t Kt = args.K / TILE;
    const uint32_t Nt = args.N / TILE;

    // TODO(user): fill in the actual program-construction calls. The
    // skeleton below shows the *shape* — the precise cmake-exported API
    // (CreateProgram, CreateKernel, CreateCircularBuffer, SetRuntimeArgs,
    // EnqueueProgram, Finish) varies between TT-Metal versions; consult
    // tt-metal/programming_examples/matmul_multi_core for the exact
    // calls in your installed version.
    //
    // Pseudocode:
    //
    //     auto program = tt_metal::CreateProgram();
    //     auto core_set = CoreRange{{0,0},{Nx-1,Ny-1}};
    //     auto reader = CreateKernel(program, "kernels/reader.cpp", core_set,
    //         DataMovementConfig{...DRAM_to_L1...});
    //     auto writer = CreateKernel(program, "kernels/writer.cpp", core_set,
    //         DataMovementConfig{...L1_to_DRAM...});
    //     auto comp = CreateKernel(program, compute, core_set,
    //         ComputeConfig{...with compile-time defs for {Mt,Kt,Nt,LAYER,Q36}...});
    //     // CBs: one per operand, sized for double-buffered tiles.
    //     CreateCircularBuffer(...); CreateCircularBuffer(...); CreateCircularBuffer(...);
    //     // DRAM-resident input/output buffers, deterministically initialized.
    //     auto a_dram = CreateBuffer(...); auto b_dram = CreateBuffer(...); auto c_dram = CreateBuffer(...);
    //     SetRuntimeArgs(program, reader, core_set, {a_dram->address(), b_dram->address(), Mt, Kt, Nt});
    //     SetRuntimeArgs(program, writer, core_set, {c_dram->address(), Mt, Nt});
    //     SetRuntimeArgs(program, comp,   core_set, {Mt, Kt, Nt, args.layer == "C" ? 1 : 0, (uint32_t)(args.q36 & 0xFFFFFFFF), (uint32_t)(args.q36 >> 32)});
    //
    // For now we just go straight to the timing loop with a no-op
    // program; the wrapper script will see a "skipped" record because
    // the timer reports 0.0 ms which fails the plausibility check.

    auto cq = device->command_queue(0);  // adjust per API

    // === Warmup ===
    for (int i = 0; i < args.warmup; ++i) {
        // tt_metal::EnqueueProgram(cq, program, /*blocking=*/false);
        (void)cq;
    }
    tt_metal::Finish(cq);

    // === Timed loop ===
    std::vector<double> times_ms;
    times_ms.reserve(args.iters);
    for (int i = 0; i < args.iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        // tt_metal::EnqueueProgram(cq, program, /*blocking=*/false);
        tt_metal::Finish(cq);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    // === Cleanup ===
    tt_metal::CloseDevice(device);

    // === Stats ===
    if (times_ms.empty()) { emit_skip("no samples"); return 4; }
    std::sort(times_ms.begin(), times_ms.end());
    const size_t n = times_ms.size();
    const double median = times_ms[n / 2];
    const double p10 = times_ms[std::max<size_t>(0, static_cast<size_t>(0.10 * n))];
    const double p90 = times_ms[std::min<size_t>(n - 1, static_cast<size_t>(0.90 * n))];

    emit_csv(median, p10, p90, arch_str.c_str(), n_cores, "");
    return 0;
}
