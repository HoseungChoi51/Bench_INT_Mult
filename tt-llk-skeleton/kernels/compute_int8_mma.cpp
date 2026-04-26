// Tensix Compute kernel: INT8 32×32 MMA tile, with optional q36 modular
// reduction epilogue for Layer C.
//
// Compile-time defines (set in host main.cpp via ComputeConfig):
//   Mt, Kt, Nt   — output tile counts
//   LAYER_C      — 0 for Layer B (raw GEMM), 1 for Layer C (modmul)
//   Q36_LO, Q36_HI — 32-bit halves of the q36 prime (only used if LAYER_C)
//
// CB indices (must match host setup):
//   c_0  : input A tiles (int8, 32×32)
//   c_1  : input B tiles (int8, 32×32)
//   c_16 : output tiles (int32, 32×32)
//
// References:
//   tt-metal/programming_examples/matmul_multi_core/kernels/compute.cpp
//   (BF16 version; the INT8 path uses matmul_tiles with int-format CBs.)

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
// TODO(user): Some TT-Metal versions split SFPU intrinsics across multiple
// headers; if you need them for the modular epilogue, add includes such as
// "compute_kernel_api/eltwise_unary/sfpu.h" or
// "compute_kernel_api/eltwise_binary/eltwise_binary.h" here.

namespace NAMESPACE {

constexpr uint32_t cb_a   = tt::CBIndex::c_0;
constexpr uint32_t cb_b   = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

void MAIN {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);
    const uint32_t LAYER_C = get_compile_time_arg_val(3);
    // Q36 lo/hi reserved for compile-time arg slots 4 and 5; the modular
    // epilogue currently goes through SFPU and reads them from runtime
    // args if needed.

    // -------------------------------------------------------------------
    // Init MMA. The exact init call depends on TT-Metal version; common
    // names are mm_init / matmul_init / matmul_init_short.
    // -------------------------------------------------------------------
    mm_init(cb_a, cb_b, cb_out);

    // -------------------------------------------------------------------
    // Main GEMM loop.
    // -------------------------------------------------------------------
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_a, 1);
                cb_wait_front(cb_b, 1);

                // TODO(user): the INT8 path takes the same matmul_tiles
                // shape as BF16 but with int-format CBs. Confirm against
                // your installed TT-Metal version's `matmul.h`.
                matmul_tiles(cb_a, cb_b, /*tile_a=*/0, /*tile_b=*/0,
                             /*dst_tile=*/0, /*transpose=*/false);

                cb_pop_front(cb_a, 1);
                cb_pop_front(cb_b, 1);
            }

            // ----- Epilogue (Layer B or Layer C) ---------------------------
            if constexpr (LAYER_C) {
                // TODO(user): Layer C modular epilogue.
                //
                // The 25-INT8-MMA decomposition produces, for each output
                // element, an int32 partial that must be:
                //   1. Lifted to int64 (or two int32 halves on SFPU).
                //   2. Scaled by 2^(8(i+j)) mod q36, reduced mod q36 per pair.
                //   3. Summed across the 5×5 grid, reduced mod q36.
                //
                // The host driver loops 25× over (i,j); the kernel sees a
                // single (i,j) pair per program-launch and accumulates into
                // a persistent SRAM scratch buffer between launches.
                //
                // Implementation hints:
                //   - SFPU intrinsics: sfpu_mul_int / sfpu_div_uint /
                //     sfpu_add_int (names vary; check sfpi.h in your install).
                //   - Barrett constants for q36 can be precomputed on the
                //     host and passed via runtime args.
                //
                // Until this is filled in, Layer C reports the raw int32
                // GEMM result, which fails the correctness gate (this is
                // by design — a missing TODO must show up as a failure).
            }
            // -----------------------------------------------------------

            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);

            tile_regs_commit();
            tile_regs_release();
        }
    }
}

}  // namespace NAMESPACE
