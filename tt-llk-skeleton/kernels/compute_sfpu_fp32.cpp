// SFPU FP32 vector path. Diagnostic backend per BENCHMARK.md §3 — the
// matrix engine is not used here; arithmetic goes through the Tensix
// SFPU vector unit operating on fp32 lanes.
//
// Layer B only in v1. Layer C through SFPU is theoretically possible
// (FP32 has enough precision for 36-bit modmul if you're careful with
// rounding) but is deferred to v2.

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
// TODO(user): include the SFPU eltwise multiply / FMA headers your
// TT-Metal version exposes, e.g.:
//   #include "compute_kernel_api/eltwise_binary/eltwise_binary.h"

namespace NAMESPACE {

constexpr uint32_t cb_a   = tt::CBIndex::c_0;
constexpr uint32_t cb_b   = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

void MAIN {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    // TODO(user): SFPU GEMM is not a single instruction — it's a tile-by-
    // tile FMA loop using sfpu_mul / sfpu_add primitives. This stub just
    // copies A → out so the build succeeds; real numbers require the
    // SFPU FMA loop.
    (void)Mt; (void)Kt; (void)Nt; (void)cb_b;

    for (uint32_t i = 0; i < Mt * Nt; ++i) {
        cb_wait_front(cb_a, 1);
        cb_reserve_back(cb_out, 1);
        copy_tile_init();
        copy_tile(cb_a, 0, 0);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        cb_pop_front(cb_a, 1);
    }
}

}  // namespace NAMESPACE
