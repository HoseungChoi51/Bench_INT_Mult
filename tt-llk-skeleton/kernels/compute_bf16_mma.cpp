// Tensix Compute kernel: BF16 32×32 MMA tile.
//
// Layer B only (no modular reduction). Used as the BF16 row of the
// BENCHMARK.md §4 comparison table. Layer C BF16 is not required by v1.
//
// Compile-time defines:
//   Mt, Kt, Nt   — output tile counts
//
// CB indices:
//   c_0  : input A tiles (bf16, 32×32)
//   c_1  : input B tiles (bf16, 32×32)
//   c_16 : output tiles (fp32, 32×32)
//
// Lifted from tt-metal/programming_examples/matmul_multi_core/kernels/compute.cpp.

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {

constexpr uint32_t cb_a   = tt::CBIndex::c_0;
constexpr uint32_t cb_b   = tt::CBIndex::c_1;
constexpr uint32_t cb_out = tt::CBIndex::c_16;

void MAIN {
    const uint32_t Mt = get_compile_time_arg_val(0);
    const uint32_t Kt = get_compile_time_arg_val(1);
    const uint32_t Nt = get_compile_time_arg_val(2);

    mm_init(cb_a, cb_b, cb_out);

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            tile_regs_acquire();
            tile_regs_wait();
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_wait_front(cb_a, 1);
                cb_wait_front(cb_b, 1);
                matmul_tiles(cb_a, cb_b, 0, 0, 0, false);
                cb_pop_front(cb_a, 1);
                cb_pop_front(cb_b, 1);
            }
            cb_reserve_back(cb_out, 1);
            pack_tile(0, cb_out);
            cb_push_back(cb_out, 1);
            tile_regs_commit();
            tile_regs_release();
        }
    }
}

}  // namespace NAMESPACE
