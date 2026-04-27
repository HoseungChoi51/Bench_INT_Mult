// Tensix BF16 32x32 matmul tile loop. Lifted from
// tt_metal/programming_examples/matmul/matmul_multi_core/kernels/compute/mm.cpp;
// the API is the public mm_init / matmul_tiles / pack_tile sequence so the
// kernel is portable across the TT-Metal versions that expose those calls.
//
// Runtime args (set in host/main.cpp):
//   0: num_output_tiles  — output tiles this core is responsible for
//   1: Kt                — inner-dim tile count
//
// Circular buffers (configured on the host):
//   c_0  : input A tiles  (Float16_b)
//   c_1  : input B tiles  (Float16_b)
//   c_16 : output tiles   (Float16_b)

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

void kernel_main() {
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(0);
    const uint32_t Kt = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_16;

    mm_init(cb_in0, cb_in1, cb_out);

    for (uint32_t i = 0; i < num_output_tiles; ++i) {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            cb_wait_front(cb_in0, 1);
            cb_wait_front(cb_in1, 1);
            matmul_tiles(cb_in0, cb_in1, /*tile_a=*/0, /*tile_b=*/0, /*dst=*/0);
            cb_pop_front(cb_in0, 1);
            cb_pop_front(cb_in1, 1);
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
    }
}
