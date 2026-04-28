// Tensix SFPU INT32 fused mul+add: per-element  d = a * b + c.
//
// One tile (32x32 = 1024 INT32 lanes) per loop. Two SFPU ops per element
// (mul_int + add_int) — counted as 2 useful ops in the bench.
//
// CBs (configured on the host with DataFormat::Int32, 4096 B/tile):
//   c_0  : input A
//   c_1  : input B
//   c_2  : input C
//   c_16 : output D
//
// Runtime args:
//   0: n_tiles  (per-core)
//
// Notes on the DST budget: the API doc for mul_int_tile / add_int_tile says
// "for 32 bit formats" the DST register can hold up to 4 tile slots
// (2 per operand). We use slots 0/1/2 simultaneously — within budget.
// fp32_dest_acc_en MUST be true on the host ComputeConfig for INT32 dst.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/add_int_sfpu.h"

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_a   = tt::CBIndex::c_0;
    constexpr auto cb_b   = tt::CBIndex::c_1;
    constexpr auto cb_c   = tt::CBIndex::c_2;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_a, cb_out);
    mul_int_tile_init<DataFormat::Int32>();
    add_int_tile_init();

    for (uint32_t i = 0; i < n_tiles; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_wait_front(cb_c, 1);

        tile_regs_acquire();
        copy_tile(cb_a, /*tile_idx=*/0, /*dst_idx=*/0);  // A → DST[0]
        copy_tile(cb_b, /*tile_idx=*/0, /*dst_idx=*/1);  // B → DST[1]
        copy_tile(cb_c, /*tile_idx=*/0, /*dst_idx=*/2);  // C → DST[2]

        mul_int_tile<DataFormat::Int32>(/*idst0=*/0, /*idst1=*/1, /*odst=*/0);
        add_int_tile<DataFormat::Int32>(/*idst0=*/0, /*idst1=*/2, /*odst=*/0);

        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_out, 1);
        pack_tile(/*dst_idx=*/0, cb_out);
        cb_push_back(cb_out, 1);

        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_pop_front(cb_c, 1);
        tile_regs_release();
    }
}
