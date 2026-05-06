// Tensix SFPU INT32 inner product (per-core partial):
//
//   per-core lane-wise sum:  acc[lane] = Σ_t  A[t][lane] * B[t][lane]
//
// for n_tiles tile pairs (A, B) per core. Each tile is 32×32 = 1024 INT32
// lanes; the per-lane partial sum stays in the SFPU side. A final 1024-lane
// reduction across all cores is done host-side after readback.
//
// Memory pattern (per the upstream INT32 cumsum recipe in
// `ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/
// accumulation_compute.cpp`): per-tile acquire/release, with a "carry"
// CB (c_24) that round-trips the accumulator tile through L1 between
// iterations. A single tile_regs_acquire spanning the whole loop is
// fragile for INT32 SFPU work — the upstream code uses the carry-CB
// pattern, so we follow.
//
// CBs (host configures Int32, 4096 B/tile):
//   c_0  : input A
//   c_1  : input B
//   c_24 : intermediate carry (accumulator, looped back into the kernel)
//   c_16 : output (one tile per core, the final accumulator)
//
// Runtime args:
//   0: n_tiles  (>= 1)

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
    constexpr auto cb_acc = tt::CBIndex::c_24;
    constexpr auto cb_out = tt::CBIndex::c_16;

    init_sfpu(cb_a, cb_out);
    mul_int_tile_init<DataFormat::Int32>();
    add_int_tile_init();

    // --- Iter 0: seed accumulator with first product, pack to c_24 ---
    cb_wait_front(cb_a, 1);
    cb_wait_front(cb_b, 1);
    tile_regs_acquire();
    copy_tile(cb_a, /*tile_idx=*/0, /*dst_idx=*/0);
    copy_tile(cb_b, /*tile_idx=*/0, /*dst_idx=*/1);
    mul_int_tile<DataFormat::Int32>(0, 1, 0);  // DST[0] = a[0] * b[0]
    cb_pop_front(cb_a, 1);
    cb_pop_front(cb_b, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_acc, 1);
    pack_tile(/*dst_idx=*/0, cb_acc);
    cb_push_back(cb_acc, 1);
    tile_regs_release();

    // --- Iters 1..n_tiles-1: acc += a[i] * b[i] ---
    for (uint32_t i = 1; i < n_tiles; ++i) {
        cb_wait_front(cb_a, 1);
        cb_wait_front(cb_b, 1);
        cb_wait_front(cb_acc, 1);
        tile_regs_acquire();
        copy_tile(cb_a,   0, 0);
        copy_tile(cb_b,   0, 1);
        copy_tile(cb_acc, 0, 2);
        mul_int_tile<DataFormat::Int32>(0, 1, 0);  // DST[0] = a*b
        add_int_tile<DataFormat::Int32>(0, 2, 0);  // DST[0] = a*b + carry
        cb_pop_front(cb_a, 1);
        cb_pop_front(cb_b, 1);
        cb_pop_front(cb_acc, 1);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_acc, 1);
        pack_tile(0, cb_acc);
        cb_push_back(cb_acc, 1);
        tile_regs_release();
    }

    // --- Final: copy c_24 → c_16 (output is the last accumulator tile) ---
    cb_wait_front(cb_acc, 1);
    tile_regs_acquire();
    copy_tile(cb_acc, 0, 0);
    cb_pop_front(cb_acc, 1);
    tile_regs_commit();
    tile_regs_wait();
    cb_reserve_back(cb_out, 1);
    pack_tile(0, cb_out);
    cb_push_back(cb_out, 1);
    tile_regs_release();
}
