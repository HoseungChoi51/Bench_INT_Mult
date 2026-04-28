// NoC reader for the two-input inner-product harness:
// streams `num_tiles` tiles of A and B from DRAM into c_0 and c_1.
//
// Each core handles a contiguous range of `num_tiles` tiles starting at
// `start_id` in the linearized tile space.
//
// Compile-time args (TensorAccessorArgs back-to-back: A, B):
//   tensor accessor metadata for each of the 2 DRAM operands.
//
// Runtime args:
//   0: a_dram_addr
//   1: b_dram_addr
//   2: num_tiles  (this core's share)
//   3: start_id   (linear tile id of this core's first tile)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t num_tiles = get_arg_val<uint32_t>(2);
    const uint32_t start_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    constexpr auto a_args = TensorAccessorArgs<0>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto a = TensorAccessor(a_args, a_addr);
    const auto b = TensorAccessor(b_args, b_addr);

    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_a, 1);
        cb_reserve_back(cb_b, 1);
        noc_async_read_tile(i, a, get_write_ptr(cb_a));
        noc_async_read_tile(i, b, get_write_ptr(cb_b));
        noc_async_read_barrier();
        cb_push_back(cb_a, 1);
        cb_push_back(cb_b, 1);
    }
}
