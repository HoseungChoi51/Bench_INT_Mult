// NoC reader: streams A and B operand tiles from DRAM into L1 circular buffers.
// Lifted from
// tt_metal/programming_examples/matmul/matmul_multi_core/kernels/dataflow/reader_mm_output_tiles_partitioned.cpp.
//
// Compile-time args (appended via TensorAccessorArgs on the host, two
// accessors back-to-back: A then B).
//
// Runtime args:
//   0: a_dram_addr
//   1: b_dram_addr
//   2: Mt  (output rows in tiles)
//   3: Kt  (inner dim in tiles)
//   4: Nt  (output cols in tiles)
//   5: output_tile_start_id  (this core's first output tile, linear)
//   6: num_output_tiles      (how many output tiles this core handles)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src0_addr = get_arg_val<uint32_t>(0);
    const uint32_t src1_addr = get_arg_val<uint32_t>(1);
    // Mt (arg 2) is unused by this kernel; the reader works in linear-tile space.
    const uint32_t Kt = get_arg_val<uint32_t>(3);
    const uint32_t Nt = get_arg_val<uint32_t>(4);
    const uint32_t output_tile_start_id = get_arg_val<uint32_t>(5);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(6);

    constexpr uint32_t cb_in0 = tt::CBIndex::c_0;
    constexpr uint32_t cb_in1 = tt::CBIndex::c_1;

    constexpr auto a_args = TensorAccessorArgs<0>();
    const auto a = TensorAccessor(a_args, src0_addr);
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto b = TensorAccessor(b_args, src1_addr);

    for (uint32_t out = 0; out < num_output_tiles; ++out) {
        const uint32_t out_id = output_tile_start_id + out;
        const uint32_t out_row = out_id / Nt;
        const uint32_t out_col = out_id % Nt;

        for (uint32_t k = 0; k < Kt; ++k) {
            const uint32_t a_tile = out_row * Kt + k;
            cb_reserve_back(cb_in0, 1);
            noc_async_read_tile(a_tile, a, get_write_ptr(cb_in0));
            noc_async_read_barrier();
            cb_push_back(cb_in0, 1);

            const uint32_t b_tile = k * Nt + out_col;
            cb_reserve_back(cb_in1, 1);
            noc_async_read_tile(b_tile, b, get_write_ptr(cb_in1));
            noc_async_read_barrier();
            cb_push_back(cb_in1, 1);
        }
    }
}
