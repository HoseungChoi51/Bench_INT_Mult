// NoC writer: streams output tiles from L1 to DRAM.
// Lifted from
// tt_metal/programming_examples/matmul/matmul_multi_core/kernels/dataflow/writer_unary_interleaved_start_id.cpp.
//
// Compile-time args (one TensorAccessorArgs for the C buffer).
//
// Runtime args:
//   0: c_dram_addr
//   1: num_tiles    (output tiles produced by this core)
//   2: start_id     (linear tile id of this core's first output tile)

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_tiles = get_arg_val<uint32_t>(1);
    const uint32_t start_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr);

    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_wait_front(cb_out, 1);
        noc_async_write_tile(i, c, get_read_ptr(cb_out));
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
