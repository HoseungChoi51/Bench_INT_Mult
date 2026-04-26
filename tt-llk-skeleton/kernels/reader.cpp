// NoC reader kernel: streams A/B operand tiles from DRAM into L1 circular
// buffers for the compute kernel. Standard pattern lifted from
// tt-metal/programming_examples/matmul_multi_core/kernels/reader.cpp.
//
// Runtime args (set from host main.cpp):
//   0: a_dram_addr
//   1: b_dram_addr
//   2: Mt   (output rows in tiles)
//   3: Kt   (inner dim in tiles)
//   4: Nt   (output cols in tiles)

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t Mt = get_arg_val<uint32_t>(2);
    const uint32_t Kt = get_arg_val<uint32_t>(3);
    const uint32_t Nt = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_a = tt::CBIndex::c_0;
    constexpr uint32_t cb_b = tt::CBIndex::c_1;

    const uint32_t a_tile_bytes = get_tile_size(cb_a);
    const uint32_t b_tile_bytes = get_tile_size(cb_b);

    InterleavedAddrGenFast<true> a_gen{
        .bank_base_address = a_addr,
        .page_size = a_tile_bytes,
        .data_format = get_dataformat(cb_a),
    };
    InterleavedAddrGenFast<true> b_gen{
        .bank_base_address = b_addr,
        .page_size = b_tile_bytes,
        .data_format = get_dataformat(cb_b),
    };

    // Block-tiled read pattern: for each output (mt, nt) tile, stream Kt A-row
    // tiles and Kt B-col tiles into the CBs in lockstep.
    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            for (uint32_t kt = 0; kt < Kt; ++kt) {
                cb_reserve_back(cb_a, 1);
                cb_reserve_back(cb_b, 1);

                noc_async_read_tile(mt * Kt + kt, a_gen, get_write_ptr(cb_a));
                noc_async_read_tile(kt * Nt + nt, b_gen, get_write_ptr(cb_b));
                noc_async_read_barrier();

                cb_push_back(cb_a, 1);
                cb_push_back(cb_b, 1);
            }
        }
    }
}
