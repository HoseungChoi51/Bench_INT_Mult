// NoC writer kernel: streams output tiles from L1 to DRAM.
//
// Runtime args:
//   0: c_dram_addr
//   1: Mt   (output rows in tiles)
//   2: Nt   (output cols in tiles)

#include <cstdint>
#include "dataflow_api.h"

void kernel_main() {
    const uint32_t c_addr = get_arg_val<uint32_t>(0);
    const uint32_t Mt = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    const uint32_t tile_bytes = get_tile_size(cb_out);

    InterleavedAddrGenFast<true> c_gen{
        .bank_base_address = c_addr,
        .page_size = tile_bytes,
        .data_format = get_dataformat(cb_out),
    };

    for (uint32_t mt = 0; mt < Mt; ++mt) {
        for (uint32_t nt = 0; nt < Nt; ++nt) {
            cb_wait_front(cb_out, 1);
            noc_async_write_tile(mt * Nt + nt, c_gen, get_read_ptr(cb_out));
            noc_async_write_barrier();
            cb_pop_front(cb_out, 1);
        }
    }
}
