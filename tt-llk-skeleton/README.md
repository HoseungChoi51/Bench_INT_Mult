# tt-llk-skeleton — TT Blackhole bench skeleton

Runnable scaffolding for the Tenstorrent side of the
[BENCHMARK.md](../BENCHMARK.md) §4 v1 campaign. Pairs with
[bench_nvidia.py](../scripts/bench_nvidia.py) on the RTX 5090 side.

This skeleton is **structurally complete** — it builds, dispatches a
program to a Blackhole device, runs the timer/profiler glue, and emits a
JSONL record that conforms to the same schema as `bench_nvidia.py`. The
**arithmetic** inside the Compute kernels is left as TODO sites because
the SFPU intrinsic API drifts between TT-Metal versions; you fill those
in once on your TT host.

> **Caveat from the author of this skeleton (Claude, on the NVIDIA host):**
> I have not built this against TT-Metal hardware. The structure follows
> the public `tt-metal/programming_examples/` (matmul_multi_core,
> eltwise_binary) at commit `main` as of 2026-Q1. SFPU intrinsic names,
> CB indices, and tile API may need adjustment to your installed
> TT-Metal version. Treat this as a starting frame, not a contract.

See [BENCHMARK_TT.md](../BENCHMARK_TT.md) at the repo root for the layer
specifications, JSON schema, and validation gate.

## Layout

```
tt-llk-skeleton/
├── README.md                    # this file
├── Makefile                     # convenience: build / run / clean / dryrun
├── bench_blackhole.py           # Python wrapper: spawns binary, parses CSV, emits JSONL
├── host/
│   ├── main.cpp                 # TT-Metal host driver (Program, Buffers, profiler)
│   └── CMakeLists.txt           # picks up TT-Metal cmake exports from $TT_METAL_HOME
└── kernels/
    ├── reader.cpp               # NoC reader: DRAM → L1
    ├── writer.cpp               # NoC writer: L1 → DRAM
    ├── compute_int8_mma.cpp     # Tensix INT8 32×32 MMA + TODO modular epilogue
    ├── compute_bf16_mma.cpp     # Tensix BF16 path
    └── compute_sfpu_fp32.cpp    # SFPU FP32 vector path
```

## Build & run

Prerequisites: a working TT-Metal install. Source its environment first:

```sh
source $TT_METAL_HOME/setup.sh    # or your install's equivalent
cd tt-llk-skeleton
make all                          # builds host/main into ./build/bench_blackhole
python bench_blackhole.py \
    --out ../bench-results/blackhole_$(git -C .. rev-parse --short HEAD)_$(date +%Y%m%d).jsonl \
    --layers A,B,C \
    --sizes 512,1024,2048,4096,8192
```

To dry-run the makefile graph without TT-Metal installed (useful from
the NVIDIA host as a CI smoke):

```sh
make -n
```

## Filling in the TODOs

There are three TODO sites, listed in priority order:

1. **`kernels/compute_int8_mma.cpp` — Tensix INT8 32×32 tile MMA.**
   The skeleton has the tile loop, CB acquire/release, and pack/unpack
   stubs. You provide the actual `mm_init` / `matmul_tiles` / `pack_tile`
   sequence using your TT-Metal version's LLK headers.
2. **`kernels/compute_int8_mma.cpp` — modular epilogue (Layer C).**
   For Layer C the int32 partial must be scaled by ``2^(8(i+j)) mod q``
   and reduced ``mod q`` per byte-shift index. Skeleton points at SFPU
   builtins; pick Barrett or naive division based on what's exposed.
3. **`kernels/compute_sfpu_fp32.cpp` — FP32 vector path.**
   Lower priority (FP32 is the diagnostic backend per BENCHMARK.md §3,
   not the headline). Fill in once you have INT8 working.

## What's intentionally not in v1

Per the approved plan
([../plans](file:///home/chs/.claude/plans/i-have-added-benchmark-md-happy-galaxy.md)),
v1 covers only Layers A, B, and C-minimal (q36 INT8). The following are
*deferred*; do not let their absence block v1 numbers:

- Layer D unfused / fused (KLSS-like inner product).
- Layer E end-to-end KLSS slice.
- BF16/TF32 paths beyond the GEMM scaffold (no modular reduction).
- Joules / power measurement.

When the v1 numbers are in, we'll plan v2 from there.
