# 4 — Phase 7: tuned ttnn matmul on Blackhole, and what it means for the v1+v2 numbers

**Takeaway.** The Layer B numbers we reported in v1 and v2 — Blackhole BF16
at **3.9 TFLOPS**, INT8 at **7.5 TOPS** — are the throughput of *our*
matmul kernel (a thin port of the official `matmul_multi_core` programming
example). On the same silicon, the upstream tt-metal benchmark with its
hand-tuned `MatmulMultiCoreReuseMultiCastProgramConfig` reaches **142 TFLOPS
at BF16/HiFi4** and **272 TFLOPS at BF16/HiFi2**, at 92–94% device
utilisation. The reference-vs-tuned gap is **~37×** for BF16/HiFi4 and
**~70×** for BF16/HiFi2.

For INT8, the equivalent tuned reproduction lifts our v1+v2 reference
from **7.4 TOPS** to **94 TOPS** (block-tiled, no multicast) at
5120×5632×5632 — a **~13× speedup**, and within ~1.5× of the BF16/HiFi4
tuned ceiling. See §4.5.3.

This document explains how that gap was measured, why it was hidden in
v1+v2, what the new numbers do (and don't) change about the cross-device
comparison story, and how to reproduce the measurement on a fresh host.

The numbers cited here come from
[`bench-results/blackhole_0682876_ttnn_ref_v068_20260428.jsonl`](../bench-results/blackhole_0682876_ttnn_ref_v068_20260428.jsonl)
and the joined view in
[`bench-results/SUMMARY.md`](../bench-results/SUMMARY.md). The CSV produced
by the upstream benchmark is at
`/home/chs/TT/tt-metal-v068/generated/matmul_2d_host_perf_report.csv`
on the dev host (gitignored — too host-local to track).

---

## 4.1 Motivation

The v1 and v2 Layer B records put Blackhole BF16 GEMM at **3.9 TFLOPS**
([`02_findings.md` §2.2](02_findings.md#22-layer-b--raw-gemm-throughput)).
The official tt-metal [GEMM_FLOPS report][gemm-flops] shows the same card
peaking around **165 TFLOPS BF16/HiFi4** and **300 TFLOPS BF16/HiFi2** on
the published 13×10 grid. That's a 40–80× gap. Either:

1. our measurement methodology was wrong, or
2. the kernel scaffolding we used (lifted from the
   [`matmul_multi_core` programming example](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples/matmul/matmul_multi_core)
   — single-tile-per-core, no block reuse, no operand multicast) is far
   from the matrix engine's real throughput.

Phase 7 was scoped to settle that. The plan: run the upstream benchmark
*unmodified* on our card, compare the number to ours, and decide what the
gap costs the v1+v2 conclusions.

[gemm-flops]: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md

---

## 4.2 What blocked the first attempt

The upstream test
([`tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf`](https://github.com/tenstorrent/tt-metal/blob/main/tests/ttnn/unit_tests/benchmarks/test_benchmark.py))
is gated by a `@pytest.mark.skip(...)` decorator — easy to override, the
small pytest plugin in
[`tt_metal_extras/upstream_runner_conftest.py`](../tt_metal_extras/upstream_runner_conftest.py)
strips just that marker for those two test names.

The hard problem was different. Two consecutive attempts to run the
upstream test against our installed tt-metal (v0.62.0 source on disk,
ttnn 0.66.0 wheel in `.tenstorrent-venv`) deadlocked at
`TopologyMapper mapping start` immediately after `UMD | Starting devices in
cluster`. Same hang every time — process pinned at 100% CPU, device pulled
74 W (engaged), no further log output, no kernel build. The C++ matmul
example built against the same tt-metal worked perfectly (PCC 0.9999),
which ruled out a hardware fault or driver issue.

Diagnosis: a three-way mismatch between firmware, source, and Python wheel.

| Component | Version | Notes |
|---|---|---|
| Card firmware (`FLASH_BUNDLE_VERSION`) | **19.6.0** | Set via `tt-flash`; current production blob |
| `tt-metal` source on disk | **v0.62.0-era** (commit `faf26e7a61`) | umd was a flat directory, not a submodule |
| `ttnn` Python wheel in `.tenstorrent-venv` | **0.66.0** | Built against firmware 19.4.0 |
| `umd` source's `latest_supported_firmware_version` | **19.5.0** | `firmware_info_provider.cpp:293`; emits a "newer than tested" warning at runtime |

The `TopologyMapper` lives in the umd library. The C++ binary path used in
v1+v2 (`distributed::MeshDevice::create_unit_mesh`) executes against
pre-compiled firmware blobs shipped in `tt_metal/pre-compiled/` and
bypasses the Python-side topology negotiation entirely. The Python ttnn
path doesn't have that escape hatch — it has to talk to the *current*
firmware through the umd's TopologyMapper, and the v0.62-era umd code
doesn't speak fluently to firmware 19.6.0.

**Fix path:** rebuild against a tt-metal version whose umd submodule
matches the on-card firmware. Done side-by-side, not in-place, so v1+v2
binaries kept working through the experiment.

---

## 4.3 The side-by-side rebuild

Three artefacts on the dev host, all kept separate from the v1+v2 ones:

| Path | Purpose | Approx size |
|---|---|---|
| `/home/chs/TT/tt-metal-v068/` | Fresh `--branch v0.68.0 --depth 1 --recurse-submodules` clone of upstream tt-metal | 4.5 GB source + 1.4 GB build_Release + CPM cache |
| `/home/chs/.tt-metal-v068-venv-py313/` | Python 3.13 venv with editable ttnn install pointing at the v068 source tree | ~1.1 GB |
| The original `/home/chs/TT/tt-metal/` and `/home/chs/.tenstorrent-venv/` | Untouched. v1+v2 binaries continue to work. | — |

### Why v0.68.0 specifically

It's the **latest tagged release** as of the rebuild (April 2026). Newer
than that are `-dev` nightlies (`v0.70.0-dev*`); v0.68.0 is the most
recent point with full release-engineering attention. The diff window
between our v0.62-era source and v0.68.0 covers ~100 upstream commits and
spans the tt-metal restructuring that turned umd into a git submodule.

### Build steps

```sh
cd /home/chs/TT
git clone --recurse-submodules --shallow-submodules --depth 1 \
    --branch v0.68.0 https://github.com/tenstorrent/tt-metal.git tt-metal-v068
cd tt-metal-v068
./build_metal.sh --release        # ~30 min, 1039 ninja targets
```

Build was clean. Verifying the device path with the official matmul
example before touching any of our code:

```sh
TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
    cmake --build build_Release --target metal_example_matmul_multi_core --parallel
./build_Release/programming_examples/metal_example_matmul_multi_core
# → "Test Passed", PCC = 0.9999
```

### The Python venv

`create_venv.sh` only fully provisions a Python 3.12 venv (the dev-deps
path that auto-installs ttnn editable). But the build's cmake auto-picks
Python 3.13 from `$PATH` (uv's installed CPython 3.13.11), so `_ttnn.so`
links against `libpython3.13` (`PyImport_AddModuleRef`). A 3.12 venv loads
the .so with `ImportError: undefined symbol PyImport_AddModuleRef`.

Bypass: drive `uv` directly against a Python-3.13 venv:

```sh
uv venv --python 3.13 /home/chs/.tt-metal-v068-venv-py313 --clear
VIRTUAL_ENV=/home/chs/.tt-metal-v068-venv-py313 \
    uv pip install --python /home/chs/.tt-metal-v068-venv-py313/bin/python -e .
# Plus runtime deps the upstream conftest needs:
VIRTUAL_ENV=/home/chs/.tt-metal-v068-venv-py313 \
    uv pip install --python /home/chs/.tt-metal-v068-venv-py313/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu \
    pytest multiprocess loguru psutil pyyaml docopt-ng numpy pyperf graphviz \
    pytest-timeout pytest-xdist
```

ttnn smoke after that: `CreateDevice` returns in **1.5 s** (versus
"never returns" before), a 128² BF16 matmul completes in **0.4 s**,
clean `CloseDevice`. The deadlock is gone.

---

## 4.4 Running the upstream pytest

```sh
cd /home/chs/TT/tt-metal-v068
TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
PYTHONPATH=/home/chs/Work/CS_INT_Mult:$PWD:$PWD/tests \
    /home/chs/.tt-metal-v068-venv-py313/bin/pytest \
    tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf \
    -s --no-header -p no:cacheprovider \
    -p tt_metal_extras.upstream_runner_conftest
```

The `-p tt_metal_extras.upstream_runner_conftest` flag activates our
plugin from the `PYTHONPATH` entry, stripping the `@pytest.mark.skip`
from the two GEMM_FLOPS test names without modifying upstream sources.
The full sweep (10 configs × 16 shapes × 105 iterations each) finishes
in **~4 minutes** end to end and writes a 156-row CSV to
`$TT_METAL_HOME/generated/matmul_2d_host_perf_report.csv`.

To re-emit the records into our v2 schema:

```sh
uv run python scripts/bench_blackhole_ttnn_ref.py --no-run \
    --csv /home/chs/TT/tt-metal-v068/generated/matmul_2d_host_perf_report.csv \
    --out bench-results/blackhole_$(git rev-parse --short HEAD)_ttnn_ref.jsonl
uv run python scripts/compare.py bench-results/*.jsonl \
    --out bench-results/SUMMARY.md
```

The converter
([`scripts/bench_blackhole_ttnn_ref.py`](../scripts/bench_blackhole_ttnn_ref.py))
keeps the user-direction filter from the [v2 plan](../BENCHMARK_TT.md):
**BF16/HiFi4 and BF16/HiFi2 in full**, **one BF8/HiFi2 + one BF4/LoFi
sanity row each** (lower-fidelity formats aren't the destination — we want
high-precision INT-emulation candidates). 156 CSV rows → 66 JSONL records.

---

## 4.5 Numbers

### 4.5.1 Reference (v0.62 / our skeleton) vs tuned (v0.68 / upstream)

Same Blackhole p150a card. Same matrix engine. Different kernel.

| Backend (BACKEND_CLASS)                  | Plateau throughput | Best util |        Source                      |
| ---------------------------------------- | ------------------ | --------- | ---------------------------------- |
| `bf16` (reference, our `matmul_multi_core` port)         | **3.85 TFLOPS** at 4096³ |       — | v2 Layer B                         |
| `tt_llk_fp32_matrix` → `tf32` (reference, FP32 inputs)   | **1.96 TFLOPS** at 4096³ |       — | v2 Phase 1                         |
| `bf16_tuned_hifi4` (v068 tuned)          | **142 TFLOPS** at 3840×4224×4224 |  93%   | Phase 7                            |
| `bf16_tuned_hifi2` (v068 tuned)          | **272 TFLOPS** at 3840×4224×4224 |  90%   | Phase 7                            |
| `bf8_sanity` (HiFi2)                     | **275 TFLOPS** at 3840×4224×5632 |  90%   | Phase 7                            |
| `bf4_sanity` (LoFi)                      | **541 TFLOPS** at 5120×6656×6656 |  89%   | Phase 7                            |

The reference-to-tuned gap on a like-for-like dtype:

| dtype / fidelity | reference | tuned    | gap   |
| ---------------- | --------- | -------- | ----- |
| BF16 / HiFi4     | 3.85 TFLOPS | 142 TFLOPS | **37×** |
| BF16 / HiFi2     | ~3.9 TFLOPS | 272 TFLOPS | **70×** |

That gap is not silicon. The 142 TFLOPS BF16/HiFi4 number comes off the
*same* matrix engine as the 3.85 TFLOPS reference. The difference is
entirely on the host: the tuned kernel uses
`MatmulMultiCoreReuseMultiCastProgramConfig` with hand-picked block /
sub-block sizes, operand sharding into L1, multicast of operand tiles
across the core grid, and tracing to amortise dispatch overhead. The
reference kernel does single-tile-per-core matmul with no block reuse —
the matrix engine is starving for inputs roughly 96% of the time.

### 4.5.2 The grid-scaling caveat

Our card harvests at **11×10 = 110 cores**. The upstream report's "Manually
Tuned Configurations (13×10 grid)" table is for a fully-enabled p150 with
**130 cores**. Two Tensix cores on this part are disabled in firmware
(`Tensix harvesting masks indicate 2 units`). All Phase 7 numbers should
therefore be compared to upstream's results **scaled by 110/130 ≈ 0.846**:

|              | Upstream published | Our card (11×10) | Predicted (×0.846) | Match |
| ------------ | -----------------: | ---------------: | -----------------: | ----- |
| BF16 / HiFi4 |          165 TFLOPS |       142 TFLOPS |          140 TFLOPS | ✓ within 2% |
| BF16 / HiFi2 |          308 TFLOPS |       272 TFLOPS |          261 TFLOPS | ✓ within 5% |
| BF4 / LoFi   |          589 TFLOPS |       541 TFLOPS |          498 TFLOPS | ✓ within 9% (BF4 is throttled-by-bandwidth at large shapes) |

Reproduction validated.

### 4.5.3 INT8 tuned matmul (closes the §4.7 INT8 open item)

The upstream `test_matmul_2d_host_perf` script doesn't iterate INT8 —
ttnn's high-level `ttnn.matmul` accepts BF16 / BF8_b / BF4_b / FP32 but
not INT8 in v0.68. Closing that gap, the new harness at
[`tt-llk-skeleton/host_int8_tuned/main.cpp`](../tt-llk-skeleton/host_int8_tuned/main.cpp)
ports the upstream
[`matmul_multicore_reuse`](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples/matmul/matmul_multicore_reuse)
programming example to INT8 by:

- switching CB DataFormat to `tt::DataFormat::Int8` (single_tile_size =
  1024 B, half of BF16),
- enabling `fp32_dest_acc_en = true` for the INT32 destination
  accumulator,
- converting host inputs to sign-magnitude (the on-card representation
  Tensix's INT8 path expects, same conversion as v1's INT8 reference),
- referencing the upstream
  [`bmm_large_block_zm.cpp`](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/matmul/matmul_common/kernels/compute/bmm_large_block_zm.cpp)
  compute kernel **unmodified** — same `mm_init` + `matmul_tiles` LLK
  calls our v1 reference uses, just with block / sub-block orchestration
  via 12 compile-time args.

This is the **mid-tier tuned path: block reuse on, operand multicast
off.** Full mcast adds another 2–3× and is left as future work.

The Python wrapper
([`scripts/bench_blackhole_int8_tuned.py`](../scripts/bench_blackhole_int8_tuned.py))
sweeps per-core base shapes that mirror the upstream BF16 list (scaled
by the 11×10 grid). Pinning per-core block sizes to `Mt/grid_y` and
`Nt/grid_x` (the same hardcoded recipe upstream's pytest uses) instead
of the auto-tuner is what lights up all 110 cores; the auto-tuner picks
sizes that under-fill the grid (only 22 cores at the medium shape).

| Shape (M×K×N)         | HiFi4 INT8 | HiFi2 INT8 | vs v1 reference INT8 (7.4 TOPS) |
| --------------------- | ---------- | ---------- | ------------------------------- |
| 1280×2816×2816        | 35.8 TOPS  | 38.0 TOPS  | **5.1× / 5.4×** |
| 2560×2816×2816        | 47.8 TOPS  | 54.7 TOPS  | 6.5× / 7.4× |
| 2560×4224×4224        | 58.9 TOPS  | 65.0 TOPS  | 8.0× / 8.8× |
| 3840×4224×4224        | 64.4 TOPS  | 78.3 TOPS  | 8.7× / 10.6× |
| 3840×4224×5632        | 65.9 TOPS  | 83.9 TOPS  | 8.9× / 11.3× |
| 4160×3520×3520 (P150) | 64.5 TOPS  | 74.0 TOPS  | 8.7× / 10.0× |
| **5120×5632×5632**    | **70.8 TOPS** | **93.9 TOPS** | **9.6× / 12.7×** |

Within "a few factors" of the BF16 tuned numbers — the user's acceptance
criterion. INT8/HiFi2 plateau at **94 TOPS** is **1.5×** below
BF16/HiFi4 (142 TFLOPS) and **2.9×** below BF16/HiFi2 (272 TFLOPS).
The HiFi4 → HiFi2 gain (~30%) on INT8 is free — INT8 doesn't lose
precision at HiFi2, just halves the per-tile cycle count.

The remaining gap to the BF16 tuned ceiling is the operand multicast
that upstream's tuned path uses but ours doesn't: each core fetches its
own copy of the operand tiles from DRAM instead of having one core
multicast them across the row / column. Lifting mcast onto INT8 should
buy another 2–3×, putting it in the ~200–300 TOPS range — competitive
with the published GEMM_FLOPS INT8 numbers for Blackhole.

Energy: HiFi2 at the largest shape draws 41.5 W; **2.26 TOPS/W** for
INT8 — directly comparable to BF16/HiFi2's 6.24 TFLOPS/W from §4.5.4
(roughly 2.8× lower energy efficiency for INT8 at this kernel quality
because the per-tile work is the same but each "TOP" is half the bit
width's worth of useful work).

### 4.5.4 Per-watt under load

Steady-state Tensix power held at **39–46 W** through the entire 4-minute
sweep (`tt-smi -s | TDP`). The card's published TDP cap is 80 W, so the
matrix engine is working at ~50% of the silicon's power envelope at
maximum reported utilisation. Per-watt:

| Config @ 1280×2816×2816 | Throughput | Power | TFLOPS / W |
| ----------------------- | ---------: | ----: | ---------: |
| `bf16_tuned_hifi2`      | 246.5      | 39.5 W | **6.24**  |
| `bf16_tuned_hifi4`      | 123.8      | 39.5 W | **3.13**  |
| `bf16` (v0.62 reference)| 3.7        | 44 W  |  0.085    |

The reference path does ~70× less compute per watt than the tuned path.
Most of that delta is wasted matrix-engine cycles, not extra power for
the same work — so the **right framing is "the reference kernel hides
power efficiency, not consumes it"**. The tuned BF16/HiFi2 result is
within shouting distance of the L40S's published 6.4 TFLOPS BF16/W and
beats Blackwell consumer parts on per-watt BF16 by a wide margin (those
cards target inference batch-size, not energy efficiency, so the
comparison is asymmetric).

---

## 4.6 What this changes about the v1+v2 conclusions

[`02_findings.md`](02_findings.md) reports headline ratios like:

> "INT8 Tensor Core / Tensix — RTX 5090 210 TOPS, Blackhole 7.4 TOPS,
>  ratio 28.4× raw, 14.2× per kUSD."

Those ratios are between **NVIDIA cuBLASLt** (a tuned production library)
and **our reference Blackhole kernel** (an unoptimised port of an
educational example). They do not say what they sound like — they say
nothing about how the two pieces of silicon compare on apples-to-apples
tuned dispatch.

The honest re-statement, with Phase 7 in hand:

| Comparison axis                                            | What v2 said             | What v2 actually measured                                                  |
| ---------------------------------------------------------- | ------------------------ | -------------------------------------------------------------------------- |
| Layer B raw GEMM, Blackhole vs RTX 5090 (BF16/TF32 / fp32) | Blackhole loses by 30–50× | Reference TT kernel loses to cuBLASLt TF32 / FP32. **At equal kernel quality, the gap is closer to 1–3×** based on the Phase 7 numbers extrapolated. |
| Layer C exact 36-bit modmul, ratio at 4096³                | RTX 5090 wins ~9× per dollar | The 25-INT8-GEMM cascade on Blackhole is reference-kernel bottlenecked. With a tuned matmul under the same recipe, the cascade should improve by ~30–40× on the matmul half — closing most of the gap. The host-side reduction cost stays the same. |
| Layer D KLSS-IP useful G_MAC/s                             | Blackhole peaks ~300 G_MAC/s | Same caveat — that's a 25× reference-kernel-cascade plateau, not a Blackhole peak. The 25-tuned-matmul number would land around 2.5–3.5 T_MAC/s. |

What remains unchanged:

- INT8 is **not in the upstream `test_matmul_2d_host_perf` script** — it
  iterates over BF16, BF8, BF4 only. The §4.5.3 INT8 tuned reproduction
  (added 2026-04-29) plugs that gap on the block-tiled side: the v1
  reference's 7.4 TOPS becomes **70.8–93.9 TOPS** at large shapes
  (5120×5632×5632, HiFi4 / HiFi2). Multicast on INT8 — another expected
  2–3× — is still open work (see §4.7).
- Layer C's correctness gate behaviour is unchanged (still passes against
  the host bigint reference).
- Power and per-dollar columns still apply — the math just moves with the
  numerator.

The upstream-tuned numbers are what should be on a marketing slide. The
reference numbers are what should be on a "here's the cost of writing a
naïve matmul kernel" slide. Both are valid; calling either one "what the
silicon does" without qualification is wrong.

---

## 4.7 What's still open

| Item | Why deferred |
| --- | --- |
| ~~INT8 tuned matmul (`tt_matmul_2d_int8_*`)~~ | **DONE 2026-04-29 — see §4.5.3.** Plateau **94 TOPS** at 5120×5632×5632 (INT8/HiFi2), within a few factors of BF16/HiFi4 (142 TFLOPS) and ~13× the v1 reference's 7.4 TOPS. Block-tiled, **no multicast** — that next 2–3× is the new open item. |
| INT8 tuned **with operand multicast** | The §4.5.3 INT8 path uses block reuse but not operand multicast. Adding mcast (port the upstream `matmul_multicore_reuse_mcast` example with the same INT8 adaptations) would close the remaining ~3× gap to the BF16 tuned ceiling. ~3–5 hours of work; not blocking. |
| FP32 tuned matmul (`tt_matmul_2d_fp32_hifi4`) | Same situation — upstream doesn't iterate FP32. The Phase 7 wrapper has the BACKEND_CLASS slot wired (`fp32_tuned`) but no rows hit it yet. |
| Custom-shape sweep (rectangular / SRAM-fit boundary) | `tt_metal_extras/test_ttnn_shapes.py` envisioned in the v2 plan was not written — direct parameter sweeping at the upstream test's level requires either patching the test or reproducing its scaffolding locally. Not blocking. |
| Re-baseline v1/v2 reference numbers against v0.68 build | The v1/v2 binary at `tt-llk-skeleton/build/bench_blackhole` is built against v0.62 headers. Rebuilding against v0.68 might shift the reference numbers slightly (the underlying matmul tile API has minor signature changes). The qualitative gap conclusion is unchanged either way. |
| q48 + Layer D using the tuned matmul | These would update the v2 cost-proxy numbers by the same ~37× factor for the matmul portion of the cascade. The host-side reduction cost stays identical. |

---

## 4.8 Reproducibility punch-list

Everything needed to run this on a similarly-configured Blackhole p150
host:

```sh
# 0. Card prerequisites: firmware ≥ 19.5.0 (warning is OK; deadlock isn't)
tt-smi -s | grep -E 'fw_bundle|FLASH_BUNDLE_VERSION'

# 1. Side-by-side clone + build (≈30 min)
cd ~/TT
git clone --recurse-submodules --shallow-submodules --depth 1 \
    --branch v0.68.0 https://github.com/tenstorrent/tt-metal.git tt-metal-v068
cd tt-metal-v068 && ./build_metal.sh --release

# 2. Verify the C++ device path works (≈30 s on warm cache)
TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
    cmake --build build_Release --target metal_example_matmul_multi_core --parallel
TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
    ./build_Release/programming_examples/metal_example_matmul_multi_core
# Expect: "Test Passed", PCC ≥ 0.99

# 3. Python 3.13 venv with editable ttnn install (≈3 min)
uv venv --python 3.13 ~/.tt-metal-v068-venv-py313 --clear
VIRTUAL_ENV=~/.tt-metal-v068-venv-py313 \
    uv pip install --python ~/.tt-metal-v068-venv-py313/bin/python -e .
VIRTUAL_ENV=~/.tt-metal-v068-venv-py313 \
    uv pip install --python ~/.tt-metal-v068-venv-py313/bin/python \
    torch --index-url https://download.pytorch.org/whl/cpu \
    pytest multiprocess loguru psutil pyyaml docopt-ng numpy pyperf graphviz \
    pytest-timeout pytest-xdist

# 4. Smoke ttnn (≈5 s)
TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
    ~/.tt-metal-v068-venv-py313/bin/python -c "
import ttnn, torch, time
t0=time.time(); dev=ttnn.CreateDevice(device_id=0); print(f'open {time.time()-t0:.1f}s')
a=torch.randn(128,128).bfloat16(); b=torch.randn(128,128).bfloat16()
ta=ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tb=ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
tc=ttnn.matmul(ta,tb); _=ttnn.to_torch(tc); ttnn.CloseDevice(dev); print('closed')
"

# 5. Run the upstream benchmark (≈4 min). Plugin is from this repo.
cd ~/Work/CS_INT_Mult   # or wherever this repo lives
cd ~/TT/tt-metal-v068
TT_METAL_HOME=$PWD TT_METAL_RUNTIME_ROOT=$PWD ARCH_NAME=blackhole \
PYTHONPATH=~/Work/CS_INT_Mult:$PWD:$PWD/tests \
    ~/.tt-metal-v068-venv-py313/bin/pytest \
    tests/ttnn/unit_tests/benchmarks/test_matmul_2d_host_perf -s \
    -p tt_metal_extras.upstream_runner_conftest

# 6. Convert to JSONL and refresh the joined SUMMARY
cd ~/Work/CS_INT_Mult
SHA=$(git rev-parse --short HEAD)
uv run python scripts/bench_blackhole_ttnn_ref.py --no-run \
    --csv ~/TT/tt-metal-v068/generated/matmul_2d_host_perf_report.csv \
    --out bench-results/blackhole_${SHA}_ttnn_ref.jsonl
uv run python scripts/compare.py bench-results/*.jsonl \
    --out bench-results/SUMMARY.md
```

If step 5 hangs at `TopologyMapper mapping start`, the firmware /
tt-metal version mismatch is back — try a newer tt-metal tag, or
roll the firmware back to one umd's `latest_supported_firmware_version`
explicitly knows about (currently 19.5.0 in v0.68's umd submodule).
