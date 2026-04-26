# BENCHMARK_TT — TT Blackhole side of the §4 v1 campaign

The companion to [BENCHMARK.md](BENCHMARK.md) §4. Specifies the
TT Blackhole-side contract that pairs with
[scripts/bench_nvidia.py](scripts/bench_nvidia.py): same JSON schema,
same shapes, same correctness gate, same headline metrics. The runnable
scaffolding lives at [tt-llk-skeleton/](tt-llk-skeleton/).

> **Caveat.** I (Claude, on the NVIDIA host) wrote the skeleton without
> a TT card to test against. The structure follows the public
> `tt-metal/programming_examples` patterns at commit `main` as of
> 2026-Q1. Some SFPU intrinsic names and `compute_kernel_api` headers
> drift between TT-Metal versions; treat the skeleton as a starting
> frame, not a contract. Edit freely.

---

## 1 Headline metrics (frozen)

The TT-side bench must emit, per Layer C size, the same metric as the
NVIDIA side:

```
effective exact modular multiplications per second
  = (M · N output elements) / (median wall-time seconds for one m×k×n matmul)
```

reported with unit `G_modmul/s`. Layer B reports raw GEMM TFLOPS (or
TOPS for INT8) at the same shapes. Layer A is informational only.

The comparison summary in `bench-results/SUMMARY.md` joins the two
sides on `(layer, useful_op_kind, shape, backend_class)` and renders a
**price-adjusted ratio** column:

```
RTX5090 throughput / 1999 USD     vs.     Blackhole throughput / 999 USD
```

so a `1.0×` ratio means the two are equally good per dollar, and `>1.0×`
means RTX 5090 wins per dollar.

---

## 2 Frozen JSON schema

Identical to NVIDIA. **Do not invent new fields.** Source of truth is
[`scripts/_bench_common.py:BenchResult`](scripts/_bench_common.py).
Required fields per record:

```json
{"schema_version": "1",
 "device": "Blackhole",
 "device_detail": {"name": "...", "arch": "...", "n_cores": 120, ...},
 "layer": "A|B|C",
 "backend": "tt_llk_int8|tt_llk_bf16|tt_llk_sfpu_fp32",
 "shape": {"M": 4096, "K": 4096, "N": 4096},
 "dtype_in": "int8|bfloat16|fp32",
 "dtype_acc": "int32|fp32",
 "iters": 50, "warmup": 5,
 "median_ms": ..., "p10_ms": ..., "p90_ms": ...,
 "useful_ops": ...,
 "useful_op_kind": "gemm_mac|exact_modmul",
 "throughput": ..., "throughput_unit": "TOPS|TFLOPS|G_modmul/s",
 "correctness": {"gate": "passed|skipped|failed", ...},
 "host_overhead_ms": null,
 "git_sha": "...", "timestamp": "..."}
```

`tt-llk-skeleton/bench_blackhole.py` builds these records for you
provided the C++ binary's CSV output is parseable.

---

## 3 Build & run

Pre-flight:

```sh
# Activate TT-Metal's environment (the skeleton's CMakeLists.txt looks
# up $TT_METAL_HOME via the env).
source $TT_METAL_HOME/setup.sh
```

Build the host driver (compiles `host/main.cpp`, copies kernels to
`build/kernels/`):

```sh
cd tt-llk-skeleton
make all
```

Run the bench (one process spawn per (layer, backend, size) cell —
the wrapper subprocesses `build/bench_blackhole` once per cell):

```sh
python bench_blackhole.py \
    --out ../bench-results/blackhole_$(git -C .. rev-parse --short HEAD)_$(date +%Y%m%d).jsonl \
    --layers A,B,C \
    --sizes 512,1024,2048,4096,8192
```

Quick smoke (sub-30-second, 256/512 only):

```sh
python bench_blackhole.py --out /tmp/tt.jsonl --quick
```

Then merge with the NVIDIA output back on the NVIDIA host (or anywhere
with both JSONL files):

```sh
python scripts/compare.py bench-results/*.jsonl --out bench-results/SUMMARY.md
```

---

## 4 Layer specifications (must match NVIDIA)

### Layer A — capability probe

- One 1024² matmul per backend (`tt_llk_int8`, `tt_llk_bf16`,
  `tt_llk_sfpu_fp32`), `iters=30`, `warmup=5`.
- Records: `layer="A"`, `useful_op_kind="gemm_mac"`, `correctness.gate="skipped"`.
- TT-specific metadata to capture in `device_detail`:
  - `arch` (Blackhole vs. Wormhole vs. ...)
  - `n_cores` (Tensix grid size)
  - `tt_metal_version` (output of `git -C $TT_METAL_HOME describe`)
  - **packing overhead**: tile-format-conversion time as a fraction of
    matmul time (separate timing region in `host/main.cpp`).
  - **SFPU dispatch latency**: a single SFPU op's time as a baseline
    (only relevant for `tt_llk_sfpu_fp32`).

The NVIDIA side records `cublasLtMatmulAlgo` `algorithm_id` and `tile`;
the TT analogue is the LLK kernel's `MATH_FIDELITY` setting (HiFi2 /
HiFi4 / LoFi) and the matrix-engine config used. Capture these in
`device_detail.llk_config`.

### Layer B — raw GEMM

- Square sizes `{512, 1024, 2048, 4096, 8192}`. Backends:
  `tt_llk_int8`, `tt_llk_bf16`, `tt_llk_sfpu_fp32`.
- 5 backends-or-fewer × 5 sizes = up to 15 records.
- Tile alignment: all sizes must be multiples of `TILE=32`. The
  skeleton's `host/main.cpp` enforces this and emits a "skipped" record
  if violated.
- Metric: `2·M·K·N / median_seconds / 1e12`, reported as TFLOPS
  (`tt_llk_bf16`, `tt_llk_sfpu_fp32`) or TOPS (`tt_llk_int8`).
- **No** modular reduction — this is pure GEMM time, comparable to the
  NVIDIA `cublaslt_*` Layer B records. Layer C-specific cost is in
  Layer C only.

### Layer C-minimal — exact 36-bit modular product (q36 INT8)

The headline measurement.

- Same NTT-friendly q36 prime as NVIDIA: ``q = 0xFFFF00001`` (= `2^36 −
  2^20 + 1`, prime, `q ≡ 1 (mod 2^16)`).
- Recipe: 5×5 = 25 INT8 32×32 MMA partials, each scaled by `2^(8(i+j))
  mod q` and reduced mod q before accumulation, then one final mod q.
  This is the same recipe as the NVIDIA path; the difference is only
  the dispatch (TT INT8 Tensix MMA + SFPU epilogue vs. cuBLASLt IGEMM
  + tensor `% q`).
- Same shape sweep: `{512, 1024, 2048, 4096, 8192}` square. Up to 5 records.
- **Bit-exact correctness gate** (`scripts/_bench_common.py::adversarial_modmul_inputs`
  is the canonical generator):
  - `0, 1, q-1, q-2`
  - `2^k ± 1` for `k ∈ {30, 32, 34, 35}`
  - 1000 random pairs in `[0, q)`
  - 100 near-quotient-boundary pairs (products near `m·q ± δ`)
- Gate must pass before any perf record is emitted. Failed gate ⇒
  `correctness.gate == "failed"` and **all perf fields null**. This is
  enforced in both the C++ binary and the Python wrapper.

The skeleton's `compute_int8_mma.cpp` has an explicit TODO for the
modular epilogue; the wrapper currently emits `correctness.gate ==
"skipped"` for every Layer C record until that TODO is filled. Don't
delete the skip — it's the honest signal that no Layer C number is
valid yet.

---

## 5 Filling in the kernel TODOs

There are **three** TODO sites in the skeleton:

### (a) `compute_int8_mma.cpp` — INT8 matmul tile loop

Lifted shape from `tt-metal/programming_examples/matmul_multi_core/kernels/compute.cpp`,
adapted for INT8 input format. The `mm_init` / `matmul_tiles` /
`pack_tile` sequence is the same; the difference is that the input CBs
are `INT8` format and the output CB is `INT32`. Verify by building the
unmodified tt-metal example, confirming it works, then changing the CB
formats.

### (b) `compute_int8_mma.cpp` — modular reduction epilogue (Layer C)

The post-matmul step that turns the int32 partial into a modular
result. SFPU primitives you'll likely need:

- `sfpu_mul_int` (or whatever your version names it) — multiply by the
  precomputed `2^(8(i+j)) mod q` constant.
- `sfpu_add_int` — accumulate across the 5×5 grid.
- `sfpu_mod` (Barrett-like) — final reduction. If your SFPU lacks an
  integer mod, implement Barrett: precompute `m_q = floor(2^k / q)`,
  reduce by `(x − ((x · m_q) >> k) · q)` and a conditional subtract.

The NVIDIA side does the same algebra in `tensor.remainder_(q)`; the
SFPU is more manual but the recipe is identical. Run the correctness
gate's adversarial set against your implementation **before** trusting
any perf number — `host/main.cpp` exposes a `--gate` flag (TODO: wire
this in once the epilogue exists).

### (c) `compute_sfpu_fp32.cpp` — FP32 GEMM via SFPU

Lower priority. Per BENCHMARK.md §3, this is the diagnostic backend —
the matrix engine isn't used; you're doing tile-by-tile FMA in the
SFPU. Only fill this in once INT8 works end-to-end and you have time
for the FP32 row of the comparison table.

---

## 6 Diagnostics

When numbers look wrong:

- **TT-Metal device profiler**: `make profile` runs with
  `TT_METAL_DEVICE_PROFILER=1`. The profiler dump goes to
  `$TT_METAL_HOME/generated/profiler/` by default; cross-check matrix-
  engine cycle counts against your wall-clock numbers.
- **`tt-smi`**: confirms which Blackhole card is exposed, its firmware
  rev, and live power/temperature.
- **Tile-format mismatches**: the most common cause of wrong numbers.
  Verify that the CB you've configured matches the kernel's expected
  format by adding a one-shot `print_tile` (`debug/dprint.h`).
- **SFPU↔Compute interleaving stalls**: if your modular epilogue uses
  many SFPU ops between MMA tiles, the matrix engine sits idle. Profile
  this — sometimes it's worth doing the SFPU work *across* tiles in a
  separate pass rather than per-tile.

NVIDIA-side diagnostics that pair with the above:

- `nsys profile` / `ncu` for kernel-level instrumentation. The bench
  script doesn't invoke either; run them manually if a number looks
  off.
- `nvmath.linalg.advanced.Matmul.algorithms` exposes the candidate algo
  list; iterate through them with `plan_preferences` if cuBLASLt's
  default heuristic underperforms.

---

## 7 What's intentionally not in v1

Per the approved plan, the following are **deferred** and should not
hold up v1 numbers:

- Layer C with q48 (48-bit prime, 6 limbs → 36 partials).
- FP32-limb decomposition (10–12 bit limbs through the SFPU FP32 path).
- FP64 2-limb segmented accumulation (TT has no FP64 matrix engine
  anyway; SFPU FP64 is not in scope).
- Layer D unfused / fused (KLSS-like inner product).
- Layer E end-to-end KLSS slice.
- Joules / power measurement.
- Intel HEXL / AVX-512-IFMA CPU baseline.

When the v1 numbers are in and we know which contrasts matter most,
v2 priorities will be planned from there.

---

## 8 v1 acceptance for the TT side

The TT side of v1 is done when:

1. `make all` succeeds on the TT host.
2. `python bench_blackhole.py --out bench-results/blackhole_<sha>.jsonl
   --quick` runs to completion and emits ≥ 1 record per
   (layer, backend, size) cell.
3. **Layer C records have `correctness.gate == "passed"`** with
   `edge_cases_failed == 0` (the adversarial set bit-exactly matches
   the Python `int` reference on host). Failed gate ⇒ null perf fields.
4. `scripts/compare.py bench-results/nvidia_*.jsonl
   bench-results/blackhole_*.jsonl --out bench-results/SUMMARY.md`
   produces a joined markdown table with both `RTX5090` and `Blackhole`
   columns populated for the matched-shape rows, plus the
   price-adjusted ratio columns.
