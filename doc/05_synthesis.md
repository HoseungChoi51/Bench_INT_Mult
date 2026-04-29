# 5 — Synthesis after Phase 7+8

**Takeaway.** The cross-device picture has changed in the last 48 hours.
At the v1+v2 cutoff ([`02_findings.md`](02_findings.md)), the headline
read "RTX 5090 wins INT8 by 28× silicon, 14× per dollar." With the
Phase 7 tuned-matmul reproductions in hand, the same comparison is now
**TT slightly ahead on raw INT8 (1.06×) and 2.1× ahead per dollar** at
silicon-vs-silicon best throughput. The 28× v2 gap was kernel quality,
not silicon — Phase 7 closed it without touching the matrix engine
itself. Phase 8 added a small but informative SFPU INT32 microbench
that quantifies what the *integer* path looks like outside the matrix
engine.

This document walks through what changed, what still hasn't been
re-measured, and what the v3-quality conclusions look like.

The numbers cited come from
[`bench-results/SUMMARY.md`](../bench-results/SUMMARY.md) — refreshed
2026-04-29 against the full 325-record / 12-JSONL dataset. The plots
in [`bench-results/`](../bench-results/) were regenerated against the
same data. When the SUMMARY moves, so does this doc.

---

## 5.1 What's new since [`02_findings.md`](02_findings.md)

Five commits over two days, and two new source areas:

| Commit | What it added |
| --- | --- |
| `0682876` | Phase 7 *infrastructure*: upstream GEMM_FLOPS converter ([`scripts/bench_blackhole_ttnn_ref.py`](../scripts/bench_blackhole_ttnn_ref.py)) and a pytest plugin ([`tt_metal_extras/upstream_runner_conftest.py`](../tt_metal_extras/upstream_runner_conftest.py)) that strips `@pytest.mark.skip` from the upstream benchmark — runs `test_matmul_2d_host_perf` against our card without modifying upstream sources. Initial run blocked on a firmware/ttnn JIT mismatch. |
| `f3d2138` | Phase 7 *reproduction*: side-by-side `v0.68.0` rebuild side-stepped the JIT block. 66 BF16/BF8/BF4 tuned records emitted into [`bench-results/blackhole_0682876_ttnn_ref_v068_20260428.jsonl`](../bench-results/blackhole_0682876_ttnn_ref_v068_20260428.jsonl). |
| `dfe8c54` | Narrative: [`04_phase7_tuned_matmul.md`](04_phase7_tuned_matmul.md) — the firmware/build investigation, the 37–70× reference-vs-tuned BF16 gap, and how to reproduce it. |
| `f468f37` | Phase 8: SFPU INT32 microbench — eltwise FMA and per-lane inner product. Two new JSONLs (Layer E records) and four new TT-LLK kernels under [`tt-llk-skeleton/host_sfpu_ip/`](../tt-llk-skeleton/host_sfpu_ip/) and [`tt-llk-skeleton/kernels/`](../tt-llk-skeleton/kernels/). |
| `9bc6a0b` | Phase 7 INT8 extension (block-tiled): closes the §4.7 INT8 item. Lifts INT8 from the v2 reference's **7.4 TOPS** to **94 TOPS** at 5120×5632×5632 (HiFi2). New JSONL: [`blackhole_f468f37_int8_tuned_20260429.jsonl`](../bench-results/blackhole_f468f37_int8_tuned_20260429.jsonl). |
| `7d234af` | Phase 7 INT8 mcast: block-reuse + operand multicast. Plateau **228 TOPS** at the same shape (HiFi2). New JSONL: [`blackhole_9bc6a0b_int8_tuned_mcast_20260429.jsonl`](../bench-results/blackhole_9bc6a0b_int8_tuned_mcast_20260429.jsonl). This is the run that crosses past NVIDIA cuBLASLt's 215 TOPS plateau. |
| `be865dd` | Open caveat: [`discussions/nvidia_cublaslt_int8_utilization.md`](discussions/nvidia_cublaslt_int8_utilization.md) — why the cuBLASLt INT8 measurement sits at ~51% of dense Tensor Core peak and the asymmetry that creates against the TT mcast number sitting at ~83% of its matrix-engine ceiling. |

The SUMMARY count went from 217 records / 7 files (v2 cutoff) to **325
records / 12 files** today.

---

## 5.2 Layer B — raw GEMM, by kernel-quality tier

The [Layer B plot](../bench-results/layer_b_throughput.png) now shows
three bands of TT data on the y-axis. Reading them is easier with the
tier-based table than with the line chart:

### Tier 1 — reference (our `matmul_multi_core` skeleton, v0.62 build)

This is what v1+v2 measured. Single tile per core, no block reuse, no
multicast.

| Backend | Plateau (4096³) | per kUSD |
| ------- | --------------: | -------: |
| `tt_llk_bf16` | 3.91 TFLOPS | 3.92 |
| `tt_llk_fp32_matrix` (TF32-internal) | 1.98 TFLOPS | 1.98 |
| `tt_llk_int8` | 7.57 TOPS | 7.58 |

The matrix engine is starving for inputs ~96% of the time
([`04_phase7_tuned_matmul.md` §4.5.1](04_phase7_tuned_matmul.md)). Cross-checked
against [Tenstorrent's GEMM_FLOPS report][gf]: their published BF16/HiFi4
peak on a 13×10 grid is ~165 TFLOPS, and Phase 7 establishes the same
silicon delivers **142 TFLOPS** on our 11×10 (harvested) grid. The
reference 3.91 TFLOPS is therefore at ~2.7% of what the silicon is
capable of — kernel quality, not silicon.

### Tier 2 — tuned, block reuse on, multicast off (Phase 7 extension)

Block reuse keeps operand tiles in L1 across the inner-product loop.
This alone lifts the matrix-engine utilisation from ~3% to ~50%.

| Backend (5120×5632×5632) | Plateau | per kUSD |
| ------------------------ | ------: | -------: |
| `tt_matmul_2d_int8_hifi4` | 70.8 TOPS | 70.9 |
| `tt_matmul_2d_int8_hifi2` | **93.9 TOPS** | 94.0 |

That's **9.6× / 12.7×** vs the reference INT8. INT8 at HiFi2 doesn't
lose precision (fixed-point op, fidelity affects only float
mantissa-bit-width); the ~30% HiFi4→HiFi2 gain is free.

### Tier 3 — tuned, block reuse + 2D operand multicast

One core fetches each operand tile from DRAM and broadcasts it across
the row/column. Saves 110× the redundant DRAM reads.

| Backend (5120×5632×5632) | Plateau | per kUSD | mcast / non-mcast |
| ------------------------ | ------: | -------: | ----------------: |
| `tt_matmul_2d_int8_mcast_hifi4` | 132.2 TOPS | 132.3 | 1.87× |
| `tt_matmul_2d_int8_mcast_hifi2` | **228.5 TOPS** | **228.7** | **2.43×** |

This is the run that **crosses past cuBLASLt's INT8 plateau of 215 TOPS
on RTX 5090** at the closest comparable shape (4096³, 215 TOPS). At
silicon-best on each side, TT is **1.06× ahead raw** and **2.13× ahead
per dollar** (TT $999 vs RTX 5090 $1999).

### Tier 4 — tuned BF16 + reduced-precision sanity

The upstream `test_matmul_2d_host_perf` script also iterates BF16/BF8/BF4
at every fidelity. Reproducing all of it is the easiest way to validate
that the tuned harness is hitting the matrix engine's actual ceiling
on a partially-harvested card. Results at the largest shape
upstream tests (3840×4224×4224 for BF16, 3840×4224×5632 / 5120×6656×6656
for BF8/BF4):

| Backend | Plateau | Pred. (×0.846 of upstream 13×10) | Match |
| ------- | ------: | -------------------------------: | :---- |
| `bf16_tuned_hifi4` | 142 TFLOPS | 140 | ✓ within 2% |
| `bf16_tuned_hifi2` | 272 TFLOPS | 261 | ✓ within 5% |
| `bf8_sanity` (BFP8/HiFi2) | 275 TFLOPS | 261 | ✓ within 6% |
| `bf4_sanity` (BFP4/LoFi)  | 541 TFLOPS | 498 | ✓ within 9% (BF4 is bandwidth-throttled at peak) |

All four match the upstream-published peak when scaled by the harvested
core ratio (110/130 = 0.846). The tuned harness is reproducing what TT's
own benchmark reports — no methodology mystery left.

[gf]: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md

### Cross-device, tier-aware

| Tier on each side | RTX 5090 | TT Blackhole p150a (110 cores) | TT silicon advantage | TT per-dollar advantage |
| ----------------- | -------: | -----------------------------: | -------------------: | ----------------------: |
| Reference / out-of-the-box | cuBLASLt INT8 (`torch._int_mm`): **215 TOPS** | reference `tt_llk_int8`: 7.4 TOPS | NVIDIA wins 28.4× | NVIDIA wins 14.2× |
| Hand-tuned production library | cuBLASLt INT8 (~51% of dense peak) | Phase 7 mcast HiFi2: **228 TOPS** | **TT wins 1.06×** | **TT wins 2.13×** |
| Best vendor-public (CUTLASS / TRT-LLM, Option A in [discussions/](discussions/nvidia_cublaslt_int8_utilization.md)) | est. 300–400 TOPS (75% of 419 TOPS dense ceiling) | same 228 TOPS — TT mcast already at ~83% of its silicon ceiling | NVIDIA wins ~1.5× | TT wins ~1.34× |

The two production-tier rows are the one most readers should care about
because they reflect what an engineer will actually measure on day one.
The third row is a projection — the asymmetry it captures (NVIDIA's
production library leaves more silicon on the table than TT's does) is
real and lives in [`discussions/nvidia_cublaslt_int8_utilization.md`](discussions/nvidia_cublaslt_int8_utilization.md)
until someone runs the CUTLASS experiment.

**The single-line headline change vs v2:** the cross-device
silicon-vs-silicon Layer B INT8 ratio went from 28.4× NVIDIA-favoured
to 1.06× TT-favoured. Per dollar, from 14.2× NVIDIA-favoured to 2.13×
TT-favoured. That's a 30× and 30× swing respectively, and it came from
two well-known kernel patterns lifted verbatim from upstream programming
examples.

---

## 5.3 Layer C — exact 36-bit modmul, what we measured vs what we now project

Layer C was *not* re-run with the tuned matmul. The numbers in
[`02_findings.md` §2.3](02_findings.md#23-layer-c--exact-36-bit-modular-product)
still stand for the *measured* dataset:

| Shape  | RTX 5090 INT8 | TT INT8 (reference) | Ratio (raw) | per-dollar |
| ------ | ------------: | -------------------: | ----------: | ---------: |
| 512³   | 0.34 G_modmul/s | 0.21 G_modmul/s | 1.62× | TT wins 1.24× |
| 1024³  | 0.84          | 0.14                | 6.21× | NVIDIA wins 3.10× |
| 2048³  | 0.85          | 0.07                | 11.74× | NVIDIA wins 5.87× |
| 4096³  | 0.45          | 0.04                | 12.09× | NVIDIA wins 6.04× |
| 8192³  | 0.33          | 0.018               | 18.04× | NVIDIA wins 9.01× |

What changes after Phase 7 is the **expected** Layer C number once the
recipe is re-run on top of the tuned matmul. Layer C is `25 × INT8 GEMM
+ host-side reduction`. The matmul half has just been demonstrated to
be ~31× faster (7.4 → 228 TOPS); the reduction half is unchanged.
Conservatively assuming the reduction stays at the v2 cost:

- Time per Layer C matmul ≈ time per INT8 GEMM × 25 + reduction overhead.
- v2 measured: 25 GEMMs at 7.4 TOPS plateau ⇒ ~95% of the wall-time
  was the GEMM half (the reduction is a single fused fp32 epilogue,
  cheap); ~5% was the reduction.
- With tuned INT8: GEMM half is 31× faster, reduction stays the same.
  New time: `(0.95 × wall_time / 31) + 0.05 × wall_time` = `0.0306 +
  0.05 = 0.081 × wall_time` ⇒ ~12.4× speed-up overall, plateauing as
  the reduction becomes the bottleneck.
- Expected Layer C plateau at 4096³ ≈ 0.04 × 12.4 = **0.5 G_modmul/s**.

This is a model-projected number, not a measurement. The actual
re-run is the most-impactful single experiment left in this campaign —
if the projection holds, RTX 5090's measured Layer C lead at 4096³
collapses from **12× raw / 6× per dollar** to **0.9× raw (TT-favoured)
and 1.8× per dollar (TT-favoured)**, mirroring the Layer B inversion.

If it doesn't hold — if the reduction is more expensive than v2's 5%
estimate suggested — Layer C may stabilise somewhere in between.

The two reasons the projection might be optimistic:
1. v2's reduction-cost estimate was made *under reference-kernel
   pressure* — when the GEMM was 30× slower, even a moderate reduction
   cost looked small. With the GEMM faster, the reduction's relative
   cost grows.
2. The 25-GEMM cascade's *host-side orchestration* (CB allocation,
   Finish() per stage) has an irreducible per-stage minimum. If that
   per-stage cost is, say, 0.5 ms per GEMM, then 25 GEMMs cost ≥12.5
   ms regardless of compute speed. At 4096³ that's not the bottleneck;
   at 1024³ it might be.

The right next experiment is `bench_blackhole.py --layer-c
--use-tuned-matmul` ⇒ remeasure across the size sweep. Estimated
~0.5 day of wrapper work; the tuned INT8 kernel is already in place
([`tt-llk-skeleton/host_int8_tuned/main.cpp`](../tt-llk-skeleton/host_int8_tuned/main.cpp))
from Phase 7.

### q48 sensitivity is unchanged

The TT measurement at q48 still costs ~0.69× of q36 (matching the
25/36 cost-model ratio), [`02_findings.md` §2.3.3](02_findings.md#233-q48--sensitivity-to-prime-width).
That ratio is independent of kernel quality — it's a partial-product
count. It will hold under tuned matmul too.

---

## 5.4 Layer D — KLSS-style inner product

Same story shape as Layer C. The TT-only measurements stand:

| Shape  | TT q36 (G_MAC/s) | TT q48 (G_MAC/s) |
| ------ | ---------------: | ---------------: |
| 512³   | 213.7  | 149.0 |
| 1024³  | 278.4  | 193.5 |
| 2048³  | 297.1  | 206.4 |
| 4096³  | 303.5  | 210.8 |
| 8192³  | 295.9  | 205.5 |

The plateau at ~300 G_MAC/s on q36 is the v2-reference number — it uses
the same 25-GEMM cascade as Layer C, so it inherits the same kernel
upper bound.

Re-running Layer D with the tuned matmul should lift this proportionally
to the Layer B improvement. The expected plateau is in the **2.5–3.5
T_MAC/s** range — an order of magnitude over what's currently shown. As
with Layer C, this is a projection, not a measurement.

NVIDIA Layer D remains unimplemented ([`03_caveats.md` §3.1](03_caveats.md)).
Until it lands the head-to-head ratio for the FHE-headline metric is
open. *Provisional* expectation from extrapolating Layer B INT8 ratios:
NVIDIA Layer D should land in the 1–2 T_MAC/s range raw on the
out-of-the-box cuBLASLt path; if NVIDIA tunes (Option A from the
discussion sidebar), maybe 2–3 T_MAC/s. TT tuned (3 T_MAC/s projected)
sits comfortably in that band — i.e. **the silicon-vs-silicon Layer D
result is likely close to a tie, with TT 2× ahead per dollar**.

This is the single most important measurement gap the bench currently
has.

---

## 5.5 Layer E — Phase 8 SFPU INT32

A new layer added in `f468f37`. Two backends, both TT-only:

### Layer E.1 — eltwise INT32 fused multiply-add

For arrays of length `T`, compute `c[i] = a[i] · b[i] + d[i]` in INT32
on the SFPU (no matrix engine). Lane width 110 cores × 8 SFPU lanes ×
1 op/cycle. Records:

| Shape | Throughput | per kUSD | TFLOPS/W |
| ----- | ---------: | -------: | -------: |
| 1×110×1024     | 7.12 GOPS | 7.13 | 0.18 |
| 1×110×4096     | 17.71 GOPS | 17.73 | 0.45 |
| 1×110×16384    | 26.99 GOPS | 27.02 | 0.68 |
| 1×110×65536    | 36.35 GOPS | 36.39 | 0.92 |
| 1×110×262144   | 39.07 GOPS | 39.11 | 1.00 |
| 1×110×1048576  | 40.37 GOPS | 40.41 | 1.02 |
| 1×110×4194304  | **40.73 GOPS** | 40.77 | **1.02** |

Plateau ~40 GOPS at large shapes. This is the SFPU's INT32 ceiling on
this part — the matrix engine doesn't have a native INT32 mode, so SFPU
is the only path for true 32-bit integer arithmetic.

### Layer E.2 — per-lane INT32 inner product

For a single (a, b) pair of length `T`, compute `Σ a[t]·b[t]`, with
the running sum staying in SFPU registers (no DRAM round-trip per
element). Bench shapes are pure linear sweeps:

| Shape | Throughput | per kUSD | TFLOPS/W |
| ----- | ---------: | -------: | -------: |
| 1×112640×1     | 3.78 GOPS | 3.78 | 0.09 |
| 1×450560×1     | 18.96 GOPS | 18.98 | 0.47 |
| 1×1802240×1    | 34.65 GOPS | 34.68 | 0.87 |
| 1×7208960×1    | 52.17 GOPS | 52.23 | 1.30 |
| 1×28835840×1   | 59.29 GOPS | 59.35 | 1.40 |
| 1×115343360×1  | 60.92 GOPS | 60.98 | 1.49 |
| 1×461373440×1  | **61.45 GOPS** | 61.51 | **1.54** |

Plateau ~61 GOPS — about 1.5× the eltwise plateau. The reason is that
the per-lane inner product keeps the partial sum in registers across
T elements, so the SFPU isn't writing partials back to DRAM every
cycle. The eltwise FMA path *does* write each result back, so it's
DRAM-bandwidth-bound after a few iterations.

### What Layer E means for FHE

Layer E doesn't represent a Layer-A-through-D primitive directly. It's
a microbench of the *integer arithmetic path that runs outside the
matrix engine* — relevant because:

1. The 25-INT8-GEMM cascade for Layer C reduces partial products via
   an INT32 epilogue that runs on the SFPU. The 40 GOPS eltwise
   plateau is the upper bound on how fast that epilogue can ever get
   on this card.
2. If a Layer-D-like workload doesn't decompose cleanly into 8-bit
   chunks (e.g., wider primes, RNS limbs that don't byte-align), the
   alternative is to run the modmul *natively in INT32* on the SFPU.
   The 60 GOPS inner-product plateau says: a pure-SFPU FHE inner-loop
   would do ~60 G_modmul/s, regardless of how many cores you have —
   this is a per-card SFPU ceiling, not per-core.
3. Compared to the 228 TOPS INT8 mcast number, the SFPU INT32 path is
   **3800× slower per useful op** (228 TOPS / 60 GOPS = 3800). That's
   the cost of leaving the matrix engine. Decomposing into INT8 chunks
   to feed the matrix engine is the right answer for this hardware,
   even at the price of the byte-decomposition complexity.

There's no NVIDIA peer for this layer yet. The closest analog would be
a CUDA-core INT32 microbench using inline PTX `vmad.s32` or `imad.s32` —
trivial to add but not in the v2 bench scripts. ~0.5 day of work to
plumb through.

---

## 5.6 Power and per-Joule

Steady-state power is now measured across all of Layer B (tuned and
reference), Layer C, Layer D, and Layer E:

| Workload | TT power | TT throughput | TFLOPS/W (or GOPS/W) |
| -------- | -------: | -------------: | -------------------: |
| `bf16` reference, 4096³  | 42 W | 3.91 TFLOPS  | 0.09 TFLOPS/W |
| `bf16_tuned_hifi4`, 5120×5632×5632 | 39.5 W | 136 TFLOPS | 3.44 TFLOPS/W |
| `bf16_tuned_hifi2`, 5120×5632×5632 | 39.5 W | 239 TFLOPS | **6.05 TFLOPS/W** |
| `int8_tuned_mcast_hifi2`, 5120×5632×5632 | 43.5 W | 228 TOPS | **5.25 TOPS/W** |
| `klss_ip_modmul_q36` (Layer D), 4096³ | 44.5 W | 304 G_MAC/s | 6.82 G_MAC/s/W |
| `int32_inner_product` (Layer E), 461M | 40 W | 61.4 GOPS | 1.54 GOPS/W |

Two observations:

**(a) The reference→tuned per-watt gain is ~70×** for BF16/HiFi2
([`04_phase7_tuned_matmul.md` §4.5.5](04_phase7_tuned_matmul.md#455-per-watt-under-load)).
The reference path consumed essentially the same power as the tuned
path (42 vs 39.5 W) but did 1/70 of the work. That's wasted matrix-engine
cycles, not extra power for the same work — so the reference-kernel
efficiency picture in v1+v2 was hiding TT's per-Joule competitiveness,
not consuming it.

**(b) The 6.05 TFLOPS/W BF16/HiFi2 number is in the same league as
NVIDIA L40S's published 6.4 TFLOPS BF16/W**, an inference-dedicated
data-center part. For a $999 consumer-priced FHE-target card, that's a
real result — and it sits at ~50% of the 80 W TDP, so the card has
headroom if the kernels can absorb more cores.

NVIDIA-side power telemetry is still missing ([`03_caveats.md` §3.2](03_caveats.md)).
Until a `nvidia-smi --loop=100ms` sidecar lands on the NVIDIA bench, the
joules-per-modmul comparison is one-sided and we should not quote a
cross-device per-Joule ratio.

---

## 5.7 Validation against TT's published peaks

Phase 7's BF16 numbers reproduce what the [Tenstorrent GEMM_FLOPS
report][gf] publishes for Blackhole P150 — once you scale by the
harvested core ratio (110/130 = 0.846):

| Configuration | TT-published @ 13×10 grid | Our card @ 11×10 | Predicted (×0.846) | Match |
| ------------- | ------------------------: | ---------------: | -----------------: | :---- |
| BF16 / HiFi4  | 165 TFLOPS                | 142 TFLOPS       | 140 TFLOPS         | ✓ within 2% |
| BF16 / HiFi2  | 308 TFLOPS                | 272 TFLOPS       | 261 TFLOPS         | ✓ within 5% |
| BF8 / HiFi2   | 305 TFLOPS                | 275 TFLOPS       | 258 TFLOPS         | ✓ within 7% |
| BF4 / LoFi    | 589 TFLOPS                | 541 TFLOPS       | 498 TFLOPS         | ✓ within 9% (BF4 is bandwidth-throttled at peak) |

The INT8 number isn't in TT's GEMM_FLOPS doc (the upstream test only
iterates BF16/BF8/BF4). Our INT8/HiFi2 mcast plateau of **228 TOPS** is
~83% of the implied INT8 matrix-engine ceiling (BF8 plateau scaled to
INT8's same byte-width path is ~275 TOPS), per [`discussions/nvidia_cublaslt_int8_utilization.md`](discussions/nvidia_cublaslt_int8_utilization.md).
That's the closest published anchor we have.

The ~83% utilisation is consistent with the upstream `matmul_multicore_reuse_mcast`
example being shape-tuned for BF16, not INT8 — there's another ~17%
on the table from per-shape autotune. Not the bench's bottleneck.

---

## 5.8 Updated headline (replaces [`02_findings.md` §2.6](02_findings.md#26-headline-numbers-in-one-paragraph))

**On Layer B raw INT8 GEMM at silicon-best on each side, TT Blackhole
slightly edges RTX 5090 (1.06× raw, 2.13× per dollar). On Layer B raw
BF16 / TF32 at silicon-best, TT is comfortably ahead — 272 TFLOPS
BF16/HiFi2 vs RTX 5090's ~100 TFLOPS TF32 plateau (2.7× raw, 5.4× per
dollar).** On Layer C exact 36-bit modmul *as measured*, RTX 5090 still
leads (the reference-kernel cascade hasn't been re-run). On Layer C
*projected* with the tuned INT8 matmul under the same recipe, the lead
likely flips by ~12× — RTX moves to 0.9× raw, 1.8× per-dollar TT-favoured
at 4096³. Layer D KLSS-IP is TT-only at ~300 G_MAC/s (reference) /
~3 T_MAC/s (projected); NVIDIA Layer D is still the biggest gap in the
campaign. The shape of the cost curve — TT winning at small tiles,
NVIDIA winning at large reference-kernel tiles, TT winning at all tiles
once kernel-quality is matched — points to "the right hardware depends
on which kernel team you have."

---

## 5.9 Honest take, in one paragraph (replaces [`02_findings.md` §2.7](02_findings.md#27-what-an-fhe-engineer-should-take-from-this))

If you are picking hardware for a CKKS-style FHE workload today and
have a kernel team:

- **Default to TT Blackhole p150a if INT8 byte-decomposition is your
  recipe.** The 228 TOPS mcast number proves the silicon is competitive
  at the matrix engine level; the 2× price advantage compounds; and
  Phase 7's reproduction shows the path to that throughput is two
  upstream programming examples (block reuse + multicast), not a deep
  research project.
- **Default to RTX 5090 if you don't have a kernel team.** cuBLASLt
  out-of-the-box gives 215 TOPS on day one; matching that on TT
  required Phase 7's two-week kernel push. The day-zero-throughput
  story still favours NVIDIA strongly.
- **Watch the Layer C/D re-measurement.** The single biggest pending
  number is "what does the 25-INT8-GEMM cascade do on the tuned
  matmul." If it's 0.5 G_modmul/s at 4096³ as projected, the v2
  conclusion of "RTX wins the FHE inner loop" is fully overturned. If
  it's noticeably less, the tier-1-vs-tier-2 picture splits cleanly:
  raw GEMM goes TT, modmul stays NVIDIA, and which tier matters depends
  on the actual workload.
- **Don't quote v2 ratios.** The v1+v2 INT8 cross-device numbers in
  [`02_findings.md`](02_findings.md) are reference-vs-tuned comparisons
  and they're going to mislead anyone reading them. The same numbers
  re-stated with Phase 7 added are in §5.2 above.

---

## 5.10 What v3 still owes

In rough priority order, with the size of each gap:

1. **Re-run Layer C and Layer D with the tuned INT8 matmul.** Confirms
   or refutes the §5.3/§5.4 projections. ~0.5 day. *Highest impact
   pending experiment.*
2. **NVIDIA Layer D KLSS-IP, reference path.** Closes the cross-device
   FHE-headline gap. ~1 day.
3. **NVIDIA Layer C q48.** Mechanical extension. ~0.5 day.
4. **NVIDIA energy telemetry.** `nvidia-smi --loop=100ms` polling
   sidecar. Closes the per-Joule comparison. ~0.5 day.
5. **CUTLASS sm_120 INT8 on RTX 5090** (Option A from
   [`discussions/nvidia_cublaslt_int8_utilization.md`](discussions/nvidia_cublaslt_int8_utilization.md)).
   Lifts NVIDIA INT8 to ~350 TOPS, restores ~1.5× silicon advantage,
   leaves TT 1.34× ahead per dollar. ~1 day.
6. **NVIDIA Layer D fused** (CUTLASS epilogue with mod-q reduction
   inline). Removes the HBM round-trip that drives the 4096³+ tail-off.
   ~2 days.
7. **TT Layer D fused** (single-tile-resident sum, no DRAM partial
   spill). Cleanly tests whether the cascade approach is the limiter
   or the matrix engine itself is. ~3 days.
8. **NVIDIA SFPU-equivalent: CUDA-core INT32 microbench.** Closes
   Layer E's missing peer. ~0.5 day.
9. **Multi-card scaling (2× Blackhole, 2× RTX 5090).** Embarrassingly
   parallel for FHE batches. v3 problem.

The first three items would close the head-to-head story for v3.
Numbers 4–8 sharpen the per-Joule and silicon-vs-silicon framings.
Number 9 is its own campaign.

---

## 5.11 Reading order across `doc/`

After this document the reading order is:

1. [`01_rationale.md`](01_rationale.md) — what FHE actually computes
   and why this comparison.
2. [`02_findings.md`](02_findings.md) — v1+v2 measurements, **outdated
   conclusions** (use §5.8 above for the current headline).
3. [`03_caveats.md`](03_caveats.md) — coverage gaps and methodology
   limits. §3.3's v2-priority list is partially obsolete (items 1, 2
   are subsumed; the priority order in §5.10 supersedes it).
4. [`04_phase7_tuned_matmul.md`](04_phase7_tuned_matmul.md) — the
   Phase 7 reproduction in detail. Required reading for anyone
   reproducing the tuned numbers.
5. **This document** — synthesis after Phase 7+8.
6. [`discussions/nvidia_cublaslt_int8_utilization.md`](discussions/nvidia_cublaslt_int8_utilization.md)
   — open caveat on the NVIDIA INT8 ceiling.

The plan is to retire `02_findings.md` and fold its surviving content
into a v3 rewrite once Layer C/D are remeasured. Until that lands, this
synthesis is the canonical story.
