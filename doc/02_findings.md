# 2 — Findings

**Takeaway.** With the v1 data in (1 NVIDIA host run, 3 Tenstorrent host
runs covering BF16, INT8, and KLSS / FP32-matrix), the picture is:

1. **At small shapes (512³), Tenstorrent Blackhole INT8 wins per dollar
   on Layer C exact modmul** by ~1.24×.
2. **From 1024³ upward, RTX 5090 dominates** by 3–9× per dollar on
   Layer C and 14× per dollar on raw INT8 GEMM (Layer B, 8192³).
3. **Layer D — KLSS-like inner product — is where TT looks strongest**.
   Blackhole hits ~300 G_MAC/s at 4096³ on q36. NVIDIA Layer D is not
   yet implemented, so the head-to-head ratio is open.
4. **q48 costs ~30% more** than q36 on TT (per-shape, both layers C
   and D). Going from a 36-bit prime to a 48-bit prime is tractable.
5. **TT Blackhole steady-state power is 41–46 W** (median ~45 W),
   against the p150a's 80 W TDP — meaning the workload is using
   roughly half the silicon's power envelope and there is sizable
   headroom available if the bench is the limiter.

The rest of this document walks through the data layer by layer and
explains what each plot is trying to show.

The numbers cited below are pulled from
[`bench-results/SUMMARY.md`](../bench-results/SUMMARY.md). When the
data is refreshed, the numbers here may go stale; the SUMMARY is
authoritative.

---

## 2.1 Layer A — Capability probe

| Backend                          | RTX 5090 1024³ | Blackhole 1024³ | Notes                                                |
| -------------------------------- | -------------: | --------------: | ---------------------------------------------------- |
| INT8 → INT32                     |     76.4 TOPS  |     6.5 TOPS    | Both via tensor / matrix engine                      |
| TF32 / `tt_llk_fp32_matrix`      |     76.9 TFLOPS | 1.9 TFLOPS     | Same backend class: tensor-engine fast-math FP32     |
| BF16                             |             —  |     3.7 TFLOPS  | TT only — NVIDIA does not run BF16 on this script    |
| FP32 strict (CUDA-core / SFPU)   |     48.1 TFLOPS | _SFPU TODO_    | TT SFPU FP32 kernel not yet implemented              |
| FP64                             |     1.49 TFLOPS |        —       | Confirms FP64 is **not** tensor-accelerated on consumer Blackwell |

The plausibility-check flag never triggered: every measured throughput
is within 5× of the published spec. In particular, NVIDIA's FP64 at
1.49 TFLOPS confirms BENCHMARK.md §3's claim — consumer Blackwell does
*not* have a tensor-accelerated FP64 path. That rules out a "just use
FP64 GEMM directly" workaround for 36-bit modmul on this card.

cuBLASLt selected algorithm IDs: TF32 = 21, FP32 = 20, FP64 = 22,
with 64×64 / 128×32 / 32×32 tiles respectively. The metadata is logged
in `device_detail.backend_detail` of every Layer A record.

---

## 2.2 Layer B — Raw GEMM throughput

The full sweep (5 sizes × every backend per device) is in
[`layer_b_throughput.png`](../bench-results/layer_b_throughput.png).
The peak-throughput numbers at 8192³:

| Backend class                    | RTX 5090         | Blackhole          | Ratio (raw) | Ratio (per kUSD) |
| -------------------------------- | ---------------: | -----------------: | ----------: | ---------------: |
| INT8 Tensor Core / Tensix        | 210 TOPS         | 7.4 TOPS           |       28.4× |            14.2× |
| TF32 Tensor Core / Tensix matrix | 101 TFLOPS       | 1.9 TFLOPS         |       52.7× |            26.3× |
| BF16 Tensor / Tensix             | —                | 3.9 TFLOPS         |       —     |            —     |
| FP32 strict (CUDA core / SFPU)   | 62 TFLOPS        | _SFPU TODO_        |       —     |            —     |
| FP64                             | 1.58 TFLOPS      | —                  |       —     |            —     |

**Reading the INT8 row**: at peak, RTX 5090 is 28× faster per chip and
14× faster per dollar. That is the upper bound on what Layer C can
do for either device — Layer C cannot beat the underlying GEMM, only
add overhead on top of it.

**Reading the TF32 row**: Both devices have a tensor-engine fast-math
FP32 path. NVIDIA's `cublaslt_tf32` and TT's `tt_llk_fp32_matrix` both
take FP32 inputs and use TF32-internal fidelity (NVIDIA documents this
explicitly in the cuBLASLt manual; Tenstorrent documents the same in
their `fp32_accuracy.md`). They are mapped to the same `tf32`
backend-class so the ratio is apples-to-apples. NVIDIA dominates by
~53× raw / ~26× per dollar.

**Reading the strict-FP32 row**: NVIDIA's `cublaslt_fp32`
(IEEE-compliant CUDA-core FMA, *not* tensor-core) hits 62 TFLOPS —
~60% of the 104 TFLOPS spec. Blackhole's strict-FP32 path goes
through the SFPU, which is still TODO in the skeleton; the wrapper
emits a "skipped" record with the TODO error message rather than
substitute a wrong number. That is honesty by design, not a
measurement gap.

---

## 2.3 Layer C — Exact 36-bit modular product

This is the headline FHE-relevance layer.

### 2.3.1 q36, best backend per device

[`layer_c_modmul.png`](../bench-results/layer_c_modmul.png) shows the
shape sweep. Numbers (best backend per device):

| Shape  | RTX 5090 INT8 | Blackhole INT8 | Ratio (raw) | Ratio (per kUSD) |
| ------ | ------------: | -------------: | ----------: | ---------------: |
| 512³   |  0.34 G_modmul/s |  0.21 G_modmul/s |    1.62× |        **0.81×** ← TT wins per dollar |
| 1024³  |  0.84         |  0.14          |       6.21× |             3.10× |
| 2048³  |  0.85         |  0.07          |      11.74× |             5.87× |
| 4096³  |  0.45         |  0.04          |      12.09× |             6.04× |
| 8192³  |  0.33         |  0.018         |      18.04× |             9.01× |

The crossover at 512³ — visible as the curves intersect near the y-axis
in [`layer_c_modmul_per_dollar.png`](../bench-results/layer_c_modmul_per_dollar.png) —
is the only operating point in the v1 data where Blackhole's price
advantage overcomes its raw-throughput deficit. From 1024³ upward, RTX
5090 is the better deal.

### 2.3.2 The 4096³+ tail-off on RTX 5090

RTX 5090's Layer C throughput peaks at 0.85 G_modmul/s near 1024–2048³,
then falls to 0.45 at 4096³ and 0.33 at 8192³. The corresponding raw
INT8 GEMM throughput (Layer B) goes the other way — increasing from
76 TOPS at 1024³ to 210 TOPS at 4096³.

The Layer C drop is therefore not a GEMM problem. The most likely
cause is bandwidth: each of the 25 INT8 GEMMs produces an `int32`
partial of size `M×N×4 bytes`. At 4096² that is 64 MB written per
partial; 25 partials per matmul is 1.6 GB of write traffic over the
matmul wall-time. At 8192² it is 6.4 GB. RTX 5090's HBM bandwidth (~1.8
TB/s) and L2 cache do not absorb that write traffic the way they
absorb a single fused INT8 GEMM, so the Layer C path becomes
write-bandwidth-bound.

The fix is a **fused** Layer C kernel that keeps partials in registers
or shared memory and never materializes the `int32` partials to HBM.
That is on the v2 list (CUTLASS epilogue territory).

### 2.3.3 q48 — sensitivity to prime width

Blackhole has q48 numbers. RTX 5090 does not (deferred to v2). Looking
at TT alone:

| Shape  | TT q36 | TT q48 | q48 / q36 |
| ------ | -----: | -----: | --------: |
| 512³   |  0.21  |  0.15  |     0.69× |
| 1024³  |  0.14  |  0.09  |     0.69× |
| 2048³  |  0.07  |  0.05  |     0.69× |
| 4096³  |  0.04  |  0.03  |     0.69× |
| 8192³  |  0.018 |  0.012 |     0.69× |

q48 is ~31% slower than q36 on TT, almost flat across shapes.

The cost-model expectation: q36 needs `5×5 = 25` byte cross-products
per output element; q48 needs `6×6 = 36`. The ratio 25/36 = 0.694 — i.e.
q48 should run at 69% of q36's speed if everything else is constant.
**The measurement matches the model to two significant figures**, which
is reassuring: Blackhole's Layer C cost is dominated by the partial-
product count, not by the per-pair shift/reduce overhead. That suggests
the same recipe scales predictably to wider primes.

(The q36-vs-q48 overlay is in
[`layer_c_q36_vs_q48.png`](../bench-results/layer_c_q36_vs_q48.png).
RTX 5090's curve is on the same plot but only at q36, since v1 didn't
include q48 on the NVIDIA side.)

---

## 2.4 Layer D — KLSS-like inner product

[`layer_d_klss_ip.png`](../bench-results/layer_d_klss_ip.png) shows
the TT-only Layer D measurements. Numbers:

| Shape  | TT q36 G_MAC/s | TT q48 G_MAC/s |
| ------ | -------------: | -------------: |
| 512³   |          213.7 |          149.0 |
| 1024³  |          278.4 |          193.5 |
| 2048³  |          297.1 |          206.4 |
| 4096³  |          303.5 |          210.8 |
| 8192³  |          295.9 |          205.5 |

Three observations.

**(a) Layer D throughput rises with shape** (213 → 303 G_MAC/s for q36),
opposite to Layer C's 4096³ tail-off. The reason is the same one but
inverted: in Layer D, the K-dimension reduction means the cost is
amortized over `K` useful MACs per output element. Wall-clock per
matmul is identical between Layer C and Layer D (e.g. 452 ms at 4096³
on TT, exactly the same number for both layers because they share the
same 25-GEMM core), but Layer D *credits* that wall-clock with K=4096
useful MACs per output, where Layer C credits it with one. The metric
ratio is approximately `2K`, which at 4096³ is ~8000×, matching what
we see (303 G_MAC/s vs 0.04 G_modmul/s).

This is the conceptual point KLSS papers emphasize: the per-tile cost
of a 36-bit modmul is roughly fixed by the byte decomposition, so the
right way to use a hardware matrix engine is to maximize the K
dimension within a tile, which is exactly what an inner-product
formulation does.

**(b) q48 costs 69% of q36**, again matching `25/36 = 0.69`. Same
cost-model story as Layer C, just at a higher absolute level.

**(c) NVIDIA Layer D is missing.** That is the v1 plan's biggest gap.
The TT KLSS-IP number (~300 G_MAC/s) is striking, but until NVIDIA
lands the same workload it can't be normalized into a head-to-head
ratio. *Provisional* expectation, based on Layer B INT8 ratios, is
that NVIDIA Layer D should be 5–10× the TT number raw and 2–5× per
dollar — so RTX 5090 would still likely win — but this is a guess
until measured.

---

## 2.5 Power and energy

Schema v2 added `power_w_avg` and `joules_per_useful_op` fields. TT
populates them; NVIDIA does not yet. From the TT side:

- Steady-state TT Blackhole power during compute: 41–46 W, median ~45 W
  across 78 records covering all the implemented backends and shapes.
- Blackhole p150a TDP: 80 W. The bench is therefore using ~55% of the
  silicon's headroom — if the kernels are compute-bound and not
  bandwidth-bound, this implies further optimization (better SRAM
  shuffling, more parallel cores active) could push throughput
  proportionally higher. If they are *bandwidth*-bound instead, no
  further power is available and additional throughput must come from
  algorithmic changes.

NVIDIA equivalents (`nvidia-smi`-polled power) are deferred to v2;
without them the joules-per-modmul comparison is one-sided. We do not
quote a number that requires both sides yet.

---

## 2.6 Headline numbers, in one paragraph

**On Layer C exact 36-bit modmul, RTX 5090 wins by 1.62× to 18× raw and
0.81× to 9.01× per dollar across the 512³–8192³ shape range. The only
shape where Blackhole wins per dollar is 512³ (1.24× advantage). On
Layer B raw INT8 GEMM, RTX 5090's lead is ~28× raw, ~14× per dollar at
peak (8192³). Layer D is currently TT-only at ~300 G_MAC/s on q36,
~210 G_MAC/s on q48; NVIDIA Layer D is the next gap to fill.** The
shape of the cost curve — RTX winning at large tiles, Blackhole
winning at small tiles — suggests workload-dependent answers more than
a one-line verdict.

---

## 2.7 What an FHE engineer should take from this

If you are picking hardware for a CKKS-style FHE workload today:

- **Default to RTX 5090.** It wins on every metric except per-dollar at
  512³, and the per-dollar gap at 1024+ is too large to ignore.
- **Reconsider if your tile sizes are small.** If the workload is
  dominated by 512×512-ish modular GEMMs (e.g. specific bootstrapping
  configurations or low-degree CKKS rings), Blackhole's per-dollar
  lead at that shape is real. Verify on your actual workload.
- **Watch the Layer D number.** Once NVIDIA implements its KLSS-IP
  fused path, the TT side's ~300 G_MAC/s peak may or may not be
  defensible. The architecture has SRAM-locality and core-count
  advantages there that don't show up in Layer B / Layer C.
- **Don't pick on raw TOPS alone.** RTX 5090's 210 INT8 TOPS sounds
  enormous but only ~0.4% of those operations turn into useful 36-bit
  modmuls in the v1 recipe. The headline metric in
  [`SUMMARY.md`](../bench-results/SUMMARY.md) is the one that maps
  to wall-clock CKKS performance.
