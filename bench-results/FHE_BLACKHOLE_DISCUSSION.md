# Tenstorrent Blackhole for FHE — what the numbers say

This document interprets [`SUMMARY.md`](SUMMARY.md) for the FHE-relevant
question: *is a $999 P150 a viable target for the inner-loop arithmetic of
a CKKS / BFV / KLSS pipeline, today?* It cross-checks the bench against
Tenstorrent's official GEMM_FLOPS report so the reader can see which
gaps are "the bench is undertuned" vs "the architecture is the wrong
shape."

References used for sanity-checking:
- TT [GEMM_FLOPS tech report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md)
- TT [eltwise_sfpu programming example](https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/programming_examples/eltwise_sfpu/eltwise_sfpu.md)

## 1. What Blackhole actually is

Blackhole P150 is a 13×10 = 130 Tensix grid at 1.35 GHz. Each Tensix
contains:

- a **matrix engine** (FPU) that does an `8×16 @ 16×16 → 8×16` tile
  product per cycle — 4096 MACs/cycle = **8192 FLOPs/cycle at LoFi**;
- a **SFPU** vector unit (~32 lanes per Tensix) for elementwise ops like
  `exp`, `recip`, integer ALU work, and any kernel that doesn't decompose
  into the FPU's tile-product shape.

Per-Tensix matrix-engine peak from the TT doc:

| Math fidelity | BH per-engine |
|---|---|
| LoFi (BFP4) | ~5.4 TFLOPS |
| HiFi2 (BFP8) | ~2.7 TFLOPS |
| HiFi4 (BF16) | ~1.35 TFLOPS |

Per-board peaks (130 cores, manually tuned, L1-resident operands) measured
by Tenstorrent:

| Dtype / fidelity | BH P150 peak |
|---|---|
| BF16 / HiFi4 | ~168 TFLOPS |
| BFP8 / HiFi2 | ~580–625 TFLOPS |
| BFP4 / LoFi | ~640 TFLOPS |

Critical point for this discussion: **TT's GEMM_FLOPS doc never reports
INT8 or INT32 matmul peaks.** The matrix engine is a BF16 / block-FP
engine. Any "INT8 TOPS" number on Blackhole is either an emulation of
INT through the FPU's BFP path, or a packed-element trick on the SFPU.
There is no published Tenstorrent equivalent of Ampere's INT8 tensor
core to anchor against.

This is the architectural fact that frames everything below.

## 2. Validating the bench against TT's published peaks

Our [`SUMMARY.md`](SUMMARY.md) Layer B numbers at 4096³:

| Backend | BH measured | TT-published peak (P150) | Utilization |
|---|---|---|---|
| BF16 (HiFi4) | 3.91 TFLOPS | ~168 TFLOPS | **~2.3%** |
| Matrix-engine FP32 | 1.98 TFLOPS | (not listed — TF32-internal) | n/a |
| INT8 (FPU-emulated) | 7.57 TOPS | (not characterized by TT) | n/a |

The 2.3% BF16 utilization is the headline issue. It's not subtle. There
are four known multipliers of headroom that we're *not* claiming yet:

1. **Grid size.** Records report `n_cores: 110` (an 11×10 layout), not
   the full 13×10 = 130 used by TT's reference benchmark. Pure linear
   scaling would give ~118% (still off by 35–40×).
2. **No tracing.** TT explicitly calls out tracing as the host-overhead
   removal that "helps recover lost performance on smaller tensor
   matmuls." Our LLK skeleton dispatches per-iter from the host.
3. **No L1 sharding / DRAM-resident operands.** TT's 168 TFLOPS line
   uses `in0_storage_type=L1, out_storage_type=L1`. The bench's
   [`tt-llk-skeleton/host/main.cpp`](../tt-llk-skeleton/host/main.cpp)
   path is DRAM-resident.
4. **Math fidelity left at HiFi4.** Our Layer B BF16 stays at HiFi4 by
   construction; the higher headline numbers (580 TFLOPS) are BFP8/HiFi2,
   which is a different point in the precision–throughput envelope and
   irrelevant for an FHE workload that needs a known multiplicand
   bit-width.

The 42–45 W board-power readings during compute (Layer D) corroborate
this: P150's TDP is in the 300 W class, so the bench is running the
engine at roughly 15% of TDP — exactly what you'd expect when the
matrix engine is sitting idle waiting on host dispatch.

**So: the bench is honest, and it is undertuned.** The gap between 3.9
TFLOPS and 168 TFLOPS is engineering, not physics. The numbers in
SUMMARY.md should be read as a lower bound on what a properly traced,
L1-sharded port of this work would show.

A small caveat: the SFPU FP32 backend slot
(`tt_llk_sfpu_fp32`) is a placeholder — every record has
`throughput: null` and `device_detail.error: "backend kernel TODO"`.
TT's `eltwise_sfpu` example confirms the SFPU is for elementwise ops
(`exp` etc.) and isn't a substitute for matrix-engine matmul; expecting
high TFLOPS there would be a category error.

## 3. Layer-by-layer FHE relevance

The bench is structured as four layers (`bench-results/SUMMARY.md`):

- **Layer A — capability probe.** Single shape, every backend on every
  device. Sanity-only.
- **Layer B — raw GEMM throughput envelope.** What the device can do
  per dtype, before we ask anything cryptographic of it.
- **Layer C — exact bit-correct modular product.** *q36* (36-bit prime,
  25 INT8 GEMMs/modmul) and *q48* (48-bit prime, 36 INT8 GEMMs/modmul).
  This is the FHE inner-loop atom.
- **Layer D — KLSS-style inner product.** A vector-IP-with-modular-reduction
  workload — the headline metric for an actual FHE kernel.

### Layer C is where FHE lives

Layer C uses GEMM-emulated bit-exact integer modmul: 25 INT8 GEMMs per
36-bit `(a*b) mod q`, 36 GEMMs for 48-bit. The correctness gate
(`bench-results/SUMMARY.md`, "passed" column) confirms bit-exactness
against a Python `int` reference for the adversarial input set
(boundary, `2^k ± 1`, near-quotient, random) — so the algorithm is
sound; what we're measuring is throughput on top of correctness.

At 4096³:

| Device | Backend | q36 modmul throughput | $/k$ adjusted |
|---|---|---|---|
| Blackhole | int8 (FPU-emulated) | 0.04 G_modmul/s | 0.037 |
| RTX 5090 | cuBLASLt INT8 (tensor core) | 0.45 G_modmul/s | 0.224 |

The 12× gap (6× price-adjusted) on Layer C is the same shape as the
Layer B INT8 gap (~28× raw, ~14× price-adjusted), which is what you'd
expect since Layer C is "Layer B INT8 wrapped 25 times." The
*architectural* gap is the absence of a true tensor-core-grade INT8
path on Blackhole: the FPU has to emulate INT through its BFP8 path,
losing both fidelity-mode-specific peak (HiFi2 = 2.7 TFLOPS/engine vs
LoFi 5.4) and the ability to use the cleaner INT8 tensor-core path the
RTX 5090 has.

q48 vs q36: q48 takes 36/25 = 1.44× more GEMMs per modmul, and the
measured slowdown at 4096³ is 0.04 → 0.03 G_modmul/s = ~1.4× — the
ratio matches the algorithmic GEMM count almost exactly, which is
another small confirmation that the bench is GEMM-bound rather than
memory- or correctness-overhead-bound.

### Layer D is the only number that flatters Blackhole

| Op | Shape | BH throughput | Energy |
|---|---|---|---|
| `klss_ip_modmul_q36` | 4096³ | 303.5 G_MAC/s | 6.82 G_MAC/s/W |
| `klss_ip_modmul_q48` | 4096³ | 210.8 G_MAC/s | 4.69 G_MAC/s/W |

At ~6.8 G_MAC/s/W and 44 W, BH is doing useful FHE work at modest
power. We don't have an RTX 5090 Layer D number to compare against
(the `bench_nvidia.py` Layer D path was added but no nvidia run has
been collected yet), so the per-Joule advantage that *might* exist
here is conjecture — it's the obvious next number to take.

## 4. Merits of Blackhole for FHE

Stated honestly, given the data we have:

- **Programmability of the per-tile kernel.** The LLK / Metalium
  stack lets you write the modular-reduction kernel directly, with
  control over which lanes, which fidelity, and which packer/unpacker
  path are used. RTX 5090 INT8 modmul rides cuBLASLt — a closed
  black box.
- **Open software stack.** TT-Metal, TT-LLK, and the tech reports are
  on GitHub. The bench's correctness gate exists *because* we could
  read the LLK source to know which BFP path the FPU was taking;
  there is no equivalent transparency for cuBLASLt's INT8 codepath.
- **Price.** $999 vs $1999. Layer C/D ratios show this matters: at
  Layer D our q36 number is ~300 G_MAC/s for $999, putting BH on the
  per-dollar Pareto frontier even at 2.3% of its own peak.
- **Power envelope.** Sustained 42–45 W during compute (vs ~300 W TDP)
  means the device has a lot of headroom. If the engine reaches even
  20% utilization, throughput-per-watt goes up while throughput-per-dollar
  stays the same — the cost story improves.
- **Per-Joule density at scale.** A multi-board P150 deployment
  trades raw single-card throughput for many-card concurrency; for
  embarrassingly-parallel FHE batches (which is most of them) this
  composes well.

## 5. Demerits of Blackhole for FHE

Equally honestly:

- **No native INT8 / INT32 matmul.** The architectural fact. FHE
  inner-loop operations *are* exact integer arithmetic; the device's
  matrix engine is a BF16 / BFP4 / BFP8 engine. Every modmul must be
  emulated as ≥25 GEMMs. This is the ceiling.
- **SFPU isn't a tensor core.** The eltwise_sfpu example is exactly
  what its name says — elementwise ops, ~32 lanes per Tensix. A
  cryptographic kernel that wants to run a 256-bit reduction lane-wise
  on the SFPU is going to be an order of magnitude slower than the
  same workload on a card with native wide-integer ALUs. This isn't
  recoverable with kernel tuning.
- **Software immaturity for non-AI workloads.** TT's GEMM_FLOPS
  numbers come from `pytest tests/ttnn/unit_tests/benchmarks/` —
  TTNN-level kernels with hand-tuned configs. The LLK skeleton in
  this repo lives one layer below that. Closing the 50× gap from
  3.9 TFLOPS to 168 TFLOPS BF16 means either rewriting on TTNN (and
  losing the bit-exact integer guarantees TTNN doesn't expose) or
  doing the L1-sharding / tracing / fidelity work ourselves. Both
  are real engineering bills.
- **No published TT FHE/integer reference.** All of the comparisons
  in this repo are *our* numbers vs *NVIDIA's* documented INT8
  tensor-core peak. Nothing TT publishes covers this regime, so we
  can't anchor "is 7 TOPS at 4096³ what Blackhole *should* do for
  emulated INT8?" against a vendor claim.

## 6. Verdict

**For a research prototype: yes, with the right framing.** Blackhole
is a per-tile-programmable accelerator at half the price of an RTX 5090,
and the bit-exact correctness gate proves the FHE primitive *can* be
realized on it. Layer D's 300 G_MAC/s at 44 W is a defensible
demonstration that the device is not a dead end for FHE.

**For a production FHE deployment today: no.** The 12–28× gap on the
Layer B/C metrics that *do* correspond to real FHE bottlenecks is too
large, and most of it is structural (no native INT matmul) rather than
recoverable through kernel tuning. The portion that *is* recoverable
(2.3% → maybe 50–80% of BF16 peak via tracing + L1 sharding) closes the
BF16 gap but doesn't help Layer C, since INT modmul is GEMM-bound on
the FPU's BFP path no matter what fidelity you pick.

**The honest single-sentence answer:** Blackhole is a credible
research target for FHE *if* you're willing to fund the LLK kernel
work and accept a ~5–10× per-card raw-throughput disadvantage in
exchange for a 2× price advantage and an open stack. Otherwise the
RTX 5090 INT8 tensor-core path is the better hardware fit.

## 7. What would change the picture

Concrete next measurements (in order of likely impact):

1. **Re-run Layer B BF16 with tracing + L1-sharded operands.** Target:
   ≥80 TFLOPS at 4096³ (~50% of TT's published peak). If we get there,
   re-run Layer C; the GEMM-bound expectation is ~12 G_modmul/s at
   4096³ q36, which is 25× current and would change the Layer C verdict
   above.
2. **Collect an RTX 5090 Layer D run.** The `bench_nvidia.py` Layer D
   path exists but no JSONL has been emitted; the per-Joule comparison
   is the cleanest "is BH worth it" answer we can give and we don't
   currently have it.
3. **Implement the SFPU FP32 backend.** Right now `tt_llk_sfpu_fp32`
   is a TODO. Even a slow number is better than `null`, because it
   sets the lower bound on the SFPU path and lets us reason about
   modmul implementations that route reduction through the SFPU
   instead of the FPU.
4. **Add a 2× P150 multi-board run.** Embarrassingly-parallel FHE
   batches should scale near-linearly; if they don't, that's a much
   more interesting result than the single-card numbers.
