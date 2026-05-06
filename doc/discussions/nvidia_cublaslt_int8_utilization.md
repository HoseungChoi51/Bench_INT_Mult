# NVIDIA cuBLASLt INT8 measurement vs the published peak

**Status:** open caveat on the Phase 7 INT8 cross-device comparison
in [`../04_phase7_tuned_matmul.md`](../04_phase7_tuned_matmul.md) §4.5.4.
Not yet incorporated into the main narrative — this file holds the
discussion until we either run the experiment in §"Option A" below or
decide to add a caveat in the main doc.

## What we measure on RTX 5090

`scripts/bench_nvidia.py` runs INT8 GEMM via `torch._int_mm`, which
dispatches to **cuBLASLt's IGEMM kernel**. The plateau on the RTX 5090
($1999, Blackwell consumer GB202) is:

- 4096³ : **214.5 TOPS**
- 8192³ : **210.1 TOPS**

(Saturated — both shapes deliver the same throughput, so the curve has
flattened.)

## What NVIDIA's spec sheet claims

The RTX 5090 product page advertises **838 TOPS INT8 Tensor**. There's a
crucial qualifier most people miss:

> Tensor Core throughput numbers in NVIDIA marketing tables are quoted
> **with 2:4 structured sparsity**, which doubles the dense rate.

So the dense (non-sparse) INT8 ceiling is:

- **838 / 2 = ~419 TOPS** (dense)

Our 214.5 TOPS measurement is **~51% of dense peak** (and ~26% of the
with-sparsity number, but that's not the relevant comparison since we
don't enable sparsity on either side).

## Is 51% reasonable for cuBLASLt IGEMM?

Yes. It's the expected utilization for a production library path on a
random-shape INT8 GEMM. Approximate utilization tiers for INT8 on
NVIDIA Tensor Core GPUs:

| Path | Expected % of dense peak | Notes |
|---|---|---|
| `torch._int_mm` (cuBLASLt IGEMM) | **40–60 %** | What we measured (51 %). |
| Hand-tuned CUTLASS w/ sm_120 INT8 kernels | 70–85 % | Per-shape autotune; substantial integration cost. |
| TensorRT-LLM INT8 fused kernels | 75–90 % | Inference-stack-specific; not a general PyTorch matmul. |
| Marketing peak ("with 2:4 sparsity") | 100 % (definitionally) | Different problem — half the input tokens are zero. |

Reaching the dense Tensor Core ceiling reliably needs CUTLASS or TRT-LLM
kernels per shape. cuBLASLt is what every ML engineer reaches for first
because it's one Python call and works everywhere; the price for that is
~30–40 percentage points off the silicon peak.

For comparison, cuBLAS's BF16 GEMM through `torch.matmul` typically
lands at **~70 % of BF16 Tensor Core peak** on the same generation —
better than INT8 because the BF16 paths are more mature. INT8 is a
narrower-precision, less-mature fast-path on consumer cards; 51 % is in
line with what other INT8 microbenchmarks see on Blackwell consumer
parts (Ampere INT8 cuBLAS measurements at ~55 % were the going rate).

## The asymmetry vs the TT measurement

The Phase 7 INT8 mcast measurement on Blackhole p150a hits **228.5 TOPS
(HiFi2)** at 5120×5632×5632. Where does that sit relative to TT's
silicon peak?

The Phase 7 BF16 numbers establish the matrix engine's near-peak
utilization on this card:

- BF16/HiFi4 tuned: 142 TFLOPS (94 % of the 130-core published 165 TFLOPS,
  scaled to our harvested 110 cores)
- BF16/HiFi2 tuned: 272 TFLOPS (90 %)
- BF8/HiFi2 tuned: 275 TFLOPS (90 %)

INT8 has the same byte width as BF8 and runs through the same matrix
engine path; the implied INT8 ceiling at our grid is **~275 TOPS**. Our
228.5 TOPS measurement is **~83 % of that ceiling** — close to peak,
not at peak. The remaining ~17 % gap is some mixture of:

- the upstream `matmul_multicore_reuse_mcast` programming example
  isn't fully shape-tuned (it picks block sizes per the auto-tuner;
  our INT8 tuned shapes aren't the upstream-tested BF16 shapes)
- INT8 sign-magnitude conversion overhead on host (one-time, doesn't
  affect steady-state)
- some dispatch overhead at the tested shape (still climbing at
  5120×5632×5632)

So the **asymmetry between the two devices' utilization** is real:

| | Measured | Implied silicon ceiling | % of ceiling |
|---|---|---|---|
| TT Blackhole INT8 mcast HiFi2 | 228.5 TOPS | ~275 TOPS | **~83 %** |
| RTX 5090 cuBLASLt INT8 (`torch._int_mm`) | 214.5 TOPS | ~419 TOPS (dense) | **~51 %** |

Both numbers are honest measurements of the **production library path**
on each device. They are not honest measurements of "what the silicon
can do at absolute peak" — TT lands closer to its silicon peak than
NVIDIA's cuBLASLt does.

## Two ways to address this

### Option A — re-run NVIDIA at the next tier

Replace `torch._int_mm` with one of:

- **CUTLASS sm_120 INT8 kernel** (Blackwell Tensor Core path,
  per-shape autotune). Expected: 300–400 TOPS at our shapes (70–85 %
  of dense peak).
- **TensorRT-LLM INT8 GEMM** kernels. Similar territory; depends on
  whether the TRT-LLM API exposes a callable GEMM at our shapes.

Effort: 4–8 hours.

What it changes: the cross-device peak ratio becomes:

- TT 228 vs NVIDIA ~350 ⇒ **~1.5× NVIDIA-favoured silicon**, ~1.34×
  Blackhole-favoured per dollar (TT is half the price).

The qualitative conclusion is materially different: NVIDIA has a
modest silicon advantage even at INT8, but TT remains decisively ahead
per dollar.

### Option B — leave the measurement, document the caveat

Keep the `torch._int_mm` number as the apples-to-apples production
comparison. Add a one-paragraph caveat to §4.5.4 saying:

> The cuBLASLt path measured here delivers ~51 % of RTX 5090's dense
> INT8 Tensor Core ceiling. CUTLASS or TensorRT-LLM would push that to
> ~75 %, partially restoring NVIDIA's silicon lead. The TT mcast number
> sits at ~83 % of its matrix-engine ceiling. Both rows compare what
> production libraries deliver, not what either silicon can do at
> absolute peak.

Effort: 10 minutes.

What it changes: nothing factual. Frames the conclusion correctly.

## Recommendation

Capture this asymmetry in the main doc via Option B before the headline
INT8 numbers get cited externally. Run Option A as a follow-up
experiment if/when somebody wants the silicon-vs-silicon ceiling
comparison — but it's a separate experiment and shouldn't replace the
production-library row, since "what cuBLASLt delivers" is more
representative of what an ML engineer would measure on day 1.

The cleanest framing is to keep both rows in SUMMARY when Option A
lands, with the column header naming the kernel quality tier:

```
NVIDIA backend                          INT8 throughput  % dense peak
────────────────────────────────────────────────────────────────────
cublaslt_int8 (torch._int_mm)            215 TOPS         51 %
cutlass_sm120_int8                       ~350 TOPS        ~83 %  [Option A]
```

Same shape on TT for direct comparison.

## Other paths to investigate (lower priority)

1. **Older `torch.matmul` int8 path on PyTorch 2.4+**: there's a
   different cuBLASLt entry point that may pick a faster algo for
   Blackwell. Two-line change to `bench_nvidia.py`. Worth a 30-min
   experiment.
2. **2:4 structured sparsity on both sides**: NVIDIA marketing peak
   is *with* sparsity; we'd need to apply it on the TT side too via
   `cute::sparse` patterns or skip every other tile. Probably not
   worth the complexity — sparsity matters for ML inference, not for
   the FHE-modmul recipes this benchmark targets.
3. **Test at a TT shape that maps cleanly onto NVIDIA's saturation
   shape** (e.g. add 4096³ INT8 to the TT mcast sweep). Lets us
   compare at *exactly the same problem size* instead of nearby
   shapes. Effort: 5 min — already supported by the harness, just
   needs the shape added to `PER_CORE_SHAPES` in the wrapper.
