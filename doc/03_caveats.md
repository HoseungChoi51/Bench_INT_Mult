# 3 — Caveats and deferred work

**Takeaway.** v1 of this campaign covers Layers A, B, C-minimal, and
(on the TT side only) Layer D. The numbers in
[`02_findings.md`](02_findings.md) are real, gated for correctness,
and reproducible from the JSONL files in
[`bench-results/`](../bench-results/). They are also *partial* in
specific, named ways. This document lists every gap a reader should
be aware of before quoting the comparison.

---

## 3.1 Coverage gaps

### NVIDIA Layer D is missing

The biggest hole. Layer D is the headline FHE-throughput metric, and
the v1 plan deliberately deferred it because a fused KLSS-IP kernel is
significant work (CUTLASS epilogue, possibly hand-CUDA). The TT side
went ahead and measured Layer D anyway, leaving the comparison
asymmetric. Until NVIDIA Layer D lands, the Blackhole ~300 G_MAC/s
number sits without a peer.

A *provisional* upper bound for NVIDIA Layer D can be derived from
Layer B: at 4096³ INT8 the device hits 214 TOPS = 107 G_MAC/s of
unreduced int32 work. Modular reduction inside the inner loop will
slow this down by some factor; rough guess based on the Layer C drop
is 5–10×. So the NVIDIA Layer D number — when it lands — is likely
between 10 and 20 G_MAC/s. That is roughly 5–10× lower than Blackhole
*raw*. Per-dollar with the 2× price gap, NVIDIA could end up between
2.5–5× lower per dollar than Blackhole at 4096³ on Layer D. **This is
a guess, not a measurement.** Treat it as "the comparison may flip on
this layer; we don't know yet."

### Layer C q48 on NVIDIA is missing

The v1 plan had q48 in the v2 deferred list, and that's where it
stayed for NVIDIA. TT measured q48 in both Layer C and Layer D. The
q48-vs-q36 sensitivity ratio (~0.69 — i.e. q48 is ~31% slower) holds
on TT and is consistent with the cost model; we expect the same
factor to hold on NVIDIA, but expect-then-verify.

### Blackhole `tt_llk_sfpu_fp32` is unimplemented

The TODO in `tt-llk-skeleton/kernels/compute_sfpu_fp32.cpp` is still
open. The Python wrapper correctly emits skipped records with the
TODO error string. SFPU FP32 is per BENCHMARK.md §3 a *diagnostic*
backend, not a headline path; not having it doesn't change the main
comparison, but it leaves the FP32 row of the Layer B table partly
empty.

### Layer E end-to-end is not started on either side

A real CKKS slice (gadget decomp + base conversion + NTT + IP +
ModDown) is the integration test that says whether the per-layer
arithmetic is the bottleneck or whether memory/scheduling is. v1 is
deliberately scoped before that; Layer E is for a separate campaign.

---

## 3.2 Methodology limits

### Single-host, single-run

Each device's numbers come from one host (one Ryzen + one RTX 5090,
one Linux box + one Blackhole p150a). No machine-to-machine
variation, no PCIe-topology variation, no NUMA awareness. The TT
device is the real silicon, not a simulator; the NVIDIA device is the
real silicon, not a TFLOPS estimate. But "the result reproduces on
*another* RTX 5090 to within X%" is not something this v1 establishes.

### Wall-clock timing only

The bench uses CUDA events (NVIDIA) and the host-side `chrono`
timer + `tt_metal::Finish` (TT). Both report device-side wall time
modulo dispatch overhead; neither uses Nsight Compute or the TT-Metal
profiler in the script-driven path. Per BENCHMARK.md §4 those are
manual diagnostics layered on top.

For a single-kernel matmul this is fine — the kernel-level cycle count
would not materially change the per-dollar story. For multi-kernel
sequences (Layer D's 25-GEMM-plus-reduction loop, all of Layer E) it
matters more. v2 should add `ncu` / TT-Metal profiler integration.

### Energy is one-sided

TT records `power_w_avg` and `joules_per_useful_op`. NVIDIA records
neither in v1. The joules-per-modmul comparison would be valuable —
data-center FHE deployment cost is real-world dominated by power, not
list price — but it requires a polling sidecar (`nvidia-smi --loop` or
NVML) on the NVIDIA host and we did not implement it.

### Backend selection is "best heuristic", not "best autotuned"

`nvmath.linalg.advanced.matmul` exposes 8 candidate cuBLASLt algorithms
per (shape, dtype). The bench picks `algorithms[0]` — the heuristic
default — and times it. For most shapes that is also the autotuned
best, but not always. A v2 pass should iterate the candidate list,
per-shape, and emit the best of N. Same caveat applies on the TT side
to MATH_FIDELITY (HiFi2/HiFi4/LoFi) and tile-size selection.

### The 25-GEMM Layer C recipe is not the only option

The bench's Layer C uses byte-chunk decomposition into 25 INT8 partial
products plus per-pair `% q`. Alternatives that may run faster on this
hardware:

- **Barrett reduction inside a fused CUTLASS epilogue.** Eliminates
  the `% q` round-trip through HBM that drives the 4096³+ tail-off.
- **Montgomery-form arithmetic.** Trades the per-pair reduction for an
  initial transform and final inverse-transform; can fuse better on
  some hardware.
- **A direct 36-bit modmul on a putative FP64 tensor-core path.** Not
  applicable on consumer Blackwell (no FP64 tensor) or TT Blackhole
  (no FP64 matrix engine), but relevant if these numbers are quoted
  alongside H100 / GB200 results.

The v1 recipe is the cleanest one to share between two very different
devices; it is not the theoretical maximum on either.

### Prices are MSRP-at-bench-time, not bulk / used / used-on-eBay

`DEVICE_PRICES_USD` in `scripts/_bench_common.py` lists $1999 for RTX
5090 and $999 for TT Blackhole p150a. These are list prices declared,
not detected — they have to be updated by hand. The per-dollar ratios
move 1:1 with the price ratio, so a different price assumption (cloud
hourly rates, depreciated H100s, Blackhole bulk pricing) gives a
different comparison. Keep the price assumption visible when quoting
any per-dollar number.

### CPU baseline is Python, not C++ unsigned __int128

The "CPU int128" record in Layer C is a Python `int` reference loop —
arbitrary-precision but very slow (~0.01 G_modmul/s per record). It
establishes the "no GPU" floor, not a competitive number. A C++
`unsigned __int128` loop or an Intel HEXL / AVX-512-IFMA path would
be the realistic CPU competitor. v2 should include at least one of
these.

---

## 3.3 What v2 should add

Roughly in priority order:

1. **NVIDIA Layer D KLSS-IP, unfused.** Closes the Layer D gap on the
   most-quoted side. ~1–2 days of work.
2. **NVIDIA Layer C q48.** Mechanical extension of the existing path
   (6 byte chunks → 36 partials). ~0.5 day.
3. **A fused Layer C kernel (CUTLASS epilogue) on NVIDIA.** This is
   the single change most likely to lift the 4096³+ tail-off. ~2 days.
4. **NVIDIA energy telemetry.** `nvidia-smi --loop=100ms` polling
   sidecar, joined into the JSONL via timestamp. ~0.5 day.
5. **Layer D fused on both sides.** TT's strength is supposed to be
   SRAM-local sharded execution; the unfused number above is a lower
   bound. ~3+ days each side.
6. **Intel HEXL / AVX-512-IFMA CPU baseline.** Closes the CPU row of
   the table. ~1 day.
7. **Multi-card and multi-host scaling.** Separate v3 problem.

---

## 3.4 What v1 *does not* claim

To pre-empt over-reading:

- v1 does **not** claim that RTX 5090 is the right hardware for FHE.
  It claims that it is the faster device per dollar on the specific
  arithmetic primitive in our v1 layer set. The end-to-end CKKS
  picture (Layer E) is unmeasured.
- v1 does **not** claim Blackhole is uncompetitive. It shows a
  per-dollar inflection at 512³ that is real; it shows a Layer D
  number that has no NVIDIA peer yet; and the device is at ~50% of
  TDP, suggesting the kernels themselves have headroom.
- v1 does **not** quote raw TOPS as a primary metric. The headline
  numbers are effective modmul/s and effective MAC/s — what actually
  shows up at the FHE workload level. The raw GEMM TOPS in Layer B
  are diagnostics, not headlines.
- v1 does **not** rule out FP64 / TF32 / FP32 modmul recipes. The
  bench measures the INT8 byte-decomposition recipe specifically,
  because both devices have a fast INT8 matrix engine. Other recipes
  may win on hardware with strong FP64 tensor cores (e.g. H100/GB200);
  those are out of scope for this RTX-vs-Blackhole comparison.

The honest one-line summary of v1 is: **on the FHE-relevant primitives
both devices can run, RTX 5090 is faster per dollar in most regimes
but not all, and the Layer D KLSS-IP comparison — which is the
arithmetic shape FHE-acceleration papers care about most — is not yet
complete.**
