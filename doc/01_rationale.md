# 1 — Rationale

**Takeaway.** Fully Homomorphic Encryption (FHE) schemes like CKKS spend
most of their time multiplying large integers modulo prime numbers. The
modulus is conventionally 36 bits wide. That width is awkward on
commodity hardware: the *product* of two 36-bit numbers needs 72 bits,
which doesn't fit in a `uint64`. So the question is which device runs
this odd-sized arithmetic fastest per dollar — for which we need
benchmarks, not data sheets.

This document explains the workload, the hardware, and what the five
benchmark layers measure. The numbers themselves are in
[`02_findings.md`](02_findings.md).

---

## 1.1 What FHE actually computes

Fully Homomorphic Encryption (FHE) lets a server perform arithmetic on
ciphertexts without decrypting them. The CKKS scheme — Cheon–Kim–Kim–Song,
the most popular FHE scheme for approximate real-number computation —
stores ciphertexts as polynomials with very large integer coefficients.
"Very large" meaning thousands of bits. Operations on those coefficients
are done in **Residue Number System (RNS)** form: instead of one huge
integer modulo a giant `Q`, you store a tuple

```
( x mod q_0,  x mod q_1,  …,  x mod q_{L-1} )
```

with each `q_i` (called an **RNS limb** or **RNS prime**) small enough
to fit in a machine word. The big-integer operations decompose into
independent operations on each limb — embarrassingly parallel.

So the inner loop of a CKKS run looks like this, repeated billions of
times:

```
c_i = (a_i · b_i) mod q_i
```

with `a_i, b_i, q_i` all sub-machine-word integers. **This is the
unit-of-work the bench measures.**

CKKS additionally uses the Number Theoretic Transform (NTT) to
multiply polynomials efficiently. NTT is an FFT over modular
integers. It dominates the wall-clock cost of vanilla CKKS schemes;
recent variants like KLSS (Kim–Liu–Sajadieh–Sandström, 2024) trade
NTTs for inner-product-style operations, which shifts the bottleneck
toward what we call **Layer D** below.

---

## 1.2 Why 36 bits is awkward

The choice of `q_i` width is a three-way compromise:

| Constraint                          | Direction the constraint pushes |
| ----------------------------------- | ------------------------------- |
| CKKS precision / noise budget       | bigger primes, fewer of them    |
| NTT efficiency (root of unity)      | primes congruent to 1 mod 2^k   |
| Hardware native arithmetic width    | primes that fit `uint64`        |

The first two constraints jointly suggest 36-bit primes. The third
disagrees:

- A 36-bit prime has 36-bit operands, so `a · b` needs **72 bits**.
- A `uint64` overflows silently at 64 bits — `(2^36 - 1)^2 = 2^72 - 2^37 + 1`,
  which `numpy.uint64` truncates to its low 64 bits without raising.
  The `% q` reduction afterward then returns garbage.
- The natural workarounds — `__uint128` in C++, Barrett reduction,
  byte-chunk decomposition — all add overhead that scales differently
  on different hardware.

A 31-bit prime (e.g. `2^31 - 1`) avoids this entirely: its product
fits in 62 bits, comfortably inside one `uint64`. But more limbs are
needed to cover the same total modulus size, and the per-limb NTT
behavior is also slightly worse. There is genuine tension here, not a
free lunch.

**The bench therefore measures 36-bit modular multiplication
specifically, because that is the operating point CKKS implementations
actually use.** A separate 48-bit measurement is included for
sensitivity — KLSS-style schemes occasionally push the prime up.

---

## 1.3 Why this hardware comparison

The two devices benchmarked are:

| Device                        | MSRP   | Strong arithmetic path            |
| ----------------------------- | -----: | --------------------------------- |
| NVIDIA GeForce RTX 5090       | $1999  | INT8 / TF32 Tensor Cores (sm_120) |
| Tenstorrent Blackhole p150a   |  $999  | INT8 Tensix matrix engine + SFPU  |

The price ratio is approximately 2:1 in Blackhole's favor. The hardware
shapes are quite different:

- **RTX 5090** is a consumer Blackwell GPU. Its INT8 dense throughput is
  spec'd around 838 TOPS (sustained measured: ~210 TOPS at 4096³). FP32
  CUDA-core throughput is ~104 TFLOPS spec, ~62 TFLOPS measured. FP64 is
  *not* tensor-accelerated on consumer Blackwell — the FP64 path is a
  scalar/pipeline of FP32 cores, expected at ~1/64 of FP32 (1.6 TFLOPS;
  measured 1.58 TFLOPS — within noise).
- **TT Blackhole** has 110 Tensix cores. Each Tensix has a matrix engine
  (`8×16 × 16×16 → 8×16` per cycle, INT8 / BF16 / TF32 / "FP32-matrix"
  approximate) plus an SFPU vector unit. The architecture is most
  effective when data stays in local SRAM and across-tile communication
  is via NoC sharding rather than DRAM round-trips.

Neither device has an FP64 tensor-core path that would make 36-bit
modular arithmetic free. Both can do exact 36-bit modmul through INT8
chunk decomposition (5 byte chunks per operand → 25 partial products,
recombined with byte shifts and reduced mod q). That is the **shared
recipe** that lets us compare them on equal footing.

---

## 1.4 What the layers measure

The campaign is structured as five layers, A through E. v1 of the bench
covers A through D; E (an end-to-end CKKS slice) is deferred. Each
layer answers a more specific question than the one before.

### Layer A — Capability probe

One small (1024³) matmul per backend per device. Confirms each backend
dispatches at all and records the algorithm metadata
(`cuBLASLtMatmulAlgo` ID + tile, or the LLK kernel config on TT). No
correctness gate, no large workload — it's just a sanity check that
the rest of the campaign isn't measuring a misconfigured kernel.

A throughput plausibility flag is set if the achieved TFLOPS deviates
more than 5× from the published device spec. This is what would catch
a "FP64 is somehow tensor-core-fast on consumer Blackwell" misconception
before it polluted Layer B.

### Layer B — Raw GEMM throughput

Five square sizes (512², 1024², 2048², 4096², 8192²) × every backend the
device has. Pure matrix multiply, no modular reduction. The raw TOPS /
TFLOPS here are not directly meaningful for FHE — they are the
ceiling each backend could hit if modmul cost nothing.

What Layer B is good for is *diagnosis*: if Layer C is slow, Layer B
tells you whether the GEMM itself was slow or whether the modular
recomposition was the culprit.

### Layer C — Exact modular product

The headline FHE-relevance layer. For each shape, we compute

```
C[i, j] = a[i, j] · b[i, j] mod q
```

with `q` a 36-bit NTT-friendly prime. The recipe — same on both devices
— is to decompose each operand into 5 byte chunks (each in `[0, 127]`,
keeping the high bit of each `int8` clear), run 25 INT8 GEMMs, scale
each partial by `2^(8(i+j)) mod q` (precomputed), reduce mod q
**per pair** (otherwise the int64 accumulator overflows on the
high-shift terms), and sum.

A bit-exact correctness gate runs against a Python `int` reference on
an adversarial set:

- 4 boundary values (`0`, `1`, `q-1`, `q-2`) and their 16 pairwise products
- `2^k ± 1` for `k ∈ {30, 32, 34, 35}` squared and crossed against `q-1`
- 1000 random pairs in `[0, q)`
- 100 near-quotient-boundary pairs (products `≈ m·q ± δ` for small δ)

**Failed gate ⇒ no perf number ships.** The script emits a record with
null perf fields and `correctness.gate == "failed"`, which compare.py
renders as `—` — the headline tables can never silently lie about a
backend that doesn't compute the right answer.

The throughput unit is `G_modmul/s` (giga-modmul per second), defined
as `(M·N output elements) / median_seconds`. Each output element costs
one logical 36-bit modular multiplication regardless of what the
backend did internally, so the metric is comparable across recipes.

A separate q48 measurement uses 6 chunks → 36 partial products. It is
not meant to be the headline number; it tests how badly the per-limb
cost scales with prime width.

### Layer D — KLSS-like inner product

For matrices `A, K`, compute

```
C[i, j] = Σ_t (A[i, t] · K[t, j]) mod q
```

— the modular **inner product**, with reduction across the K dimension.
This is the operation that dominates KLSS-variant schemes, and it is
what hardware matrix engines should actually be good at: the GEMM
shape matches their natural dispatch unit.

The throughput unit is `G_MAC/s` (giga–multiply-accumulate per second).
Each output element `C[i, j]` represents `K` useful modular MACs
(after reduction), so for a square `n³` matmul `useful_ops = 2 n³`. A
single Layer D matmul therefore reports ~K × the throughput of a
single Layer C matmul of the same shape, even though both took the
same wall-clock — Layer C wastes the K dimension on element-wise work,
Layer D uses it.

This is where the FHE-relevance argument lives: Layer D is the
operation an FHE accelerator is judged on.

### Layer E — End-to-end KLSS slice

A reduced KLSS pipeline (gadget decomposition, base conversion, NTT,
inner product, ModDown). Not measured in v1. Once both sides have
Layer D fused implementations, Layer E is what tells you whether the
practical bottleneck is arithmetic, memory bandwidth, or scheduling.

---

## 1.5 Headline metrics

Per BENCHMARK.md §4, the campaign reports two headline numbers — *not*
raw TOPS:

1. **Effective exact modular multiplications per second.** Layer C
   throughput. The natural per-element cost.
2. **Effective KLSS-IP useful MACs per second.** Layer D throughput.
   The natural per-output-element cost when reduction is part of the
   work.

Raw TOPS would overstate INT8 approaches because 25 small INT8 MMAs
are emulating one useful 36-bit modmul; the headline metrics divide
that out.

Both metrics are also reported **per dollar** (per $1000 of device
cost), so the ~2× price gap can be normalized into the comparison
without hiding it.

---

## 1.6 Why the correctness gate is non-negotiable

The first version of `scripts/bench_gpu.py` (kept in the repo for the
notebook §4.5 cost-model demonstration) silently overflows int64 on the
high-shift partials in the 25-INT8 reconstruction, so its `% q` outputs
are incorrect. The throughput numbers it reports are real wall-clock
times, but for a computation that produces wrong answers. That bug went
undetected for a turn until the new bench's adversarial gate failed.

The takeaway, recorded here to keep us honest: any time a 36-bit modmul
recipe is measured, the gate must run *first*. A passing gate is the
preflight; the perf number only lives downstream of it.
