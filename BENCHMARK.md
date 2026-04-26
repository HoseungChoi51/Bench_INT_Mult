## 1. Does “36-bit WordSize” mean 36-bit or 72-bit arithmetic?

It means **36-bit residues / RNS limbs at the algorithmic interface**, but the operation must preserve the semantics of a **72-bit product reduced modulo a 36-bit modulus**.

For one modular multiplication:

```text
q < 2^36
a, b ∈ [0, q)

a · b < 2^72
result = (a · b) mod q
```

So if you implement it with ordinary integer arithmetic, you need something **equivalent to a 72-bit product**. You do **not** necessarily need to materialize a 72-bit integer register, but the algorithm must produce exactly the same residue as if you had done the 72-bit product and reduced it.

This distinction is why Neo can talk about **36-bit and 48-bit integer matrix multiplication** while using FP64/INT8 decomposition methods. They are not saying “the product is only 36 bits.” They are saying the **input word/residue precision** is 36 or 48 bits, and the chosen decomposition/emulation method is sufficient to compute the needed modular arithmetic. The Neo snippets explicitly discuss 36-bit and 48-bit integer matrix multiplication using FP64 and INT8, and note that splitting 36-bit integers into five INT8 matrices causes **25 cross matrix multiplications**. ([ACM 디지털 도서관][1])

For KLSS specifically, the inner-product stage is even more demanding than a single multiply, because it is a sum of many products. Taiyi’s KLSS accelerator discussion describes 36-bit data elements being multiplied with key elements and accumulated into a **128-bit polynomial accumulator register**, which is a good clue: the external word is 36 bits, but internal accumulation needs substantially more width. ([arXiv][2])

## 2. Why stick to 36-bit KLSS word size?

It is **not simply a security constant**. It is a three-way compromise among:

```text
CKKS precision / noise budget
RNS limb count and modulus-chain structure
hardware cost of modular multiplication
```

CKKS security depends mainly on the lattice dimension and the total ciphertext modulus size, not on one magic limb width. Taiyi’s CKKS parameter discussion says the security level is affected by the lattice dimension and modulus, and it adopts 36-bit word length because prior work regarded it as the minimum needed to meet FHE application precision needs. ([arXiv][2])

Changing the word size changes the number of RNS limbs:

```text
#limbs ≈ ceil(log2(Q) / WordSize)
```

So:

|               Word size | Effect                                                                                                          |
| ----------------------: | --------------------------------------------------------------------------------------------------------------- |
| Smaller, e.g. 30/31-bit | Easier arithmetic on commodity INT32 units, but more RNS limbs, more NTT/BConv/IP/key traffic.                  |
|                  36-bit | Common accelerator compromise: fewer limbs than 30/31-bit while still much smaller than full 60-bit HE primes.  |
|            48/52/60-bit | Fewer limbs, but product/reduction hardware becomes much harder; tensor-core decomposition gets more expensive. |

KLSS itself reduces NTT-related work but shifts the bottleneck toward **BConv and especially IP / inner-product**. Taiyi says KLSS reduces NTT overhead, but increases BConv/IP complexity, and its breakdown shows IP becoming the dominant component in KLSS-based workloads. ([arXiv][3])

So Neo likely stays near the 36-bit regime because it is a practical benchmark-compatible CKKS parameter point, not because cryptography requires exactly 36. Neo also appears to introduce a selective `WordSize_T` for the KLSS transfer base, so the internal KLSS base may be tuned separately from the original ciphertext limb size. ([ACM 디지털 도서관][4])

## 3. FP32 splitting overhead vs 25 INT8 matrix multiplications

The useful rule is:

```text
For exact limb-product accumulation:
2 · limb_bits + ceil(log2(K_accum)) <= significand_or_accumulator_bits
```

For a **single product**, FP32 has 24 significant bits, so you can multiply at most about **12-bit × 12-bit** exactly. For a matrix engine that accumulates over a K dimension, you need extra headroom for the sum, so the safe FP32 limb size is often **less than 12 bits**.

For example, with a small accumulation segment of `K_accum = 16`:

```text
FP32: 2s + 4 <= 24  →  s <= 10 bits
FP64: 2s + 4 <= 53  →  s <= 24 bits
INT8→INT32: 2·8 + 4 = 20, safely below 31 bits
```

That gives this rough decomposition count:

| Method               | Safe limb size, assuming K segment ≈ 16 |           36-bit word |           48-bit word |
| -------------------- | --------------------------------------: | --------------------: | --------------------: |
| FP64 exact limb path |                                ~24 bits |  2 limbs → 4 products |  2 limbs → 4 products |
| FP32 exact limb path |                                ~10 bits | 4 limbs → 16 products | 5 limbs → 25 products |
| INT8 tensor path     |                                  8 bits | 5 limbs → 25 products | 6 limbs → 36 products |
| TF32 tensor path     |                   too few mantissa bits |   usually impractical |   usually impractical |

This explains Neo’s result: FP64 can use a **2×2 decomposition**, while INT8 needs **5×5** for 36-bit and **6×6** for 48-bit. Neo’s accessible snippet says FP64 uses complexity **2×2 = 4** and reports **1.74×** the INT8 speed for the 48-bit case. ([ACM 디지털 도서관][4])

For your RTX 5090, however, be careful: GeForce RTX Blackwell Tensor Cores support FP16, BF16, TF32, INT8, FP8, FP4, and FP6, while FP64 Tensor Cores are described as minimal and included for program correctness. That is not the same environment as an H100-style FP64 Tensor Core path. ([NVIDIA Images][5]) H100, by contrast, publicly lists high-throughput FP64 Tensor Core performance. ([NVIDIA][6])

For TT Blackhole, the matrix engine path is also not an FP64 path. Tenstorrent’s Hot Chips material lists Blackhole matrix support for BlockFP, FP8, BF16, TF32, and INT8→INT32; vector support includes FP32, INT16, and INT32. ([Hot Chips 2024][7]) Tenstorrent’s own FP32 accuracy page also warns that the matrix engine is throughput-oriented and mainly uses bfloat16/TF32-style formats, while the vector/SFPU path is the one to use when higher FP32 accuracy matters. ([Tenstorrent Documentation][8])

So, for TT:

```text
INT8 Tensix matrix path: worth investigating.
FP32 Tensix matrix path: probably not valid for exact 36-bit modular arithmetic.
FP32 SFPU path: valid-ish for FP32 experiments, but likely much lower throughput.
FP64 path: essentially unavailable on Tensix.
```

The conclusion is not “INT8 has no point.” It is:

> INT8 is worse than a good FP64 tensor path if such a path exists, but on RTX 5090 and TT Blackhole, INT8 may be the only realistic way to exploit otherwise idle matrix engines for exact-ish multi-limb FHE arithmetic.

## 4. Benchmark plan for RTX 5090 + TT Blackhole

I would structure the benchmark in four layers.

### Layer A — Capability probe

First verify what the hardware actually executes.

| Device       | Probe                                                                                                                                                                                                                                                                               |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RTX 5090     | Use CUTLASS/cuBLASLt and Nsight Compute to distinguish INT8 MMA, TF32 MMA, FP32 CUDA-core FMA, FP64 scalar/minimal tensor behavior. Do not assume FP64 Tensor Core throughput.                                                                                                      |
| TT Blackhole | Use TT-Metal profiler and tiny matmul kernels to confirm INT8 matrix-engine availability, FP32/SFPU behavior, and packing overhead. Tenstorrent’s GEMM report gives official benchmark commands and notes the matrix engine performs `8×16 × 16×16 → 8×16` per cycle. ([GitHub][9]) |
| CPU          | Use `unsigned __int128` or Boost.Multiprecision as correctness reference; optionally compare Intel HEXL/AVX512-IFMA if available. Intel HEXL is specifically designed for HE modular arithmetic and reports large NTT/modmul speedups with AVX512. ([arXiv][10])                    |

### Layer B — Raw matrix-engine throughput

Measure pure GEMM/MMA first, without modular reduction:

```text
C = A × B
```

Run at matrix sizes like:

```text
512², 1024², 2048², 4096², 8192²
```

Use dimensions aligned to:

```text
NVIDIA MMA tile sizes
TT 32×32 tile / 16-wide K structure
```

Backends:

```text
RTX5090 INT8 Tensor Core
RTX5090 TF32 Tensor Core
RTX5090 FP32 CUDA core
RTX5090 FP64 path, measured but not assumed tensor-accelerated

TT INT8 matrix engine
TT BF16/TF32/FP32-like matrix path, marked approximate
TT SFPU FP32 vector path

CPU int64/int128
```

Report:

```text
raw TOPS/TFLOPS
actual kernel time
device utilization
power / joules
host overhead separately
```

### Layer C — Exact 36-bit and 48-bit modular product

Now benchmark the real primitive:

```text
c = (a · b) mod q
```

Use NTT-friendly primes:

```text
q36 < 2^36
q48 < 2^48
```

Correctness tests must include adversarial values:

```text
0, 1, q-1, q-2
2^k ± 1
random full-range residues
products near quotient-boundary cases: a·b ≈ m·q ± 1
```

Each backend must pass bit-exact comparison against CPU multiprecision before performance numbers matter.

For decomposition:

```text
INT8:
    36-bit → 5 limbs → 25 partial products
    48-bit → 6 limbs → 36 partial products

FP32 exact scalar:
    choose limb bits from 2s + log2(Kseg) <= 24
    likely 10–12-bit limbs depending on accumulation segmentation

FP64:
    36/48-bit can use 2 limbs if accumulation is segmented carefully

TT FP32 matrix:
    run only as approximate/diagnostic unless it passes strict tests
```

### Layer D — KLSS-like inner product

Benchmark the Neo-relevant operation, not just isolated multiplication:

```text
C[i, j] = Σ_t A[i, t] · K[t, j] mod q
```

This is where tensor/matrix engines matter.

You need two versions:

```text
unfused:
    partial GEMMs → materialize partials → recombine → reduce

fused:
    partial GEMMs + recombination/reduction tiled in SRAM/shared memory
```

For TT, the fused version is especially important. Blackhole p150a has 120 Tensix cores, 32 GB GDDR6, and 512 GB/s DRAM bandwidth, but the architecture is most interesting when data stays in local SRAM and sharded layouts. ([Tenstorrent][11])

### Layer E — End-to-end KLSS slice

Finally benchmark a reduced KLSS pipeline:

```text
Gadget decomposition
BConv / base conversion
NTT
IP
Recover limbs
ModDown
```

This is necessary because KLSS may reduce NTT count but increase IP/BConv/memory pressure. Taiyi’s analysis reports that KLSS changes the bottleneck structure and that IP can dominate execution time. ([arXiv][3])

## Benchmark result table I would target

| Backend                         |  Word size | Expected correctness                                           | Expected value                                      |
| ------------------------------- | ---------: | -------------------------------------------------------------- | --------------------------------------------------- |
| CPU `__int128` / multiprecision |      36/48 | exact                                                          | correctness baseline                                |
| CPU AVX512-IFMA / HEXL          | ≤50–52-ish | exact under constraints                                        | strong CPU HE baseline                              |
| CUDA INT32/INT64 custom         |      36/48 | exact if implemented carefully                                 | conventional GPU baseline                           |
| CUDA INT8 Tensor Core           |      36/48 | exact if carry/reduction is correct                            | main RTX 5090 tensor path                           |
| CUDA FP64 Tensor Core           |      36/48 | only meaningful on H100/B200-class GPUs                        | Neo-like comparison, probably not RTX 5090          |
| CUDA FP32                       |      36/48 | exact only with small limbs and careful segmented accumulation | likely not tensor-accelerated                       |
| TT INT8 Tensix                  |      36/48 | possible, must validate                                        | most interesting TT path                            |
| TT FP32 matrix                  |      36/48 | likely not exact                                               | diagnostic only                                     |
| TT SFPU INT32/FP32              |      36/48 | possible for substeps                                          | likely lower throughput, useful for carry/reduction |

The benchmark should report two headline metrics:

```text
1. Effective exact modular multiplications per second
2. Effective KLSS-IP useful MACs per second
```

not raw TOPS. Raw TOPS will overstate INT8 approaches because 25 or 36 small MMAs are being used to emulate one useful wide-word operation.

My practical expectation:

```text
RTX5090:
    INT8 Tensor Core path is likely the useful tensor path.
    FP64 Tensor Core comparison will probably be weak or unavailable.

TT Blackhole:
    INT8 Tensix path is the one worth serious effort.
    FP32 matrix path is unlikely to be valid for exact FHE arithmetic.
    SFPU/RISC-V code should handle carry, correction, and modular reduction around matrix-engine partial products.

CPU:
    use as correctness and latency baseline, especially with int128/HEXL.
```

So the forward-looking experiment is not “can TT emulate FP64?” It is:

> Can TT’s INT8 matrix engine plus local SRAM plus explicit carry/reduction scheduling beat conventional INT32/INT64 modular arithmetic for the KLSS inner-product shape?

[1]: https://dl.acm.org/doi/10.1145/3695053.3731408?utm_source=chatgpt.com "Neo: Towards Efficient Fully Homomorphic Encryption ..."
[2]: https://arxiv.org/html/2403.10188v1?utm_source=chatgpt.com "Taiyi: A high-performance CKKS accelerator for Practical Fully Homomorphic Encryption"
[3]: https://arxiv.org/html/2403.10188v1 "Taiyi: A high-performance CKKS accelerator for Practical Fully Homomorphic Encryption"
[4]: https://dl.acm.org/doi/pdf/10.1145/3695053.3731408?utm_source=chatgpt.com "Neo: Towards Efficient Fully Homomorphic Encryption ..."
[5]: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf?utm_source=chatgpt.com "https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf"
[6]: https://www.nvidia.com/en-us/data-center/h100/?utm_source=chatgpt.com "H100 GPU"
[7]: https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf?utm_source=chatgpt.com "https://hc2024.hotchips.org/assets/program/conference/day1/88_HC2024.Tenstorrent.Jasmina.Davor.v7.pdf"
[8]: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/fp32_accuracy.html "Achieving FP32 Accuracy for Computation — TT-Metalium  documentation"
[9]: https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md "tt-metal/tech_reports/GEMM_FLOPS/GEMM_FLOPS.md at main · tenstorrent/tt-metal · GitHub"
[10]: https://arxiv.org/abs/2103.16400?utm_source=chatgpt.com "Intel HEXL: Accelerating Homomorphic Encryption with Intel AVX512-IFMA52"
[11]: https://tenstorrent.com/hardware/blackhole?utm_source=chatgpt.com "Blackhole™"
