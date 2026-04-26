Below is a detailed plan for an **interactive educational Python notebook series**. The target reader is a SW developer who understands integer arithmetic and Python, but not necessarily FHE, hardware multiplier design, or low-level optimization.

The central narrative should be:

> For 36-bit RNS arithmetic, Booth-style recoding is intellectually useful, but on commodity CPU/GPU hardware it is usually less practical than choosing RNS prime sizes that match native arithmetic, such as 30/31-bit primes.

---

# Proposed notebook series

I would make this as **4 notebooks + 1 optional benchmark notebook**.

```text
Notebook 1 — Binary multiplication and Booth recoding
Notebook 2 — Why 36-bit RNS modular multiplication is awkward
Notebook 3 — Hardware-aligned RNS primes: 30/31-bit versus 36-bit
Notebook 4 — INT8 / Tensor Core decomposition thought experiment
Notebook 5 — Optional CPU/GPU microbenchmarking
```

Each notebook should have three layers:

```text
1. Concept explanation
2. Interactive toy implementation
3. Cost model / practical implication
```

Do not start with FHE. Start with ordinary integer multiplication, then gradually introduce RNS and modular multiplication.

---

# Notebook 1 — Binary multiplication and Booth recoding

## Goal

Explain what Booth’s algorithm actually does:

> Booth recoding reduces the number of signed partial products in a multiplier. It is not, by itself, a magic method for mapping 36-bit arithmetic onto INT32 or INT8 hardware.

## Target concepts

The reader should understand:

```text
A × B can be viewed as a sum of shifted copies of A.

Naive binary multiplication:
    one partial product per 1-bit in B.

Booth recoding:
    convert runs/transitions in B into signed digits:
        -2, -1, 0, +1, +2

Radix-4 Booth:
    roughly halves the number of partial-product rows.
```

## Sections

### 1.1 Ordinary shift-add multiplication

Use a small example:

```text
A = 13
B = 60 = 0b00111100
```

Show:

```text
13 × 60
= 13×2^2 + 13×2^3 + 13×2^4 + 13×2^5
```

Then show the compressed identity:

```text
0b00111100 = 2^6 - 2^2
```

So:

```text
13 × 60 = (13 << 6) - (13 << 2)
```

The important clarification:

```text
This is not literally “one CPU add and one CPU subtract” in a real high-level program.

It means the multiplier has two signed partial-product rows instead of four positive rows.
```

### 1.2 Radix-2 Booth

Introduce adjacent-bit recoding:

```text
Look at pairs: (b_i, b_{i-1})
with b_{-1} = 0
```

Table:

| Pair | Meaning             | Action     |
| ---- | ------------------- | ---------- |
| `00` | outside run of ones | 0          |
| `01` | start of run        | +A shifted |
| `10` | end of run          | -A shifted |
| `11` | inside run          | 0          |

Depending on convention, the sign direction may flip. The notebook should explicitly define one convention and stick to it.

### 1.3 Radix-4 Booth

Introduce overlapping 3-bit groups:

[
(b_{2i+1}, b_{2i}, b_{2i-1})
]

Digit set:

[
d_i \in {-2,-1,0,+1,+2}
]

Table:

| Bits  | Booth digit |
| ----- | ----------: |
| `000` |           0 |
| `001` |          +1 |
| `010` |          +1 |
| `011` |          +2 |
| `100` |          -2 |
| `101` |          -1 |
| `110` |          -1 |
| `111` |           0 |

Then:

[
A B = \sum_i d_i (A \ll 2i)
]

### 1.4 Interactive widget

Use `ipywidgets` sliders:

```python
A_slider      # 0 to 255
B_slider      # 0 to 255
bit_width     # 4, 8, 12, 16
radix         # naive / Booth radix-2 / Booth radix-4
signed_mode   # unsigned / two's complement signed
```

Display:

```text
Binary form of A and B
Naive partial products
Booth digits
Signed partial products
Reconstructed product
Correctness check
Number of nonzero partial rows
```

### 1.5 Key educational plot

Plot expected number of partial rows for random multipliers:

```text
bit width: 4, 8, 16, 32, 36, 64

naive binary:
    average nonzero rows ≈ bit_width / 2

radix-4 Booth:
    groups = ceil(bit_width / 2)
    expected nonzero groups ≈ 0.75 × groups
```

For 36 bits:

```text
naive average ≈ 18 nonzero rows
radix-4 Booth average ≈ 13.5 nonzero rows
```

This is a real improvement, but not a dramatic enough improvement to overcome all software-level carry/recomposition overhead.

## Takeaway cell

```text
Booth is a partial-product reduction technique.

It is useful inside multiplier hardware.

As a software method on commodity INT32/INT8 units, it usually becomes a sequence of shifted multiword add/sub operations, which is not attractive for dense random FHE residues.
```

---

# Notebook 2 — Why 36-bit RNS modular multiplication is awkward

## Goal

Explain the arithmetic problem behind 36-bit FHE/RNS residues.

A single RNS limb has:

[
a,b < q \approx 2^{36}
]

A raw product has:

[
a b < 2^{72}
]

So a 36-bit modular multiplication needs a **72-bit intermediate product** before reduction.

## Sections

### 2.1 What is an RNS limb?

Keep the explanation minimal.

```text
In RNS arithmetic, a large integer is represented by residues modulo several smaller primes.

Instead of one huge integer X, we store:

    X mod q0
    X mod q1
    X mod q2
    ...

Each modulus qi is called an RNS limb or RNS prime.
```

For one modular multiply:

```text
c_i = (a_i × b_i) mod q_i
```

Each limb can be processed independently.

No need to explain CKKS deeply. Just say:

```text
CKKS/FHE uses many such modular multiplications inside NTT, key switching, rescaling, and bootstrapping.
```

### 2.2 Product-width visualization

Interactive slider:

```python
q_bits = IntSlider(20, 60)
```

Display:

```text
Input width: q_bits
Raw product width: 2 × q_bits
Fits in uint32?  product_bits <= 32
Fits in uint64?  product_bits <= 64
Comfortably fits in signed int64? product_bits <= 63
```

Examples:

| RNS prime size | Product width | Fits in uint64?       |
| -------------: | ------------: | --------------------- |
|        30 bits |       60 bits | yes                   |
|        31 bits |       62 bits | yes                   |
|        32 bits |       64 bits | barely, unsigned only |
|        36 bits |       72 bits | no                    |
|        60 bits |      120 bits | no                    |

### 2.3 Reference implementation using Python big integers

Implement:

```python
def mul_mod_reference(a: int, b: int, q: int) -> int:
    return (a * b) % q
```

Explain:

```text
Python integers are arbitrary precision, so this is correct but not representative of commodity machine-word cost.
```

### 2.4 Demonstrate overflow

Use NumPy `uint64`:

```python
import numpy as np

a = np.uint64((1 << 36) - 1)
b = np.uint64((1 << 36) - 1)
p = a * b
```

Show that NumPy wraps modulo (2^{64}), losing the high 8 bits of the 72-bit product.

Expected explanation:

```text
The arithmetic did not fail loudly. It silently computed the low 64 bits.
For modular arithmetic, this is usually fatal unless the algorithm was designed around that truncation.
```

### 2.5 36-bit multiplication using commodity words

Since the scope is commodity hardware, avoid custom 18-bit multipliers.

Use a practical decomposition into 32-bit chunks:

[
a = a_0 + 2^{32}a_1
]

where:

```text
a0: lower 32 bits
a1: upper 4 bits
```

Similarly:

[
b = b_0 + 2^{32}b_1
]

Then:

[
ab =
a_0b_0

* 2^{32}(a_0b_1 + a_1b_0)
* 2^{64}a_1b_1
  ]

This needs four conceptual products:

```text
a0*b0   # 32×32 -> up to 64 bits
a0*b1   # 32×4  -> up to 36 bits
a1*b0   # 4×32  -> up to 36 bits
a1*b1   # 4×4   -> up to 8 bits
```

Represent the result as three 32-bit words:

```text
product = w0 + 2^32 w1 + 2^64 w2
```

This is the correct commodity-software view of a 36×36→72 product.

### 2.6 Compare against Booth

Add a function:

```python
def booth_radix4_cost(bit_width: int) -> dict:
    groups = math.ceil(bit_width / 2)
    expected_nonzero = 0.75 * groups
    return {
        "groups": groups,
        "expected_nonzero_partial_rows": expected_nonzero,
    }
```

Then show for 36 bits:

```text
Radix-4 Booth:
    18 groups
    about 13.5 nonzero signed partial rows for random inputs

32-bit chunk decomposition:
    4 conceptual products
    plus carry/recomposition
```

The point:

```text
On commodity CPUs, native multiplication is much more valuable than reducing shift-add rows.

A few native word multiplications are usually better than many shifted multiword additions.
```

## Takeaway cell

```text
36-bit modular multiplication is awkward because the product is 72 bits.

Booth reduces partial-product rows inside a multiplier, but a software implementation on INT32 words still needs a multiword accumulator and many carry-propagating add/sub operations.

For commodity hardware, word-level limb decomposition is usually the more relevant model.
```

---

# Notebook 3 — Hardware-aligned RNS primes: 30/31-bit versus 36-bit

## Goal

Compare two strategies:

```text
Strategy A:
    Keep 36-bit RNS primes.
    Emulate 72-bit products using multiword arithmetic or compiler-provided 128-bit intermediates.

Strategy B:
    Use 30/31-bit RNS primes.
    Each product fits comfortably in 64-bit arithmetic.
    Use more RNS limbs to reach the same total modulus size.
```

This is the core notebook.

---

## 3.1 Total modulus size model

In RNS/FHE, you often care about the total number of modulus bits:

[
Q_{\text{bits}} = \sum_i \log_2 q_i
]

If every RNS prime has approximately (w) bits, then the number of limbs is approximately:

[
N_{\text{limbs}} = \left\lceil \frac{Q_{\text{bits}}}{w} \right\rceil
]

Interactive widget:

```python
total_modulus_bits = IntSlider(120, 2000, value=720)
prime_bits = IntSlider(20, 60, value=36)
```

Display:

```text
Number of RNS limbs
Raw product width
Whether product fits in uint64
Approximate memory per polynomial coefficient
```

Example:

| Total modulus bits | Prime size | Number of limbs |
| -----------------: | ---------: | --------------: |
|                360 |     36-bit |              10 |
|                360 |     31-bit |              12 |
|                720 |     36-bit |              20 |
|                720 |     31-bit |              24 |

So 31-bit primes may need roughly:

[
\frac{36}{31} \approx 1.16
]

times more limbs than 36-bit primes.

That is only about **16% more limbs**, but each modular multiply becomes much easier.

---

## 3.2 Product-width comparison

For each prime size (w):

[
\text{product bits} = 2w
]

Plot:

```text
x-axis: RNS prime bit width
y-axis: product bit width
horizontal line: 64-bit boundary
```

Important thresholds:

```text
w = 31:
    product <= 62 bits
    safe in signed/unsigned 64-bit arithmetic

w = 32:
    product <= 64 bits
    fits unsigned 64-bit only in the ideal case, but reductions may become awkward

w = 36:
    product <= 72 bits
    cannot be represented in one uint64
```

---

## 3.3 Modular multiplication methods

Define several conceptual implementations.

### Method 1: Native-safe 31-bit modular multiply

For:

[
q < 2^{31}
]

[
a,b < q
]

Then:

[
ab < 2^{62}
]

So:

```python
def mul_mod_31_reference(a, b, q):
    return (a * b) % q
```

In Python this is arbitrary precision, but explain that in C/C++ this can be implemented with one 64-bit product.

### Method 2: 36-bit modular multiply using 128-bit intermediate

In C/C++ on many x86-64/AArch64 compilers, one might write:

```cpp
uint64_t mul_mod_36_u128(uint64_t a, uint64_t b, uint64_t q) {
    __uint128_t p = (__uint128_t)a * b;
    return (uint64_t)(p % q);
}
```

But the notebook should emphasize:

```text
This is convenient in C/C++, but not always available in vectorized form, GPU code, or high-level array libraries.

Also, the modulo reduction itself may compile to expensive operations unless Barrett or Montgomery reduction is used.
```

### Method 3: 36-bit modular multiply using 32-bit chunks

Implement educationally:

```python
BASE = 1 << 32

def product_36_as_32_words(a, b):
    a0 = a & (BASE - 1)
    a1 = a >> 32
    b0 = b & (BASE - 1)
    b1 = b >> 32

    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1

    p = p00 + ((p01 + p10) << 32) + (p11 << 64)
    return p
```

Then verify:

```python
assert product_36_as_32_words(a, b) == a * b
```

Later, add a version that returns three 32-bit words instead of a Python integer.

### Method 4: Radix-4 Booth shift-add model

Implement:

```python
def booth_radix4_digits(b, bit_width):
    ...
```

Then:

```python
def booth_mul_reference(a, b, bit_width):
    product = 0
    for i, d in enumerate(booth_digits):
        product += d * (a << (2*i))
    return product
```

Verify correctness.

Then count operations:

```text
number of Booth groups
number of nonzero signed partial products
number of multiword add/sub operations
accumulator width
```

This is not intended to be fast. It is a cost model.

---

## 3.4 Main comparison table

For a single modular multiplication:

| Method                      | Keeps 36-bit primes? |          Product width problem |                          Approximate cost model | Commodity practicality                           |
| --------------------------- | -------------------: | -----------------------------: | ----------------------------------------------: | ------------------------------------------------ |
| 31-bit RNS primes           |                   no |        product fits in 64 bits |       more RNS limbs, simpler per-limb multiply | high                                             |
| 36-bit with `__uint128_t`   |                  yes |            handled by compiler |            one 64×64→128 product plus reduction | good in scalar C/C++, less portable              |
| 36-bit with 32-bit chunks   |                  yes |     explicit multiword product |      four chunk products plus carries/reduction | educational, sometimes useful                    |
| 36-bit with Booth shift-add |                  yes | explicit multiword accumulator | about 13.5 signed rows for random 36-bit values | usually poor as software                         |
| INT8 chunking               |                  yes |         explicit recomposition |   25 byte products plus recomposition/reduction | only possibly useful with huge tensor throughput |

---

## 3.5 Break-even model

Define adjustable cost parameters:

```python
cost_native_31_mulmod = 1.0
cost_36_u128_mulmod = slider, default maybe 2.0 to 8.0
cost_36_chunk_mulmod = slider, default maybe 4.0 to 12.0
cost_36_booth_mulmod = slider, default maybe 10.0 to 30.0
```

Then compare total cost for a target modulus size:

```python
total_cost = number_of_limbs * cost_per_limb_mulmod
```

Interactive widget:

```python
total_modulus_bits
prime_bits_31
prime_bits_36
cost_36_over_31
```

Show break-even condition:

[
\left\lceil \frac{Q_{\text{bits}}}{31} \right\rceil C_{31}
<
\left\lceil \frac{Q_{\text{bits}}}{36} \right\rceil C_{36}
]

The 31-bit scheme wins when:

[
\frac{C_{36}}{C_{31}}

>

\frac{\lceil Q_{\text{bits}}/31 \rceil}
{\lceil Q_{\text{bits}}/36 \rceil}
]

For large (Q_{\text{bits}}), this is roughly:

[
\frac{C_{36}}{C_{31}} > \frac{36}{31} \approx 1.16
]

That is the most important educational result.

Meaning:

```text
If 36-bit modular multiplication is more than about 16% slower per limb than 31-bit modular multiplication, then 31-bit primes can already be competitive or better, before considering vectorization and implementation simplicity.
```

Of course, this ignores NTT details, memory hierarchy, prime availability, rescaling behavior, and cryptographic parameter choices. But as a first-order model, it is powerful.

---

## 3.6 Key takeaway

```text
Using 31-bit RNS primes increases the number of limbs.

But preserving 36-bit primes may force non-native product widths.

The extra limb count may be smaller than the cost of emulating 72-bit products.
```

This is the main conceptual conclusion.

---

# Notebook 4 — INT8 / Tensor Core decomposition thought experiment

## Goal

Explain why INT8 Tensor Cores are tempting but difficult for exact FHE modular arithmetic.

The key distinction:

```text
Tensor Cores are excellent at dense low-precision matrix multiply-accumulate.

FHE modular arithmetic requires exact integer products, recomposition, carries, and modular reduction.
```

---

## 4.1 Byte decomposition of a 36-bit integer

A 36-bit integer needs five 8-bit chunks:

[
a = a_0 + 2^8 a_1 + 2^{16}a_2 + 2^{24}a_3 + 2^{32}a_4
]

[
b = b_0 + 2^8 b_1 + 2^{16}b_2 + 2^{24}b_3 + 2^{32}b_4
]

Then:

[
ab = \sum_{i=0}^{4}\sum_{j=0}^{4} a_i b_j 2^{8(i+j)}
]

So:

```text
5 chunks × 5 chunks = 25 byte-level products
```

This is the central result.

---

## 4.2 Interactive product-count model

Widget:

```python
operand_bits = IntSlider(8, 128, value=36)
chunk_bits = Dropdown([4, 8, 16, 32])
```

Compute:

```python
num_chunks = ceil(operand_bits / chunk_bits)
num_cross_products = num_chunks ** 2
```

Example table:

| Operand width | Chunk width | Chunks | Cross-products |
| ------------: | ----------: | -----: | -------------: |
|            36 |          32 |      2 |              4 |
|            36 |          16 |      3 |              9 |
|            36 |           8 |      5 |             25 |
|            64 |           8 |      8 |             64 |
|           128 |           8 |     16 |            256 |

This should make the quadratic penalty obvious.

---

## 4.3 Tensor Core mapping idea

For arrays or matrices of residues, one could decompose each matrix into byte matrices:

```text
A = A0 + 2^8 A1 + 2^16 A2 + ...
B = B0 + 2^8 B1 + 2^16 B2 + ...
```

Then matrix multiplication becomes:

```text
C = A × B
  = Σ_i Σ_j (Ai × Bj) << 8(i+j)
```

For 36-bit values:

```text
25 INT8 GEMMs
plus weighted recomposition
plus carry management
plus modular reduction
```

This can be visualized as a 5×5 grid of chunk-products:

```text
        b0   b1   b2   b3   b4
      --------------------------
a0 |   0    1    2    3    4
a1 |   1    2    3    4    5
a2 |   2    3    4    5    6
a3 |   3    4    5    6    7
a4 |   4    5    6    7    8
```

Each number is the byte-shift index (i+j).

Products on the same diagonal contribute to the same byte position.

---

## 4.4 Why Booth is not the answer for Tensor Cores

Explain directly:

```text
Booth recoding produces signed shifted copies of one operand.

Tensor Cores consume dense low-precision matrix tiles.

Those are different computational models.
```

For random FHE residues, Booth has no strong sparsity advantage:

```text
Radix-4 Booth on 36-bit random values:
    18 groups
    roughly 13.5 nonzero groups

INT8 chunking:
    25 dense chunk-products
```

Booth does not reduce the 5×5 byte-product grid into a small number of Tensor Core calls in a clean way.

---

## 4.5 Break-even model for INT8

Define:

```python
R = raw_INT8_throughput_advantage_over_native_integer
K = chunk_product_count
O = overhead_factor_for_recomposition_and_reduction
```

INT8 decomposition can win only if:

[
R > K \times O
]

For 36-bit values with INT8 chunks:

[
K = 25
]

So even before recomposition overhead:

```text
INT8 needs at least around 25× raw throughput advantage to break even.
```

With recomposition, carry handling, data layout, and modular reduction:

```text
required advantage > 25×
```

This is the right educational framing.

Do not claim that Tensor Cores can never help. Say:

```text
Tensor Cores may help when the computation can be expressed as very large dense matrix operations and the overheads are amortized.

But for ordinary coefficient-wise RNS modular multiplication, the mapping is not naturally Tensor-Core-shaped.
```

---

# Notebook 5 — Optional microbenchmark notebook

This should be clearly labeled:

```text
Optional and approximate.
Do not use Python-loop timing to infer hardware arithmetic throughput.
```

## Goal

Demonstrate qualitative trends, not publishable performance.

## Suggested benchmark layers

### Layer 1: Pure Python correctness

Use Python integers.

Methods:

```text
reference arbitrary-precision multiplication
Booth shift-add
32-bit chunk decomposition
8-bit chunk decomposition
31-bit RNS multiplication
```

Purpose:

```text
correctness only
```

### Layer 2: NumPy vectorized demonstration

Use arrays of residues.

Test:

```python
N = 1_000_000
a31 = np.random.randint(0, q31, size=N, dtype=np.uint64)
b31 = np.random.randint(0, q31, size=N, dtype=np.uint64)
```

Then:

```python
c31 = (a31 * b31) % q31
```

For 31-bit primes, this is valid because product fits in uint64.

For 36-bit primes, show that the same method is invalid because of overflow.

This is a very useful educational demonstration.

### Layer 3: Optional C++ extension

Use C++ only if you want more realistic CPU timing.

Possible methods:

```text
u31_mulmod_u64
u36_mulmod_u128
u36_mulmod_32chunk
u36_booth_shiftadd
```

Compile from notebook using:

```python
%%writefile
```

and `subprocess`.

Avoid over-engineering. The goal is to show relative shapes.

### Layer 4: Optional GPU / Tensor Core proxy

Use PyTorch or CuPy only as a proxy.

Possible demonstration:

```text
INT8 GEMM is fast.
But exact 36-bit multiplication requires 25 GEMMs plus recomposition.
```

The benchmark should separately time:

```text
1. chunk extraction
2. 25 low-precision products/GEMMs
3. recomposition
4. modular reduction
```

Otherwise the conclusion will be misleading.

---

# Suggested code architecture

Use a small educational package inside the notebook folder:

```text
rns_arithmetic/
    bits.py
    booth.py
    limb_decompose.py
    rns_cost_model.py
    tensor_int8_model.py
    visualization.py
```

Or keep them as notebook cells at first, then refactor later.

## Core functions

### Bit utilities

```python
def bit_width(x: int) -> int:
    return x.bit_length()

def bits_lsb_first(x: int, width: int) -> list[int]:
    return [(x >> i) & 1 for i in range(width)]

def format_binary(x: int, width: int) -> str:
    return format(x, f"0{width}b")
```

### Booth recoding

```python
def booth_radix4_digits_unsigned(b: int, width: int) -> list[int]:
    """
    Return radix-4 Booth digits for an unsigned multiplier b.

    digits[i] corresponds to coefficient of 2^(2i).
    digit is in {-2, -1, 0, +1, +2}.
    """
```

### Product reconstruction

```python
def booth_mul(a: int, b: int, width: int) -> int:
    digits = booth_radix4_digits_unsigned(b, width)
    return sum(d * (a << (2*i)) for i, d in enumerate(digits))
```

### Chunk decomposition

```python
def decompose_base(x: int, chunk_bits: int, chunks: int | None = None) -> list[int]:
    mask = (1 << chunk_bits) - 1
    out = []
    while x:
        out.append(x & mask)
        x >>= chunk_bits
    if chunks is not None:
        out += [0] * (chunks - len(out))
    return out
```

### Chunk product count

```python
def chunk_product_count(operand_bits: int, chunk_bits: int) -> int:
    chunks = math.ceil(operand_bits / chunk_bits)
    return chunks * chunks
```

### RNS limb count

```python
def rns_limb_count(total_modulus_bits: int, prime_bits: int) -> int:
    return math.ceil(total_modulus_bits / prime_bits)
```

### Cost model

```python
@dataclass
class CostModel:
    total_modulus_bits: int
    prime_bits: int
    cost_per_limb: float

    @property
    def limbs(self):
        return math.ceil(self.total_modulus_bits / self.prime_bits)

    @property
    def total_cost(self):
        return self.limbs * self.cost_per_limb
```

---

# Main educational comparisons to include

## Comparison 1: Booth versus naive shift-add

For 36-bit random multipliers:

```text
Naive binary:
    average ≈ 18 nonzero partial rows

Radix-4 Booth:
    average ≈ 13.5 nonzero partial rows
```

Conclusion:

```text
Booth helps, but not enough to make software shift-add attractive compared to native multiplication.
```

---

## Comparison 2: 36-bit RNS prime versus 31-bit RNS prime

For same total modulus bits:

```text
36-bit primes:
    fewer limbs
    product requires 72 bits

31-bit primes:
    more limbs
    product requires 62 bits
    fits naturally in uint64
```

Approximate limb increase:

[
\frac{36}{31} \approx 1.16
]

Conclusion:

```text
31-bit primes require about 16% more limbs than 36-bit primes, but each modular multiplication is much easier on commodity hardware.
```

This is probably the most important result in the notebook.

---

## Comparison 3: 36-bit on INT8

For 36-bit operands:

```text
chunk width = 8 bits
chunks = ceil(36 / 8) = 5
cross-products = 5 × 5 = 25
```

Conclusion:

```text
INT8 hardware must be more than 25× better before it even starts to compensate for chunking, and real recomposition/reduction overhead pushes the break-even point higher.
```

---

# Suggested final narrative

The final notebook conclusion should say something like:

```text
Booth's algorithm is valuable for understanding how multipliers reduce partial products.

However, for FHE-style RNS arithmetic on commodity hardware, the dominant question is not:
    "Can I Booth-recode a 36-bit multiplication?"

The more practical question is:
    "Can I choose RNS primes so that modular multiplication fits the native arithmetic width?"

For x86-64 and AArch64 CPUs, 30/31-bit RNS primes are attractive because products fit comfortably in 64-bit arithmetic.

For NVIDIA Tensor Cores, INT8 chunking is possible in principle, but exact 36-bit arithmetic requires 25 chunk-products plus recomposition and modular reduction. That only becomes attractive if the workload is sufficiently dense and matrix-shaped.
```

That gives SW developers the correct abstraction boundary:

```text
Booth: multiplier-internal partial-product optimization.

Limb decomposition: software method for representing larger arithmetic using smaller machine words.

RNS prime-size selection: algorithm/hardware co-design choice.

Tensor Core INT8: high-throughput low-precision matrix engine, not a natural exact modular arithmetic engine unless carefully reformulated.
```
