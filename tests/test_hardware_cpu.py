"""Hardware-specific tests for x86_64 CPU (target: AMD Ryzen 9000-series).

These tests pin down the central arithmetic narratives of the notebook
series on the actual machine running the suite:

- 31-bit RNS modular multiplication is exact through ``np.uint64`` — the
  vectorized path that x86-64 SIMD lanes implement natively
  (Notebook 3 §3.3 Method 1).
- 36-bit operands silently wrap modulo :math:`2^{64}` under the same path,
  losing the high 8 bits of the 72-bit true product (Notebook 2 §2.4).
- The 32-bit chunk decomposition recovers the exact 72-bit product on every
  operand pair (Notebook 2 §2.5).

Throughput is intentionally out of scope; see Notebook 5 for benchmarking.
"""

from __future__ import annotations

import platform
import warnings

import numpy as np
import pytest

from rns_arithmetic.limb_decompose import product_36_as_32_words

X86_64 = platform.machine() in {"x86_64", "AMD64"}

pytestmark = pytest.mark.skipif(
    not X86_64,
    reason=f"x86_64 CPU expected; got {platform.machine()}",
)


def test_cpu_arch_is_x86_64() -> None:
    """Surface the architecture so the suite records what hardware it ran on."""
    assert platform.machine() in {"x86_64", "AMD64"}


def test_uint64_31bit_rns_mulmod_is_exact() -> None:
    """31-bit Mersenne RNS: vectorized ``(a * b) % q`` over uint64 matches truth.

    With operands :math:`< 2^{31}`, the raw product is :math:`< 2^{62}` and
    fits comfortably in one uint64; the modulo reduction then becomes a
    single integer divide. This is the headline practical case for
    commodity CPUs.
    """
    rng = np.random.default_rng(20260426)
    q = (1 << 31) - 1  # 31-bit Mersenne prime
    n = 1_000_000
    a = rng.integers(0, q, size=n, dtype=np.uint64)
    b = rng.integers(0, q, size=n, dtype=np.uint64)
    result = (a * b) % np.uint64(q)

    sample = rng.choice(n, size=2000, replace=False)
    for idx in sample:
        assert int(result[idx]) == (int(a[idx]) * int(b[idx])) % q


def test_uint64_36bit_product_silently_wraps() -> None:
    """36-bit operands: np.uint64 multiplication drops the top 8 bits.

    The 72-bit true product is truncated to its low 64 bits without raising
    an exception. This is the exact failure mode that motivates the chunk
    decomposition / 31-bit prime workarounds.
    """
    a_py = (1 << 36) - 1
    b_py = (1 << 36) - 1
    truth = a_py * b_py
    assert truth.bit_length() == 72  # confirms the 72-bit raw-product claim

    a = np.array([a_py], dtype=np.uint64)
    b = np.array([b_py], dtype=np.uint64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wrapped = int((a * b)[0])

    assert wrapped == truth & ((1 << 64) - 1)
    assert wrapped != truth  # high 8 bits silently dropped


def test_uint64_36bit_chunk_decomposition_is_exact() -> None:
    """The 32-bit-chunk workaround recovers the full 72-bit product on a batch.

    Drives ``product_36_as_32_words`` from a vectorized np.uint64 sample so
    the inputs come from real x86-64 integer arithmetic, not Python big
    ints. The reference is Python's arbitrary-precision multiply.
    """
    rng = np.random.default_rng(0xC0FFEE)
    n = 5_000
    a = rng.integers(0, 1 << 36, size=n, dtype=np.uint64)
    b = rng.integers(0, 1 << 36, size=n, dtype=np.uint64)
    for ai, bi in zip(a.tolist(), b.tolist(), strict=True):
        assert product_36_as_32_words(ai, bi) == ai * bi


def test_int64_31bit_signed_mulmod_is_exact() -> None:
    """Signed-int64 path: ``(a * b) % q`` for ``q < 2**31`` is exact.

    Many SIMD intrinsics (e.g. AVX-512 ``vpmullq``) operate on signed 64-bit
    lanes; this complements the uint64 test for that pipeline.
    """
    rng = np.random.default_rng(7)
    q = (1 << 31) - 1
    n = 100_000
    a = rng.integers(0, q, size=n, dtype=np.int64)
    b = rng.integers(0, q, size=n, dtype=np.int64)
    result = (a * b) % np.int64(q)

    sample = rng.choice(n, size=500, replace=False)
    for idx in sample:
        assert int(result[idx]) == (int(a[idx]) * int(b[idx])) % q
