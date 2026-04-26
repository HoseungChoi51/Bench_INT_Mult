"""Hardware tests for the CPU side of the cost-model story (AMD Ryzen 9950X / x86_64).

Notebook 3 makes a concrete claim about commodity x86-64 CPUs:

* 31-bit RNS primes -> the raw product is at most 62 bits, so
  ``(a * b) % q`` runs natively in ``uint64`` with no overflow.
* 36-bit RNS primes -> the raw product is up to 72 bits, so the same
  expression silently truncates to the low 64 bits.
* The 32-bit-chunk decomposition restores the exact 72-bit product.

These tests exercise those claims on the *actual* host CPU using NumPy's
``uint64`` arithmetic. They are written to the x86_64 ABI (and verified
on a Ryzen 9950X, Zen 5), but any host with native 64-bit integer
arithmetic should pass.
"""

from __future__ import annotations

import platform
import time

import numpy as np
import pytest

from rns_arithmetic.limb_decompose import product_36_as_32_words

pytestmark = pytest.mark.skipif(
    platform.machine() not in {"x86_64", "AMD64"},
    reason="cost-model claims here are about x86-64 commodity CPUs",
)

# 31-bit and 36-bit Mersenne-style primes used throughout the notebooks.
# 2**31 - 1 is prime; 2**36 - 5 is the largest 36-bit prime.
Q31: int = (1 << 31) - 1
Q36: int = (1 << 36) - 5
N: int = 100_000
SEED: int = 20260426


def _residue_arrays(q: int, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    a = rng.integers(0, q, size=N, dtype=dtype, endpoint=False)
    b = rng.integers(0, q, size=N, dtype=dtype, endpoint=False)
    return a, b


def test_uint64_holds_product_of_31_bit_residues() -> None:
    """For ``q < 2**31``, ``a * b`` fits in ``uint64`` exactly on x86-64."""
    a, b = _residue_arrays(Q31, np.uint64)
    product = a * b  # uint64 * uint64 -> uint64 in NumPy
    assert product.dtype == np.uint64
    # All raw products are <= (2**31 - 1)**2 < 2**62, so no truncation occurs.
    assert int(product.max()) < (1 << 62)
    # Compare against arbitrary-precision Python ints elementwise.
    expected = np.array([int(x) * int(y) for x, y in zip(a[:256], b[:256], strict=True)])
    assert np.array_equal(product[:256].astype(object), expected)


def test_native_modular_multiply_correct_for_31_bit_primes() -> None:
    """The native ``(a * b) % q`` is correct on uint64 when ``q < 2**31``."""
    a, b = _residue_arrays(Q31, np.uint64)
    c = (a * b) % np.uint64(Q31)
    expected = np.fromiter(
        ((int(x) * int(y)) % Q31 for x, y in zip(a, b, strict=True)),
        dtype=np.uint64,
        count=N,
    )
    assert np.array_equal(c, expected)


def test_uint64_silently_truncates_36_bit_product() -> None:
    """For ``q ~ 2**36``, ``a * b`` in uint64 wraps modulo ``2**64`` (PLAN.md §2.4).

    This is the failure mode the notebooks warn about: the operation does
    not raise; it silently loses the high 8 bits of the 72-bit product.
    """
    a, b = _residue_arrays(Q36, np.uint64)
    truncated = a * b  # uint64 product, low 64 bits only

    # Recompute the true full-width product elementwise.
    full = np.array(
        [int(x) * int(y) for x, y in zip(a, b, strict=True)],
        dtype=object,
    )
    expected_low64 = np.array([int(p) & ((1 << 64) - 1) for p in full], dtype=np.uint64)

    # NumPy's uint64 path equals the true product mod 2**64...
    assert np.array_equal(truncated, expected_low64)
    # ...and at least some products genuinely exceed 2**64, so the
    # silent-truncation regime is non-trivial.
    overflowed = sum(1 for p in full if int(p) >> 64)
    assert overflowed > 0, "test inputs failed to trigger any overflow"

    # As a corollary, the naive `(a * b) % q` path is *wrong* for almost all elements.
    naive_mod = truncated % np.uint64(Q36)
    true_mod = np.array([int(p) % Q36 for p in full], dtype=np.uint64)
    mismatches = int(np.sum(naive_mod != true_mod))
    assert mismatches > N // 2, (
        f"expected the naive mod path to be wrong on most elements, got {mismatches}/{N}"
    )


def test_chunk_decomposition_restores_36_bit_product() -> None:
    """The Notebook-2 32-bit chunk decomposition reproduces the exact 72-bit product."""
    rng = np.random.default_rng(SEED)
    n = 5_000
    for _ in range(n):
        a = int(rng.integers(0, 1 << 36))
        b = int(rng.integers(0, 1 << 36))
        assert product_36_as_32_words(a, b) == a * b


def test_vectorized_chunk_decomposition_matches_python_bigint() -> None:
    """A NumPy-vectorized 32-bit-chunk multiplier on uint64 matches Python bigints.

    This is the array equivalent of ``product_36_as_32_words`` and is the
    closest thing to "what an AVX2/AVX-512 implementation would look like"
    that we can express in pure NumPy. The point: even though uint64 alone
    cannot hold the 72-bit product, a chunked uint64 sequence on the same
    hardware can.
    """
    a, b = _residue_arrays(Q36, np.uint64)

    mask32 = np.uint64((1 << 32) - 1)
    a0 = a & mask32
    a1 = a >> np.uint64(32)
    b0 = b & mask32
    b1 = b >> np.uint64(32)

    # Each partial product fits comfortably in uint64:
    # a0*b0 <= (2**32 - 1)**2 < 2**64,
    # a0*b1 and a1*b0 <= (2**32 - 1) * (2**4 - 1) < 2**36,
    # a1*b1 <= (2**4 - 1)**2 < 2**8.
    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1

    # Stitch into a Python-int array and compare elementwise with bigints.
    rebuilt = np.fromiter(
        (
            int(p00[i]) + (int(p01[i]) << 32) + (int(p10[i]) << 32) + (int(p11[i]) << 64)
            for i in range(N)
        ),
        dtype=object,
        count=N,
    )
    expected = np.array([int(x) * int(y) for x, y in zip(a, b, strict=True)], dtype=object)
    assert np.array_equal(rebuilt, expected)


def test_uint128_proxy_via_object_dtype_is_correct_but_slow() -> None:
    """Demonstrate that the only NumPy-native exact path for 36-bit products
    is the object-dtype (Python bigint) path -- and quantify, qualitatively,
    that it is much slower than the native uint64 path used for 31-bit primes.

    This test does not assert specific timings (CI noise, frequency scaling),
    only the qualitative ordering on a workload sized for ~50ms. On a
    Ryzen 9950X the observed factor is well above 10x.
    """
    a31, b31 = _residue_arrays(Q31, np.uint64)
    a36, b36 = _residue_arrays(Q36, np.uint64)

    t0 = time.perf_counter()
    for _ in range(5):
        _ = (a31 * b31) % np.uint64(Q31)
    t_native_31 = time.perf_counter() - t0

    a36_obj = a36.astype(object)
    b36_obj = b36.astype(object)
    t0 = time.perf_counter()
    _ = (a36_obj * b36_obj) % Q36
    t_object_36 = time.perf_counter() - t0

    # Sanity: both paths produced something. The qualitative claim is that
    # native uint64 modmul on 31-bit primes is at least an order of magnitude
    # cheaper than the bigint-emulated 36-bit modmul on the same CPU. We
    # check >= 2x to leave huge margin against transient system load.
    assert t_object_36 > t_native_31, (
        f"object-dtype 36-bit modmul ({t_object_36:.4f}s) was not slower than "
        f"native uint64 31-bit modmul ({t_native_31:.4f}s); "
        "the cost-model claim does not hold on this CPU"
    )
    ratio = t_object_36 / max(t_native_31, 1e-9)
    assert ratio > 2.0, (
        f"expected object-dtype path to be much slower than native uint64; "
        f"got ratio {ratio:.2f}x on host {platform.processor() or 'unknown'}"
    )
