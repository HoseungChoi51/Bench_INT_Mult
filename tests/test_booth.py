from __future__ import annotations

import math
import random

import pytest

from rns_arithmetic.booth import (
    booth_mul,
    booth_radix2_digits,
    booth_radix4_cost,
    booth_radix4_digits_unsigned,
    naive_shift_add_cost,
)


def _rng(seed: int = 20260426) -> random.Random:
    return random.Random(seed)


@pytest.mark.parametrize("width", [4, 8, 16, 32, 36])
def test_booth_mul_matches_python(width: int) -> None:
    rng = _rng()
    n_iter = 1000
    for _ in range(n_iter):
        a = rng.randrange(0, 1 << width)
        b = rng.randrange(0, 1 << width)
        assert booth_mul(a, b, width) == a * b


@pytest.mark.parametrize("width", [4, 8, 16, 32, 36])
def test_radix4_digit_set_and_reconstruction(width: int) -> None:
    rng = _rng(seed=42)
    for _ in range(500):
        b = rng.randrange(0, 1 << width)
        digits = booth_radix4_digits_unsigned(b, width)
        # Digit set membership.
        assert all(d in {-2, -1, 0, 1, 2} for d in digits)
        # Reconstruction of b from its own digits.
        reconstructed = sum(d * (1 << (2 * i)) for i, d in enumerate(digits))
        assert reconstructed == b
        # The unsigned variant carries one closure digit beyond the structural
        # group count (see booth_radix4_digits_unsigned docstring).
        assert len(digits) == math.ceil((width + 1) / 2)


@pytest.mark.parametrize("width", [4, 8, 16, 32])
def test_radix2_digit_set_and_reconstruction(width: int) -> None:
    rng = _rng(seed=123)
    for _ in range(500):
        b = rng.randrange(0, 1 << width)
        digits = booth_radix2_digits(b, width)
        assert all(d in {-1, 0, 1} for d in digits)
        reconstructed = sum(d * (1 << i) for i, d in enumerate(digits))
        assert reconstructed == b
        # Same closure digit for unsigned reconstruction.
        assert len(digits) == width + 1


def test_booth_radix4_cost_36_bit():
    cost = booth_radix4_cost(36)
    assert cost["groups"] == 18
    assert cost["expected_nonzero_partial_rows"] == pytest.approx(13.5)


def test_naive_cost_36_bit():
    cost = naive_shift_add_cost(36)
    assert cost["groups"] == 36
    assert cost["expected_nonzero_partial_rows"] == pytest.approx(18.0)


def test_input_validation():
    with pytest.raises(ValueError):
        booth_radix4_digits_unsigned(-1, 8)
    with pytest.raises(ValueError):
        booth_radix4_digits_unsigned(1 << 8, 8)
    with pytest.raises(ValueError):
        booth_radix2_digits(1 << 4, 4)
