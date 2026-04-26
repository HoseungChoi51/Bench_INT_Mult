from __future__ import annotations

import pytest

from rns_arithmetic.bits import bit_width, bits_lsb_first, format_binary


def test_bit_width_zero_and_small():
    assert bit_width(0) == 0
    assert bit_width(1) == 1
    assert bit_width(255) == 8
    assert bit_width((1 << 36) - 1) == 36


def test_bit_width_rejects_negative():
    with pytest.raises(ValueError):
        bit_width(-1)


def test_bits_lsb_first_basic():
    assert bits_lsb_first(0b1011, 6) == [1, 1, 0, 1, 0, 0]
    assert bits_lsb_first(0, 4) == [0, 0, 0, 0]
    # Edge: max value at given width.
    assert bits_lsb_first((1 << 8) - 1, 8) == [1] * 8


def test_bits_lsb_first_round_trip():
    for x in [0, 1, 7, 0xDEADBEEF, (1 << 36) - 1]:
        w = max(x.bit_length(), 1)
        bits = bits_lsb_first(x, w)
        recon = sum(b << i for i, b in enumerate(bits))
        assert recon == x


def test_bits_lsb_first_overflow():
    with pytest.raises(ValueError):
        bits_lsb_first(0b10000, 4)


def test_format_binary():
    assert format_binary(0b1011, 6) == "001011"
    assert format_binary(0, 4) == "0000"
    assert format_binary(255, 8) == "11111111"


def test_format_binary_overflow():
    with pytest.raises(ValueError):
        format_binary(256, 8)
