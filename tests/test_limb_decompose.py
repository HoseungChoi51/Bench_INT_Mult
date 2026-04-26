from __future__ import annotations

import random

import pytest

from rns_arithmetic.limb_decompose import (
    chunk_product_count,
    decompose_base,
    product_36_as_32_words,
    product_as_chunks,
    recompose_base,
)


def _rng(seed: int = 20260426) -> random.Random:
    return random.Random(seed)


@pytest.mark.parametrize("chunk_bits", [4, 8, 16, 32])
def test_decompose_recompose_round_trip(chunk_bits: int) -> None:
    rng = _rng()
    for _ in range(1000):
        x = rng.randrange(0, 1 << 80)
        words = decompose_base(x, chunk_bits)
        assert recompose_base(words, chunk_bits) == x


def test_decompose_zero_padding():
    words = decompose_base(0xDEADBEEF, 8, chunks=6)
    assert len(words) == 6
    assert words[-2:] == [0, 0]
    assert recompose_base(words, 8) == 0xDEADBEEF


def test_decompose_zero():
    assert decompose_base(0, 8) == []
    assert decompose_base(0, 8, chunks=4) == [0, 0, 0, 0]


def test_decompose_too_few_chunks():
    with pytest.raises(ValueError):
        decompose_base(1 << 33, 8, chunks=4)  # needs 5 bytes, only 4 requested


def test_recompose_rejects_oversized_word():
    with pytest.raises(ValueError):
        recompose_base([256], 8)


def test_product_36_as_32_words_matches_python():
    rng = _rng(seed=777)
    for _ in range(2000):
        a = rng.randrange(0, 1 << 36)
        b = rng.randrange(0, 1 << 36)
        assert product_36_as_32_words(a, b) == a * b


def test_product_36_as_32_words_rejects_oversize():
    with pytest.raises(ValueError):
        product_36_as_32_words(1 << 36, 0)


@pytest.mark.parametrize("chunk_bits", [4, 8, 16])
def test_product_as_chunks_round_trip(chunk_bits: int) -> None:
    rng = _rng(seed=99)
    for _ in range(500):
        a = rng.randrange(0, 1 << 36)
        b = rng.randrange(0, 1 << 36)
        chunks_out = product_as_chunks(a, b, chunk_bits)
        assert recompose_base(chunks_out, chunk_bits) == a * b


def test_chunk_product_count_known():
    assert chunk_product_count(36, 8) == 25
    assert chunk_product_count(36, 32) == 4
    assert chunk_product_count(36, 16) == 9
    assert chunk_product_count(64, 8) == 64
