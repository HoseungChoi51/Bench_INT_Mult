"""Multiword (limb / chunk) decomposition utilities.

These functions express a single Python integer as a list of fixed-width
chunks, and recompose chunks back into an integer. The convention for chunk
order is **least-significant first** (``words[0]`` is the LSB), per
``STYLE.md`` §D.

The 36-bit-as-32-bit-chunks helper and the general INT8 chunk-product helper
are the building blocks for the cost models in Notebooks 2 and 4.
"""

from __future__ import annotations

import math


def decompose_base(x: int, chunk_bits: int, chunks: int | None = None) -> list[int]:
    """Decompose ``x`` into chunks of ``chunk_bits`` bits, least-significant first.

    Parameters
    ----------
    x : int
        Non-negative integer to decompose.
    chunk_bits : int
        Bit width of each chunk. Must be positive.
    chunks : int, optional
        If given, zero-pad (or truncate-error) so the returned list has exactly
        ``chunks`` entries. ``x`` must fit in ``chunks * chunk_bits`` bits.

    Returns
    -------
    list[int]
        Chunk values in ``[0, 2**chunk_bits)``, LSB-first.

    Examples
    --------
    >>> decompose_base(0xDEADBEEF, 8)
    [239, 190, 173, 222]
    >>> decompose_base(0xDEADBEEF, 8, chunks=6)
    [239, 190, 173, 222, 0, 0]
    """
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}")
    if chunk_bits <= 0:
        raise ValueError(f"chunk_bits must be positive, got {chunk_bits}")

    mask = (1 << chunk_bits) - 1
    out: list[int] = []
    rem = x
    while rem:
        out.append(rem & mask)
        rem >>= chunk_bits

    if chunks is not None:
        if chunks < 0:
            raise ValueError(f"chunks must be non-negative, got {chunks}")
        if len(out) > chunks:
            raise ValueError(
                f"x={x} needs {len(out)} chunks at chunk_bits={chunk_bits}, "
                f"more than the requested chunks={chunks}"
            )
        out += [0] * (chunks - len(out))
    return out


def recompose_base(words: list[int], chunk_bits: int) -> int:
    """Recompose a least-significant-first chunk list back into a single integer.

    Parameters
    ----------
    words : list[int]
        Chunk values, LSB-first. Each must satisfy ``0 <= w < 2**chunk_bits``.
    chunk_bits : int
        Bit width of each chunk. Must be positive.

    Returns
    -------
    int
        ``sum(words[i] * 2**(i * chunk_bits))``.

    Examples
    --------
    >>> recompose_base([239, 190, 173, 222], 8) == 0xDEADBEEF
    True
    """
    if chunk_bits <= 0:
        raise ValueError(f"chunk_bits must be positive, got {chunk_bits}")

    limit = 1 << chunk_bits
    result = 0
    for i, w in enumerate(words):
        if w < 0 or w >= limit:
            raise ValueError(f"word at index {i} = {w} is out of range [0, 2**{chunk_bits})")
        result += w << (i * chunk_bits)
    return result


def product_36_as_32_words(a: int, b: int) -> int:
    """Compute ``a * b`` for 36-bit operands using a 32-bit-chunk decomposition.

    Splits each operand at the 32-bit boundary,

    .. math::
        a = a_0 + 2^{32} a_1, \\quad b = b_0 + 2^{32} b_1,

    and sums the four cross-products

    .. math::
        a b = a_0 b_0 + 2^{32}(a_0 b_1 + a_1 b_0) + 2^{64} a_1 b_1.

    Used in Notebook 2 to demonstrate the commodity-software view of a
    36 × 36 → 72-bit product.

    Parameters
    ----------
    a, b : int
        Non-negative 36-bit operands, each in ``[0, 2**36)``.

    Returns
    -------
    int
        The exact product ``a * b``.
    """
    if a < 0 or a >> 36:
        raise ValueError(f"a must be a 36-bit non-negative integer, got {a}")
    if b < 0 or b >> 36:
        raise ValueError(f"b must be a 36-bit non-negative integer, got {b}")

    mask32 = (1 << 32) - 1
    a0 = a & mask32
    a1 = a >> 32
    b0 = b & mask32
    b1 = b >> 32

    p00 = a0 * b0
    p01 = a0 * b1
    p10 = a1 * b0
    p11 = a1 * b1

    return p00 + ((p01 + p10) << 32) + (p11 << 64)


def product_as_chunks(a: int, b: int, chunk_bits: int) -> list[int]:
    """Multiply ``a`` and ``b`` via chunk decomposition and return a chunk list.

    Both operands are decomposed into chunks of ``chunk_bits`` bits, all
    cross-products are accumulated with their byte-shift positions, and the
    final integer is re-decomposed into chunks of ``chunk_bits`` bits.

    Parameters
    ----------
    a, b : int
        Non-negative integers.
    chunk_bits : int
        Bit width per chunk. Must be positive.

    Returns
    -------
    list[int]
        The product expressed as a list of ``chunk_bits``-wide chunks,
        LSB-first.

    Notes
    -----
    For Notebook 4: with ``chunk_bits=8`` and 36-bit operands, this performs
    ``5 * 5 = 25`` byte-level cross-products before recomposition.
    """
    if a < 0 or b < 0:
        raise ValueError("a and b must be non-negative")
    if chunk_bits <= 0:
        raise ValueError(f"chunk_bits must be positive, got {chunk_bits}")

    a_chunks = decompose_base(a, chunk_bits)
    b_chunks = decompose_base(b, chunk_bits)

    acc = 0
    for i, ai in enumerate(a_chunks):
        for j, bj in enumerate(b_chunks):
            acc += (ai * bj) << (chunk_bits * (i + j))
    return decompose_base(acc, chunk_bits)


def chunk_product_count(operand_bits: int, chunk_bits: int) -> int:
    """Number of chunk × chunk cross-products needed for a chunked multiplication.

    Parameters
    ----------
    operand_bits : int
        Bit width of each operand.
    chunk_bits : int
        Bit width per chunk.

    Returns
    -------
    int
        ``ceil(operand_bits / chunk_bits) ** 2``.

    Examples
    --------
    >>> chunk_product_count(36, 8)
    25
    >>> chunk_product_count(36, 32)
    4
    """
    if operand_bits < 0:
        raise ValueError(f"operand_bits must be non-negative, got {operand_bits}")
    if chunk_bits <= 0:
        raise ValueError(f"chunk_bits must be positive, got {chunk_bits}")
    chunks = math.ceil(operand_bits / chunk_bits)
    return chunks * chunks
