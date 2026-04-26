"""Cost model for INT8 chunk decomposition on Tensor-Core-style hardware.

For an exact 36-bit modular multiplication mapped onto INT8 chunks, the
operands decompose into ``ceil(36 / 8) = 5`` bytes each, producing a
``5 × 5 = 25`` grid of byte-level cross-products. This module exposes that
grid (as the byte-shift-index matrix used in the Notebook 4 heatmap) and a
break-even check against a hypothetical raw-throughput advantage.
"""

from __future__ import annotations

import math

import numpy as np


def byte_grid(operand_bits: int, chunk_bits: int = 8) -> np.ndarray:
    """Byte-shift-index matrix for a chunked multiplication.

    For operands of width ``operand_bits`` decomposed into chunks of
    ``chunk_bits``, the cross-product :math:`a_i b_j` contributes to byte
    position :math:`i + j` in the recomposed product. The returned matrix
    ``G`` has ``G[i, j] = i + j``; its diagonals collect terms that share
    a recomposition position.

    Parameters
    ----------
    operand_bits : int
        Operand bit width.
    chunk_bits : int, default 8
        Chunk bit width.

    Returns
    -------
    np.ndarray
        Integer matrix of shape ``(chunks, chunks)`` with
        ``chunks = ceil(operand_bits / chunk_bits)``. Used directly as
        heatmap data in Notebook 4.

    Examples
    --------
    >>> byte_grid(36, 8).shape
    (5, 5)
    >>> int(byte_grid(36, 8)[4, 4])
    8
    """
    if operand_bits <= 0:
        raise ValueError(f"operand_bits must be positive, got {operand_bits}")
    if chunk_bits <= 0:
        raise ValueError(f"chunk_bits must be positive, got {chunk_bits}")

    chunks = math.ceil(operand_bits / chunk_bits)
    i = np.arange(chunks, dtype=np.int64).reshape(-1, 1)
    j = np.arange(chunks, dtype=np.int64).reshape(1, -1)
    return i + j


def int8_break_even(K: int, overhead_factor: float) -> float:
    """Required raw-throughput advantage for INT8 chunking to break even.

    The simple model from PLAN.md §4.5: INT8 decomposition is favorable only if

    .. math::
        R > K \\cdot O,

    where :math:`R` is the INT8 raw-throughput advantage over native integer
    multiplication, :math:`K` is the number of chunk-products required for one
    operand-pair multiplication, and :math:`O \\geq 1` lumps together the
    per-product overhead from recomposition, carry handling, and modular
    reduction.

    Parameters
    ----------
    K : int
        Number of chunk × chunk cross-products. For 36-bit operands at
        8-bit chunks, ``K = 25``.
    overhead_factor : float
        Per-product overhead multiplier, :math:`O \\geq 1`.

    Returns
    -------
    float
        The minimum raw-throughput advantage ``K * overhead_factor``.

    Examples
    --------
    >>> int8_break_even(25, 1.0)
    25.0
    """
    if K <= 0:
        raise ValueError(f"K must be positive, got {K}")
    if overhead_factor < 1.0:
        raise ValueError(
            f"overhead_factor must be >= 1.0 (no negative overhead), got {overhead_factor}"
        )
    return float(K) * float(overhead_factor)
