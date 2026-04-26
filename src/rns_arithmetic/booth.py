"""Booth recoding (radix-2 and radix-4) for unsigned multipliers.

Booth recoding rewrites a binary multiplier as a sum of *signed* shifted copies
of the multiplicand. Radix-4 Booth halves the number of partial-product rows
relative to naive shift-add at the cost of allowing digits :math:`\\in
\\{-2,-1,0,+1,+2\\}`.

Conventions used in this module (see ``STYLE.md`` §D):

- Inputs are treated as **unsigned** integers, ``b < 2**width``.
- Returned digit lists are LSB-first: ``digits[i]`` is the coefficient of
  :math:`2^i` for radix-2, and of :math:`2^{2i}` for radix-4.
"""

from __future__ import annotations

import math


def booth_radix2_digits(b: int, width: int) -> list[int]:
    """Recode an unsigned multiplier ``b`` into radix-2 Booth digits.

    Each digit is in :math:`\\{-1, 0, +1\\}` and is the coefficient of
    :math:`2^i` such that

    .. math::
        b = \\sum_{i=0}^{\\text{width}} d_i \\cdot 2^i.

    The recoding rule (with implicit ``b_{-1} = 0`` and ``b_{\\text{width}} = 0``):
    digit at position ``i`` is :math:`b_{i-1} - b_i`. To represent an unsigned
    ``width``-bit input correctly, the function returns **width + 1 digits**
    — the extra position closes a run of 1s that extends to the MSB.

    Parameters
    ----------
    b : int
        Unsigned multiplier, ``0 <= b < 2**width``.
    width : int
        Bit width of ``b``.

    Returns
    -------
    list[int]
        ``width + 1`` Booth digits, LSB-first.
    """
    if b < 0:
        raise ValueError(f"b must be non-negative, got {b}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    if b >> width:
        raise ValueError(f"b={b} does not fit in width={width} bits")

    digits: list[int] = []
    prev = 0
    for i in range(width + 1):
        cur = (b >> i) & 1 if i < width else 0
        digits.append(prev - cur)
        prev = cur
    return digits


def booth_radix4_digits_unsigned(b: int, width: int) -> list[int]:
    """Recode an unsigned multiplier ``b`` into radix-4 Booth digits.

    Each digit is in :math:`\\{-2,-1,0,+1,+2\\}` and is the coefficient of
    :math:`2^{2i}` such that

    .. math::
        b = \\sum_{i=0}^{G} d_i \\cdot 2^{2i}.

    The recoding scans overlapping 3-bit windows :math:`(b_{2i+1}, b_{2i},
    b_{2i-1})` with implicit ``b_{-1} = 0``. To represent an unsigned input
    correctly, the function returns ``ceil((width + 1) / 2)`` digits — one
    more than the *structural* group count reported by :func:`booth_radix4_cost`,
    because that structural count assumes a signed (two's-complement) operand
    in which the sign bit closes the topmost run for free. The extra digit
    here is in :math:`\\{0, +1\\}`.

    Parameters
    ----------
    b : int
        Unsigned multiplier, ``0 <= b < 2**width``.
    width : int
        Bit width of ``b``.

    Returns
    -------
    list[int]
        ``ceil((width + 1) / 2)`` Booth digits in :math:`\\{-2,-1,0,+1,+2\\}`,
        LSB-first.
    """
    if b < 0:
        raise ValueError(f"b must be non-negative, got {b}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    if b >> width:
        raise ValueError(f"b={b} does not fit in width={width} bits")

    extended_width = width + 1  # zero-extend so an MSB-run gets a closing digit
    groups = math.ceil(extended_width / 2)
    digits: list[int] = []
    for i in range(groups):
        b0 = (b >> (2 * i - 1)) & 1 if i > 0 else 0
        b1 = (b >> (2 * i)) & 1 if 2 * i < width else 0
        b2 = (b >> (2 * i + 1)) & 1 if 2 * i + 1 < width else 0
        digits.append(b0 + b1 - 2 * b2)
    return digits


def booth_mul(a: int, b: int, width: int) -> int:
    """Multiply ``a`` by an unsigned ``b`` using radix-4 Booth recoding.

    Reconstructs the product as

    .. math::
        a \\cdot b = \\sum_i d_i \\cdot (a \\ll 2i)

    where the :math:`d_i` are the radix-4 Booth digits of ``b``. This is a
    correctness reference; it is not optimized for speed.

    Parameters
    ----------
    a : int
        Multiplicand. May be any Python integer.
    b : int
        Unsigned multiplier, ``0 <= b < 2**width``.
    width : int
        Bit width of ``b``.

    Returns
    -------
    int
        ``a * b``.
    """
    digits = booth_radix4_digits_unsigned(b, width)
    return sum(d * (a << (2 * i)) for i, d in enumerate(digits))


def booth_radix4_cost(bit_width: int) -> dict[str, float]:
    """Return the *structural* cost estimate for radix-4 Booth at ``bit_width`` bits.

    This is the hardware-multiplier view of the cost (signed multiplier with
    sign extension): ``ceil(bit_width / 2)`` overlapping 3-bit groups, each
    non-zero for 6 of 8 possible patterns. For a uniformly random multiplier,
    the expected number of non-zero signed partial-product rows is therefore
    ``0.75 * groups``.

    The unsigned reconstruction in :func:`booth_radix4_digits_unsigned`
    needs one additional closure digit (in :math:`\\{0, +1\\}`); that extra
    digit is structural overhead for unsigned operands but is not counted in
    the per-group cost reported here, which models the per-bit-pair partial
    product cost inside a multiplier.

    Parameters
    ----------
    bit_width : int
        Bit width of the multiplier.

    Returns
    -------
    dict
        ``{"groups": int, "expected_nonzero_partial_rows": float}``.

    Examples
    --------
    >>> booth_radix4_cost(36)
    {'groups': 18, 'expected_nonzero_partial_rows': 13.5}
    """
    if bit_width < 0:
        raise ValueError(f"bit_width must be non-negative, got {bit_width}")
    groups = math.ceil(bit_width / 2)
    return {
        "groups": groups,
        "expected_nonzero_partial_rows": 0.75 * groups,
    }


def naive_shift_add_cost(bit_width: int) -> dict[str, float]:
    """Return a structural cost estimate for naive shift-add at ``bit_width`` bits.

    For a uniformly random ``bit_width``-bit multiplier, each bit is 1 with
    probability 0.5, so the expected number of non-zero partial-product rows is
    ``bit_width / 2``.

    Parameters
    ----------
    bit_width : int
        Bit width of the multiplier.

    Returns
    -------
    dict
        ``{"groups": int, "expected_nonzero_partial_rows": float}``.

    Examples
    --------
    >>> naive_shift_add_cost(36)
    {'groups': 36, 'expected_nonzero_partial_rows': 18.0}
    """
    if bit_width < 0:
        raise ValueError(f"bit_width must be non-negative, got {bit_width}")
    return {
        "groups": bit_width,
        "expected_nonzero_partial_rows": bit_width / 2,
    }
