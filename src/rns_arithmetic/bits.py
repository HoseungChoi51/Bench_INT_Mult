"""Bit-level utilities used across the educational notebooks.

All bit-list functions follow the project-wide convention defined in
``STYLE.md`` §D: bit and chunk lists are **least-significant first**.
"""

from __future__ import annotations


def bit_width(x: int) -> int:
    """Return the minimum number of bits needed to represent a non-negative integer.

    Parameters
    ----------
    x : int
        Non-negative integer.

    Returns
    -------
    int
        ``x.bit_length()``. Defined so ``bit_width(0) == 0``.

    Examples
    --------
    >>> bit_width(0)
    0
    >>> bit_width(1)
    1
    >>> bit_width(255)
    8
    """
    if x < 0:
        raise ValueError(f"bit_width is defined for non-negative integers, got {x}")
    return x.bit_length()


def bits_lsb_first(x: int, width: int) -> list[int]:
    """Return the bits of ``x`` as a list of 0/1 values, least-significant first.

    Parameters
    ----------
    x : int
        Non-negative integer to expand. Must satisfy ``x < 2**width``.
    width : int
        Number of bits to emit. Higher bits beyond ``bit_width(x)`` are zero-padded.

    Returns
    -------
    list[int]
        A list of length ``width`` whose entry at index ``i`` is bit ``i`` of ``x``.

    Examples
    --------
    >>> bits_lsb_first(0b1011, 6)
    [1, 1, 0, 1, 0, 0]
    """
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    if x >> width:
        raise ValueError(f"x={x} does not fit in width={width} bits")
    return [(x >> i) & 1 for i in range(width)]


def format_binary(x: int, width: int) -> str:
    """Format ``x`` as a fixed-width binary string, most-significant bit first.

    The displayed order (MSB first) is the conventional human-readable form,
    even though our internal bit lists are LSB-first.

    Parameters
    ----------
    x : int
        Non-negative integer to format.
    width : int
        Total width of the output string. Higher bits beyond ``bit_width(x)``
        are zero-padded.

    Returns
    -------
    str
        Binary representation of ``x``, length ``width``.

    Examples
    --------
    >>> format_binary(0b1011, 6)
    '001011'
    """
    if x < 0:
        raise ValueError(f"x must be non-negative, got {x}")
    if width < 0:
        raise ValueError(f"width must be non-negative, got {width}")
    if x >> width:
        raise ValueError(f"x={x} does not fit in width={width} bits")
    return format(x, f"0{width}b")
