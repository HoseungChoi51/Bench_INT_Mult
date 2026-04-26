"""First-order cost model for RNS modular multiplication.

The model in Notebook 3 compares two RNS prime-size strategies (e.g. 31-bit
vs. 36-bit primes) at fixed total modulus size :math:`Q_\\text{bits}`. It
counts a single per-limb modular multiplication as the unit of work; the
load-bearing quantity is the **break-even ratio** between the two strategies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def rns_limb_count(total_modulus_bits: int, prime_bits: int) -> int:
    """Number of RNS limbs needed to reach a target total modulus size.

    Parameters
    ----------
    total_modulus_bits : int
        Target :math:`\\sum_i \\log_2 q_i`, in bits. Must be positive.
    prime_bits : int
        Bit width of each RNS prime. Must be positive.

    Returns
    -------
    int
        ``ceil(total_modulus_bits / prime_bits)``.

    Examples
    --------
    >>> rns_limb_count(720, 36)
    20
    >>> rns_limb_count(720, 31)
    24
    """
    if total_modulus_bits <= 0:
        raise ValueError(f"total_modulus_bits must be positive, got {total_modulus_bits}")
    if prime_bits <= 0:
        raise ValueError(f"prime_bits must be positive, got {prime_bits}")
    return math.ceil(total_modulus_bits / prime_bits)


@dataclass(frozen=True)
class CostModel:
    """A simple per-strategy cost summary at fixed total modulus size.

    Attributes
    ----------
    total_modulus_bits : int
        Target total modulus size in bits.
    prime_bits : int
        Bit width of each RNS prime.
    cost_per_limb : float
        Abstract cost of one modular multiplication for this prime size,
        in arbitrary units. The per-limb costs of two strategies should be
        expressed in the *same* units to be meaningfully comparable.
    """

    total_modulus_bits: int
    prime_bits: int
    cost_per_limb: float

    def __post_init__(self) -> None:
        if self.total_modulus_bits <= 0:
            raise ValueError(
                f"total_modulus_bits must be positive, got {self.total_modulus_bits}"
            )
        if self.prime_bits <= 0:
            raise ValueError(f"prime_bits must be positive, got {self.prime_bits}")
        if self.cost_per_limb < 0:
            raise ValueError(f"cost_per_limb must be non-negative, got {self.cost_per_limb}")

    @property
    def limbs(self) -> int:
        """Number of RNS limbs required."""
        return rns_limb_count(self.total_modulus_bits, self.prime_bits)

    @property
    def total_cost(self) -> float:
        """Total per-strategy cost: ``limbs * cost_per_limb``."""
        return self.limbs * self.cost_per_limb


def break_even_ratio(
    small_prime_bits: int,
    big_prime_bits: int,
    total_modulus_bits: int,
) -> float:
    """Per-limb cost ratio at which the two strategies have equal total cost.

    For two prime sizes with ``small_prime_bits < big_prime_bits``, the
    small-prime strategy needs more limbs but should have a cheaper per-limb
    modular multiplication. The break-even ratio is

    .. math::
        \\frac{C_\\text{big}}{C_\\text{small}}
        = \\frac{\\lceil Q_\\text{bits} / \\text{small} \\rceil}
                {\\lceil Q_\\text{bits} / \\text{big} \\rceil}.

    For large :math:`Q_\\text{bits}`, this asymptotes to
    :math:`\\text{big} / \\text{small}` — for the (31, 36) pair, ≈ 1.16.
    The small-prime strategy wins when the actual per-limb cost ratio
    exceeds this value.

    Parameters
    ----------
    small_prime_bits, big_prime_bits : int
        Two RNS prime sizes. ``small_prime_bits`` must be strictly less than
        ``big_prime_bits``.
    total_modulus_bits : int
        Target total modulus size, in bits.

    Returns
    -------
    float
        The break-even ratio :math:`C_\\text{big} / C_\\text{small}`.
    """
    if small_prime_bits >= big_prime_bits:
        raise ValueError(
            f"small_prime_bits ({small_prime_bits}) must be < "
            f"big_prime_bits ({big_prime_bits})"
        )
    small_limbs = rns_limb_count(total_modulus_bits, small_prime_bits)
    big_limbs = rns_limb_count(total_modulus_bits, big_prime_bits)
    return small_limbs / big_limbs
