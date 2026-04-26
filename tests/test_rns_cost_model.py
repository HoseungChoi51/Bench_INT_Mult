from __future__ import annotations

import pytest

from rns_arithmetic.rns_cost_model import CostModel, break_even_ratio, rns_limb_count


def test_rns_limb_count_known():
    assert rns_limb_count(720, 36) == 20
    assert rns_limb_count(720, 31) == 24
    assert rns_limb_count(360, 36) == 10
    assert rns_limb_count(360, 31) == 12


def test_rns_limb_count_validation():
    with pytest.raises(ValueError):
        rns_limb_count(0, 36)
    with pytest.raises(ValueError):
        rns_limb_count(720, 0)


def test_cost_model_total_cost():
    a = CostModel(total_modulus_bits=720, prime_bits=31, cost_per_limb=1.0)
    b = CostModel(total_modulus_bits=720, prime_bits=36, cost_per_limb=1.16)
    assert a.limbs == 24
    assert b.limbs == 20
    assert a.total_cost == pytest.approx(24.0)
    assert b.total_cost == pytest.approx(20 * 1.16)


def test_cost_model_validation():
    with pytest.raises(ValueError):
        CostModel(total_modulus_bits=-1, prime_bits=31, cost_per_limb=1.0)
    with pytest.raises(ValueError):
        CostModel(total_modulus_bits=720, prime_bits=0, cost_per_limb=1.0)
    with pytest.raises(ValueError):
        CostModel(total_modulus_bits=720, prime_bits=31, cost_per_limb=-0.01)


def test_break_even_ratio_31_vs_36_at_720():
    # The headline figure of the whole notebook series.
    ratio = break_even_ratio(31, 36, 720)
    # 24 / 20 = 1.20 (at Q=720, ceiling effects keep us slightly above the
    # asymptotic 36/31 ≈ 1.161). Both quantities are exposed in Notebook 3.
    assert ratio == pytest.approx(24 / 20)
    # The asymptotic value approached as Q grows.
    big_q_ratio = break_even_ratio(31, 36, 31 * 36 * 100)
    assert big_q_ratio == pytest.approx(36 / 31, rel=1e-3)


def test_break_even_ratio_validation():
    with pytest.raises(ValueError):
        break_even_ratio(36, 31, 720)
    with pytest.raises(ValueError):
        break_even_ratio(31, 31, 720)
