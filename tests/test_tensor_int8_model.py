from __future__ import annotations

import numpy as np
import pytest

from rns_arithmetic.tensor_int8_model import byte_grid, int8_break_even


def test_byte_grid_36_8_shape_and_corners():
    g = byte_grid(36, 8)
    assert g.shape == (5, 5)
    assert int(g[0, 0]) == 0
    assert int(g[4, 4]) == 8
    # The grid is symmetric and entries depend only on (i + j).
    assert np.array_equal(g, g.T)
    # Diagonals collect terms with the same byte-shift index.
    for k in range(g.shape[0] + g.shape[1] - 1):
        diag = [g[i, j] for i in range(g.shape[0]) for j in range(g.shape[1]) if i + j == k]
        assert all(v == k for v in diag)


def test_byte_grid_36_32_smaller():
    g = byte_grid(36, 32)
    assert g.shape == (2, 2)
    assert int(g[1, 1]) == 2


def test_int8_break_even_basic():
    # PLAN.md §4.5: with K=25 and overhead = 1, INT8 needs >= 25x raw advantage.
    assert int8_break_even(25, 1.0) == pytest.approx(25.0)
    # Higher overhead pushes the break-even point higher.
    assert int8_break_even(25, 2.0) == pytest.approx(50.0)


def test_int8_break_even_validation():
    with pytest.raises(ValueError):
        int8_break_even(0, 1.0)
    with pytest.raises(ValueError):
        int8_break_even(25, 0.5)
