from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402  (must be set before pyplot is imported indirectly)

from matplotlib.figure import Figure  # noqa: E402

from rns_arithmetic.visualization import (  # noqa: E402
    PALETTE,
    SEMANTIC_COLORS,
    apply_style,
    byte_grid_heatmap,
    plot_breakeven_curve,
    plot_partial_rows_vs_width,
    plot_product_width_vs_prime_bits,
)


def test_semantic_colors_resolve_to_palette():
    palette_values = set(PALETTE.values())
    for key, color in SEMANTIC_COLORS.items():
        assert color in palette_values, f"semantic color {key!r} not in PALETTE"


def test_apply_style_runs():
    apply_style()  # smoke; no assertion beyond "no exception"


def test_plot_partial_rows_vs_width_returns_figure():
    fig = plot_partial_rows_vs_width([4, 8, 16, 32, 36, 64])
    assert isinstance(fig, Figure)


def test_plot_product_width_vs_prime_bits_returns_figure():
    fig = plot_product_width_vs_prime_bits(range(20, 41))
    assert isinstance(fig, Figure)


def test_plot_breakeven_curve_returns_figure():
    fig = plot_breakeven_curve(range(120, 2001, 60))
    assert isinstance(fig, Figure)


def test_byte_grid_heatmap_returns_figure():
    fig = byte_grid_heatmap(36, 8)
    assert isinstance(fig, Figure)
