"""Shared plotting utilities and the project-wide color palette.

All notebooks import figure helpers from this module so that semantic colors,
figure sizing, and reference-line conventions stay consistent across the
series. See ``STYLE.md`` §E for the enforced rules.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from . import booth, rns_cost_model

# --- Palette ---------------------------------------------------------------
# Okabe-Ito qualitative palette: distinguishable to most colorblind viewers.
PALETTE: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "gray": "#777777",
}

# Semantic role -> palette key. Stable across notebooks (see STYLE.md §E).
SEMANTIC_COLORS: dict[str, str] = {
    "naive": PALETTE["sky_blue"],
    "booth_r2": PALETTE["bluish_green"],
    "booth_r4": PALETTE["vermilion"],
    "prime_31": PALETTE["blue"],
    "prime_36": PALETTE["orange"],
    "int8_chunk": PALETTE["reddish_purple"],
    "reference": PALETTE["gray"],
}


def apply_style() -> None:
    """Apply the project-wide matplotlib rcParams.

    Call once near the top of each notebook (after the imports cell).
    """
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.constrained_layout.use": True,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "image.cmap": "viridis",
        }
    )


# --- Plots -----------------------------------------------------------------


def plot_partial_rows_vs_width(widths: Iterable[int]) -> Figure:
    """Expected number of non-zero partial-product rows vs. multiplier bit width.

    Plots naive shift-add (expected ``w / 2`` rows) against radix-4 Booth
    (expected ``0.75 * ceil(w / 2)`` rows). Used in Notebook 1 §1.5.

    Parameters
    ----------
    widths : Iterable[int]
        Bit widths to plot, e.g. ``[4, 8, 16, 32, 36, 64]``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ws = list(widths)
    naive_rows = [booth.naive_shift_add_cost(w)["expected_nonzero_partial_rows"] for w in ws]
    booth_rows = [booth.booth_radix4_cost(w)["expected_nonzero_partial_rows"] for w in ws]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws, naive_rows, "o-", color=SEMANTIC_COLORS["naive"], label="naive shift-add")
    ax.plot(ws, booth_rows, "s-", color=SEMANTIC_COLORS["booth_r4"], label="radix-4 Booth")
    ax.set_xlabel("multiplier bit width (bits)")
    ax.set_ylabel("expected non-zero partial-product rows")
    ax.set_title("Partial-product rows vs. multiplier bit width (random inputs)")
    ax.legend()
    return fig


def plot_product_width_vs_prime_bits(
    w_range: Iterable[int],
    word_size_bits: int = 64,
) -> Figure:
    """Product bit width vs. RNS prime bit width, with the word-size boundary.

    For an RNS prime of ``w`` bits, the raw product is ``2 * w`` bits. The
    horizontal reference line marks the native word size (default 64 bits);
    primes that push past it require multiword arithmetic. Used in
    Notebook 3 §3.2.

    Parameters
    ----------
    w_range : Iterable[int]
        Prime bit widths to plot, e.g. ``range(20, 61)``.
    word_size_bits : int, default 64
        Reference word size in bits.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ws = np.fromiter(w_range, dtype=np.int64)
    product_bits = 2 * ws

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ws, product_bits, "-", color=SEMANTIC_COLORS["prime_36"], label="product width = 2w")
    ax.axhline(
        word_size_bits,
        linestyle="--",
        color=SEMANTIC_COLORS["reference"],
        label=f"{word_size_bits}-bit word boundary",
    )
    ax.axvline(
        word_size_bits / 2,
        linestyle=":",
        color=SEMANTIC_COLORS["prime_31"],
        alpha=0.6,
        label=f"prime_bits = {word_size_bits // 2} (boundary)",
    )
    ax.set_xlabel("RNS prime bit width w (bits)")
    ax.set_ylabel("raw product bit width 2w (bits)")
    ax.set_title("Product width vs. RNS prime size")
    ax.legend()
    return fig


def plot_breakeven_curve(
    total_modulus_bits_range: Iterable[int],
    small_prime_bits: int = 31,
    big_prime_bits: int = 36,
) -> Figure:
    """Break-even per-limb cost ratio vs. total modulus size.

    The y-axis is :math:`C_\\text{big} / C_\\text{small}` at which the two
    strategies tie. The asymptotic value
    ``big_prime_bits / small_prime_bits`` (≈ 1.16 for the 31 vs. 36 pair)
    is annotated as a horizontal reference line. Used in Notebook 3 §3.5.

    Parameters
    ----------
    total_modulus_bits_range : Iterable[int]
        Total modulus sizes to sweep, in bits.
    small_prime_bits, big_prime_bits : int
        The two prime sizes to compare.

    Returns
    -------
    matplotlib.figure.Figure
    """
    qs = np.fromiter(total_modulus_bits_range, dtype=np.int64)
    ratios = np.array(
        [rns_cost_model.break_even_ratio(small_prime_bits, big_prime_bits, int(q)) for q in qs]
    )
    asymptote = big_prime_bits / small_prime_bits

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        qs,
        ratios,
        "-",
        color=SEMANTIC_COLORS["prime_31"],
        label=f"break-even ratio (primes {small_prime_bits} vs. {big_prime_bits})",
    )
    ax.axhline(
        asymptote,
        linestyle="--",
        color=SEMANTIC_COLORS["reference"],
        label=f"asymptote = {big_prime_bits}/{small_prime_bits} ≈ {asymptote:.3f}",
    )
    ax.set_xlabel("total modulus size $Q_\\text{bits}$ (bits)")
    ax.set_ylabel(r"break-even ratio $C_\text{big} / C_\text{small}$")
    ax.set_title("Break-even per-limb cost ratio vs. total modulus size")
    ax.legend()
    return fig


def byte_grid_heatmap(operand_bits: int, chunk_bits: int = 8) -> Figure:
    """Heatmap of the byte-shift-index matrix for chunked multiplication.

    Each cell ``(i, j)`` shows the byte position ``i + j`` of the cross-product
    :math:`a_i b_j`. Cells on the same diagonal contribute to the same byte of
    the recomposed product. Used in Notebook 4 §4.3.

    Parameters
    ----------
    operand_bits : int
        Operand bit width.
    chunk_bits : int, default 8
        Chunk bit width.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from . import tensor_int8_model

    g = tensor_int8_model.byte_grid(operand_bits, chunk_bits)
    chunks = g.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(g, cmap="viridis", origin="upper")
    for i in range(chunks):
        for j in range(chunks):
            ax.text(j, i, str(int(g[i, j])), ha="center", va="center", color="white", fontsize=10)
    ax.set_xticks(range(chunks))
    ax.set_yticks(range(chunks))
    ax.set_xticklabels([f"$b_{j}$" for j in range(chunks)])
    ax.set_yticklabels([f"$a_{i}$" for i in range(chunks)])
    ax.set_xlabel(f"chunk index of b ({chunk_bits}-bit chunks)")
    ax.set_ylabel(f"chunk index of a ({chunk_bits}-bit chunks)")
    ax.set_title(
        f"Byte-shift index of cross-products: "
        f"{operand_bits}-bit operands, {chunk_bits}-bit chunks "
        f"({chunks * chunks} products)"
    )
    fig.colorbar(im, ax=ax, label="byte-shift index $i + j$")
    ax.grid(False)
    return fig


def expected_nonzero_for_widths(widths: Iterable[int]) -> dict[str, list[float]]:
    """Convenience: return naive vs. radix-4 expected non-zero rows for a sweep.

    Useful in notebook tables.
    """
    ws = list(widths)
    return {
        "bit_width": [float(w) for w in ws],
        "naive_expected_rows": [
            booth.naive_shift_add_cost(w)["expected_nonzero_partial_rows"] for w in ws
        ],
        "booth_r4_expected_rows": [
            booth.booth_radix4_cost(w)["expected_nonzero_partial_rows"] for w in ws
        ],
        "booth_r4_groups": [float(math.ceil(w / 2)) for w in ws],
    }
