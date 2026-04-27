"""Render comparison plots from the bench JSONL outputs.

Reads the same JSONL files as ``scripts/compare.py`` and writes a small
fixed set of PNGs into the output directory:

- ``layer_b_throughput.png``       — raw GEMM throughput vs shape, per
  (device, backend); log-y because INT8 TOPS and FP64 TFLOPS span 3
  orders of magnitude.
- ``layer_c_modmul.png``           — exact 36-bit modmul throughput vs
  shape, best backend per device; the headline FHE-relevance plot.
- ``layer_c_modmul_per_dollar.png`` — same data, divided by device price;
  answers the ~2× price-ratio question directly.
- ``headline_4096.png``            — at one representative shape (4096³,
  the largest size where Layer C tail-off has not yet dominated), bar
  chart of best-throughput-per-device for both raw GEMM and exact
  modmul, side by side.

Style follows ``src/rns_arithmetic/visualization.py`` (Okabe-Ito palette,
constrained layout, reference lines dashed gray with inline labels).

Usage::

    python scripts/plot_summary.py bench-results/*.jsonl --out-dir bench-results/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rns_arithmetic.visualization import PALETTE, apply_style  # noqa: E402
from scripts._bench_common import DEVICE_PRICES_USD  # noqa: E402

# A semantic color per (device, backend). Stable across plots so the
# reader doesn't have to relearn legends between figures.
DEVICE_BACKEND_COLOR: dict[tuple[str, str], str] = {
    ("RTX5090", "int8"): PALETTE["vermilion"],     # tensor-core INT8
    ("RTX5090", "tf32"): PALETTE["orange"],
    ("RTX5090", "fp32"): PALETTE["blue"],
    ("RTX5090", "fp64"): PALETTE["sky_blue"],
    ("Blackhole", "int8"): PALETTE["bluish_green"],
    ("Blackhole", "bf16"): PALETTE["reddish_purple"],
    ("Blackhole", "tf32"): PALETTE["yellow"],
    ("Blackhole", "fp32"): PALETTE["yellow"],
    ("Blackhole", "fp32_sfpu"): PALETTE["black"],
    ("CPU", "cpu"): PALETTE["gray"],
}

DEVICE_MARKER: dict[str, str] = {
    "RTX5090": "o",
    "Blackhole": "s",
    "CPU": "x",
}

BACKEND_CLASS_MAP = {
    "cublaslt_int8": "int8",
    "tt_llk_int8": "int8",
    "cublaslt_tf32": "tf32",
    "tt_llk_tf32": "tf32",
    "cublaslt_bf16": "bf16",
    "tt_llk_bf16": "bf16",
    "cublaslt_fp32": "fp32",
    "tt_llk_fp32_matrix": "fp32",
    "tt_llk_sfpu_fp32": "fp32_sfpu",
    "cublaslt_fp64": "fp64",
    "cpu_int128": "cpu",
}


def backend_class(backend: str) -> str:
    return BACKEND_CLASS_MAP.get(backend, backend)


def load_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in paths:
        if not p.is_file():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def _color(device: str, bc: str) -> str:
    return DEVICE_BACKEND_COLOR.get((device, bc), PALETTE["gray"])


def _label(device: str, bc: str) -> str:
    return f"{device} {bc}"


def is_q36_modmul(op_kind: str) -> bool:
    """Match the headline 36-bit modmul op kind across schema v1 and v2.

    v1 records use bare ``"exact_modmul"`` (implicitly q36 — that's all
    Layer C-minimal supported). v2 splits into ``"exact_modmul_q36"``
    and ``"exact_modmul_q48"``; the q36 variant is the comparable one.
    """
    return op_kind in ("exact_modmul", "exact_modmul_q36")


def is_q48_modmul(op_kind: str) -> bool:
    return op_kind == "exact_modmul_q48"


def is_klss_ip(op_kind: str) -> bool:
    return op_kind.startswith("klss_ip_modmul") or op_kind == "klss_mac"


# --- Plots ------------------------------------------------------------------


def plot_layer_b_throughput(records: list[dict[str, Any]], out_path: Path) -> None:
    layer_b = [r for r in records if r["layer"] == "B" and r.get("throughput") is not None]
    if not layer_b:
        return

    by_series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in layer_b:
        bc = backend_class(r["backend"])
        m = r["shape"]["M"]  # square shapes; M==K==N
        by_series[(r["device"], bc)].append((m, r["throughput"]))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for (device, bc), pts in sorted(by_series.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys,
            marker=DEVICE_MARKER.get(device, "o"),
            color=_color(device, bc),
            label=_label(device, bc),
            linewidth=1.8,
            markersize=6,
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
    ax.set_xlabel("square matmul size N (M=K=N)")
    ax.set_ylabel("throughput (TFLOPS or TOPS, log scale)")
    ax.set_title("Layer B — raw GEMM throughput per (device, backend)")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layer_c_modmul(records: list[dict[str, Any]], out_path: Path) -> None:
    layer_c = [
        r for r in records
        if r["layer"] == "C"
        and r.get("throughput") is not None
        and is_q36_modmul(r["useful_op_kind"])
        and r["device"] != "CPU"  # CPU is on a different shape (n×1×1); separate plot if wanted
    ]
    if not layer_c:
        return

    # Pick best-throughput backend per (device, M) — that's the headline metric.
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for r in layer_c:
        m = r["shape"]["M"]
        key = (r["device"], m)
        prev = best.get(key)
        if prev is None or r["throughput"] > prev["throughput"]:
            best[key] = r

    by_device: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for (device, m), r in best.items():
        bc = backend_class(r["backend"])
        by_device[device].append((m, r["throughput"], bc))

    fig, ax = plt.subplots(figsize=(8, 5))
    for device, pts in sorted(by_device.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bcs = sorted({p[2] for p in pts})
        bc_label = "/".join(bcs)
        ax.plot(
            xs, ys,
            marker=DEVICE_MARKER.get(device, "o"),
            color=_color(device, bcs[0]),
            label=f"{device} (best: {bc_label})",
            linewidth=2.2,
            markersize=8,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
    ax.set_xlabel("square matmul size N (M=K=N)")
    ax.set_ylabel("effective exact 36-bit modmul/s (G_modmul/s)")
    ax.set_title("Layer C — exact 36-bit modular product, best backend per device")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layer_c_per_dollar(records: list[dict[str, Any]], out_path: Path) -> None:
    layer_c = [
        r for r in records
        if r["layer"] == "C"
        and r.get("throughput") is not None
        and is_q36_modmul(r["useful_op_kind"])
        and r["device"] != "CPU"
        and DEVICE_PRICES_USD.get(r["device"], 0) > 0
    ]
    if not layer_c:
        return

    best: dict[tuple[str, int], dict[str, Any]] = {}
    for r in layer_c:
        m = r["shape"]["M"]
        key = (r["device"], m)
        prev = best.get(key)
        if prev is None or r["throughput"] > prev["throughput"]:
            best[key] = r

    by_device: dict[str, list[tuple[int, float, str]]] = defaultdict(list)
    for (device, m), r in best.items():
        bc = backend_class(r["backend"])
        # G_modmul/s per $1000 spend: (G_modmul/s) / (price USD) * 1000.
        per_dollar = r["throughput"] / DEVICE_PRICES_USD[device] * 1000
        by_device[device].append((m, per_dollar, bc))

    fig, ax = plt.subplots(figsize=(8, 5))
    for device, pts in sorted(by_device.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bcs = sorted({p[2] for p in pts})
        ax.plot(
            xs, ys,
            marker=DEVICE_MARKER.get(device, "o"),
            color=_color(device, bcs[0]),
            label=f"{device} (best: {'/'.join(bcs)}, USD {DEVICE_PRICES_USD[device]:.0f})",
            linewidth=2.2,
            markersize=8,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
    ax.set_xlabel("square matmul size N (M=K=N)")
    # Avoid '$' in labels — matplotlib's mathtext steals them.
    ax.set_ylabel("modmul/s per kUSD of device cost  (G_modmul/s per kUSD)")
    ax.set_title("Layer C — exact 36-bit modmul per dollar (price-adjusted)")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_headline_at_size(
    records: list[dict[str, Any]], shape_n: int, out_path: Path
) -> None:
    """Two side-by-side bar charts at one representative shape:
    raw best-backend GEMM (left) and exact modmul (right), per device."""
    raw = [
        r for r in records
        if r["layer"] == "B"
        and r["shape"]["M"] == shape_n
        and r.get("throughput") is not None
    ]
    modmul = [
        r for r in records
        if r["layer"] == "C"
        and r["shape"]["M"] == shape_n
        and r.get("throughput") is not None
        and is_q36_modmul(r["useful_op_kind"])
    ]
    if not raw and not modmul:
        return

    def best_per_device(rs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in rs:
            d = r["device"]
            if d not in out or r["throughput"] > out[d]["throughput"]:
                out[d] = r
        return out

    raw_best = best_per_device(raw)
    mm_best = best_per_device(modmul)
    devices = sorted(set(raw_best) | set(mm_best))

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))

    def bar(ax: plt.Axes, best: dict[str, dict[str, Any]], unit: str, title: str) -> None:
        xs: list[str] = []
        thr: list[float] = []
        per_dollar: list[float] = []
        bcs: list[str] = []
        colors: list[str] = []
        for d in devices:
            if d not in best:
                continue
            r = best[d]
            xs.append(d)
            thr.append(r["throughput"])
            bc = backend_class(r["backend"])
            bcs.append(bc)
            colors.append(_color(d, bc))
            price = DEVICE_PRICES_USD.get(d, 0.0)
            per_dollar.append(r["throughput"] / price * 1000 if price > 0 else float("nan"))

        x_pos = np.arange(len(xs))
        bars = ax.bar(x_pos, thr, color=colors, edgecolor="black", linewidth=0.8)
        for b, v, bc, pd in zip(bars, thr, bcs, per_dollar, strict=True):
            label = f"{v:.2f} {unit}\n({bc})"
            ax.text(b.get_x() + b.get_width() / 2, v, label,
                    ha="center", va="bottom", fontsize=9)
            if not np.isnan(pd):
                ax.text(b.get_x() + b.get_width() / 2, v * 0.5,
                        f"{pd:.2f} per kUSD", ha="center", va="center",
                        fontsize=9, color="white", weight="bold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(xs)
        ax.set_ylabel(unit)
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", alpha=0.3)

    if raw_best:
        bar(ax_l, raw_best, "TFLOPS or TOPS",
            f"Raw GEMM at {shape_n}³ — best backend per device")
    if mm_best:
        bar(ax_r, mm_best, "G_modmul/s",
            f"Exact 36-bit modmul at {shape_n}³ — best backend per device")
    fig.suptitle(
        f"Headline at {shape_n}³ — RTX 5090 (USD {DEVICE_PRICES_USD['RTX5090']:.0f}) "
        f"vs Blackhole (USD {DEVICE_PRICES_USD['Blackhole']:.0f})"
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layer_c_q36_vs_q48(records: list[dict[str, Any]], out_path: Path) -> None:
    """For each device that has both q36 and q48 records, overlay the lines.

    Skipped if no q48 records exist (NVIDIA doesn't have q48 in v1; this
    plot is meaningful only after both sides have q48 implemented).
    """
    layer_c = [
        r for r in records
        if r["layer"] == "C"
        and r.get("throughput") is not None
        and r["device"] != "CPU"
        and (is_q36_modmul(r["useful_op_kind"]) or is_q48_modmul(r["useful_op_kind"]))
    ]
    if not any(is_q48_modmul(r["useful_op_kind"]) for r in layer_c):
        return  # only meaningful if at least one device has q48 measured

    by_series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for r in layer_c:
        m = r["shape"]["M"]
        kind = "q48" if is_q48_modmul(r["useful_op_kind"]) else "q36"
        # Pick best-throughput backend per (device, kind, M) to match the
        # headline plot's convention.
        key = (r["device"], kind)
        # We aggregate to best later; for now just collect.
        by_series[key].append((m, r["throughput"]))

    # Collapse to best-per-(device, kind, M).
    series_best: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for (device, kind), pts in by_series.items():
        for m, thr in pts:
            cur = series_best[(device, kind)].get(m)
            if cur is None or thr > cur:
                series_best[(device, kind)][m] = thr

    fig, ax = plt.subplots(figsize=(8, 5))
    for (device, kind), m_to_thr in sorted(series_best.items()):
        xs = sorted(m_to_thr.keys())
        ys = [m_to_thr[x] for x in xs]
        bc = "int8"  # both q36 and q48 use INT8 byte decomposition
        ax.plot(
            xs, ys,
            marker=DEVICE_MARKER.get(device, "o"),
            color=_color(device, bc),
            linestyle="-" if kind == "q36" else "--",
            label=f"{device} {kind}",
            linewidth=2.0,
            markersize=7,
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
    ax.set_xlabel("square matmul size N (M=K=N)")
    ax.set_ylabel("effective exact modmul/s (G_modmul/s)")
    ax.set_title("Layer C — q36 vs q48 modular product (best backend per device)")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_layer_d_klss_ip(records: list[dict[str, Any]], out_path: Path) -> None:
    """Layer D KLSS-like inner product. The headline FHE-throughput plot
    once both sides have it implemented; for now usually TT-only."""
    layer_d = [
        r for r in records
        if r["layer"] == "D"
        and r.get("throughput") is not None
        and is_klss_ip(r["useful_op_kind"])
        and r["device"] != "CPU"
    ]
    if not layer_d:
        return

    series_best: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in layer_d:
        m = r["shape"]["M"]
        # Extract q36/q48 from op_kind suffix; default 'q?' if absent.
        op = r["useful_op_kind"]
        if op.endswith("_q36"):
            kind = "q36"
        elif op.endswith("_q48"):
            kind = "q48"
        else:
            kind = "?"
        cur = series_best[(r["device"], kind)].get(m)
        if cur is None or r["throughput"] > cur:
            series_best[(r["device"], kind)][m] = r["throughput"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for (device, kind), m_to_thr in sorted(series_best.items()):
        xs = sorted(m_to_thr.keys())
        ys = [m_to_thr[x] for x in xs]
        ax.plot(
            xs, ys,
            marker=DEVICE_MARKER.get(device, "o"),
            color=_color(device, "int8"),
            linestyle="-" if kind == "q36" else "--",
            label=f"{device} {kind}",
            linewidth=2.2,
            markersize=8,
        )

    nvidia_present = any(r["device"] == "RTX5090" for r in layer_d)
    if not nvidia_present:
        ax.text(
            0.5, 0.02,
            "NVIDIA Layer D not yet implemented — TT side only",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, style="italic", color="#555",
            bbox={"facecolor": "white", "edgecolor": "#aaa", "boxstyle": "round,pad=0.4"},
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([512, 1024, 2048, 4096, 8192])
    ax.set_xticklabels(["512", "1024", "2048", "4096", "8192"])
    ax.set_xlabel("square matmul size N (M=K=N)")
    ax.set_ylabel("effective KLSS-IP useful MAC/s (G_MAC/s)")
    ax.set_title("Layer D — KLSS-like inner product (the FHE workload metric)")
    ax.legend(loc="best", framealpha=0.9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Driver ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render comparison plots from bench JSONL.")
    p.add_argument("inputs", nargs="+", type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--headline-shape", type=int, default=4096,
        help="Square shape size used for the side-by-side bar chart (default 4096).",
    )
    args = p.parse_args(argv)

    apply_style()
    records = load_records(args.inputs)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_layer_b_throughput(records, args.out_dir / "layer_b_throughput.png")
    plot_layer_c_modmul(records, args.out_dir / "layer_c_modmul.png")
    plot_layer_c_per_dollar(records, args.out_dir / "layer_c_modmul_per_dollar.png")
    plot_layer_c_q36_vs_q48(records, args.out_dir / "layer_c_q36_vs_q48.png")
    plot_layer_d_klss_ip(records, args.out_dir / "layer_d_klss_ip.png")
    plot_headline_at_size(
        records, args.headline_shape, args.out_dir / f"headline_{args.headline_shape}.png"
    )

    n_pngs = len(list(args.out_dir.glob("*.png")))
    print(f"wrote {n_pngs} PNG(s) to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
