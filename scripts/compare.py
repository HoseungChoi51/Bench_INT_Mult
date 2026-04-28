"""Merge bench JSONL outputs from multiple devices into a comparison table.

Reads any number of ``*.jsonl`` files emitted by ``bench_nvidia.py`` (and,
later, ``bench_blackhole.py``), groups records by ``(layer, useful_op_kind,
shape, backend_class)``, and writes a markdown table to
``bench-results/SUMMARY.md`` (or wherever ``--out`` points). For each
joined row, computes a **price-adjusted ratio** using the device prices in
:data:`scripts._bench_common.DEVICE_PRICES_USD`.

The merge is permissive: missing rows on one side leave the corresponding
column blank rather than failing. This lets us generate an interim
summary while the TT side hasn't yet produced records.

The "backend class" mapping below collapses device-specific backend names
into the cross-device category that's actually comparable. For example:

- ``cublaslt_int8`` and ``tt_llk_int8`` both report ``int8`` work
- ``cublaslt_tf32`` and ``tt_llk_tf32`` (if added) both report ``tf32``
- ``cpu_int128`` lives in its own ``cpu`` class

Usage::

    python scripts/compare.py bench-results/*.jsonl --out bench-results/SUMMARY.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bench_common import DEVICE_PRICES_USD  # noqa: E402

BACKEND_CLASS = {
    "cublaslt_int8": "int8",
    "tt_llk_int8": "int8",
    "cublaslt_tf32": "tf32",
    "tt_llk_tf32": "tf32",
    # Tensix matrix engine for FP32 inputs uses TF32-internal fidelity
    # (per Tenstorrent's fp32_accuracy doc), the same precision NVIDIA's
    # Tensor Core uses for FP32. Map it to the same backend class so the
    # comparison row joins cublaslt_tf32.
    "tt_llk_fp32_matrix": "tf32",
    "cublaslt_bf16": "bf16",
    "tt_llk_bf16": "bf16",
    "cublaslt_fp32": "fp32",
    # SFPU vector FP32 path: joins cuBLAS fp32 since both deliver true
    # IEEE-754 FP32 (the matrix path above is TF32-internal).
    "tt_llk_sfpu_fp32": "fp32",
    "cublaslt_fp64": "fp64",
    "cpu_int128": "cpu",
    # Phase 7 — upstream tt-metal GEMM_FLOPS reproduction (high-precision
    # candidates only; BF8/BF4 are sanity-only). Each row gets its own
    # backend_class so the tuned numbers don't collide with our reference
    # Layer B rows (`tt_llk_bf16`, `tt_llk_fp32_matrix`, etc.) — readers
    # can see the gap between reference and tuned at a glance.
    "tt_matmul_2d_bf16_hifi4":          "bf16_tuned_hifi4",
    "tt_matmul_2d_bf16_hifi4_traced":   "bf16_tuned_hifi4",
    "tt_matmul_2d_bf16_hifi2":          "bf16_tuned_hifi2",
    "tt_matmul_2d_bf16_hifi2_traced":   "bf16_tuned_hifi2",
    # Tuned FP32-matrix path joins NVIDIA cublaslt_fp32 (CUDA core, true
    # IEEE FP32) since both are end-user "FP32" surfaces — but keep in
    # mind TT's matrix engine internally truncates FP32 inputs to TF32.
    "tt_matmul_2d_fp32_hifi4":          "fp32_tuned",
    "tt_matmul_2d_fp32_hifi4_traced":   "fp32_tuned",
    "tt_matmul_2d_bf8_hifi2":           "bf8_sanity",
    "tt_matmul_2d_bf8_hifi2_traced":    "bf8_sanity",
    "tt_matmul_2d_bf4_lofi":            "bf4_sanity",
    "tt_matmul_2d_bf4_lofi_traced":     "bf4_sanity",
    # Phase 7 (INT8 extension) — block-tiled INT8 matmul (no mcast).
    # Drops the `llk` prefix to match the rest of the tuned family;
    # joins NVIDIA cublaslt_int8 in the int8 class so the row sits
    # alongside the v1 reference `tt_llk_int8` and the cuBLASLt INT8
    # reference. See doc/04_phase7_tuned_matmul.md §4.7.
    "tt_matmul_2d_int8_hifi4":          "int8",
    "tt_matmul_2d_int8_hifi2":          "int8",
    "tt_matmul_2d_int8_lofi":           "int8",
    # Phase 8 — SFPU INT32 fused mul+add. No NVIDIA counterpart in the
    # current dataset (the matrix engine has no INT32 surface, and we
    # have not yet wired a CUDA-core int32 FMA row); the comparison row
    # is TT-only by design.
    "tt_sfpu_int32_fma":                "int32_fma_eltwise",
    # Phase 8 (extension) — SFPU INT32 inner product, per-lane partial
    # sum staying in SFPU registers. Same TT-only join; input bounds are
    # planned per-shape so the per-lane sum stays in INT31.
    "tt_sfpu_int32_inner_product":      "int32_inner_product",
}


def load_jsonl(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


def shape_key(shape: dict[str, int]) -> str:
    return f"{shape['M']}×{shape['K']}×{shape['N']}"


def backend_class(backend: str) -> str:
    return BACKEND_CLASS.get(backend, backend)


# Legacy useful_op_kind values from the v1 schema. Pre-Phase-2 records used
# "exact_modmul" without a prime suffix; post-Phase-2 records use
# "exact_modmul_q36" / "exact_modmul_q48". Treat the legacy name as q36 so
# committed NVIDIA q36 JSONL records still join Blackhole q36 records.
USEFUL_OP_KIND_ALIASES = {
    "exact_modmul": "exact_modmul_q36",
}


def normalize_op_kind(kind: str) -> str:
    return USEFUL_OP_KIND_ALIASES.get(kind, kind)


def _fmt_thr(r: dict[str, Any]) -> str:
    if r.get("throughput") is None:
        return "—"
    unit = r.get("throughput_unit", "")
    return f"{r['throughput']:.2f} {unit}"


def _fmt_per_dollar(r: dict[str, Any]) -> str:
    thr = r.get("throughput")
    if thr is None:
        return "—"
    price = DEVICE_PRICES_USD.get(r.get("device", ""), 0.0)
    if price <= 0:
        return "—"
    return f"{thr / price * 1000:.3f} /k$"


def _shape_sort_key(s: str) -> tuple[int, int, int]:
    """Sort '512×512×512' before '1024×1024×1024' (numeric, not lexical)."""
    try:
        m, k, n = (int(x) for x in s.replace("x", "×").split("×"))
        return (m, k, n)
    except ValueError:
        return (0, 0, 0)


def _select_headline_devices(devices: list[str]) -> tuple[str, str] | None:
    """Pick the two GPU devices for the cross-device ratio. Skip CPU.

    Preferred ordering: RTX5090 first (it's the asking-price baseline),
    Blackhole second (the price-challenger). Fall back to any two non-CPU
    devices if the canonical pair isn't present.
    """
    non_cpu = [d for d in devices if d != "CPU"]
    if "RTX5090" in non_cpu and "Blackhole" in non_cpu:
        return "RTX5090", "Blackhole"
    if len(non_cpu) >= 2:
        return non_cpu[0], non_cpu[1]
    return None


def _fmt_power(r: dict[str, Any]) -> str:
    p = r.get("power_w_avg")
    return f"{p:.1f} W" if p is not None else "—"


def _fmt_per_joule(r: dict[str, Any]) -> str:
    """Throughput per watt (TFLOPS/W or G_modmul/s/W).

    Computed from throughput and power_w_avg directly so the unit on the
    label matches the throughput unit.
    """
    thr = r.get("throughput")
    p = r.get("power_w_avg")
    unit = r.get("throughput_unit", "")
    if thr is None or p is None or p <= 0:
        return "—"
    return f"{thr / p:.3f} {unit}/W"


def render_layer(records: list[dict[str, Any]], layer: str) -> str:
    """One markdown table per layer, joined on (op_kind, shape, backend_class)."""
    layer_records = [r for r in records if r.get("layer") == layer]
    if not layer_records:
        return ""

    devices = sorted({r["device"] for r in layer_records})
    rows: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in layer_records:
        key = (
            normalize_op_kind(r["useful_op_kind"]),
            shape_key(r["shape"]),
            backend_class(r["backend"]),
        )
        rows[key][r["device"]] = r

    # Add power columns only when at least one record in the layer carries
    # power_w_avg. Keeps the v1-style table unchanged when no telemetry was
    # captured.
    has_power = any(r.get("power_w_avg") is not None for r in layer_records)

    out: list[str] = []
    out.append(f"### Layer {layer}")
    out.append("")
    header = ["op_kind", "shape", "backend"]
    for d in devices:
        header.append(f"{d} thr")
        header.append(f"{d} /k$")
        if has_power:
            header.append(f"{d} W")
            header.append(f"{d} thr/W")
        header.append(f"{d} gate")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")

    sorted_keys = sorted(rows.keys(), key=lambda k: (k[0], _shape_sort_key(k[1]), k[2]))
    for key in sorted_keys:
        op_kind, shape, bclass = key
        row = [op_kind, shape, bclass]
        for d in devices:
            r = rows[key].get(d)
            if r is None:
                row += ["—", "—"]
                if has_power:
                    row += ["—", "—"]
                row += ["—"]
            else:
                row += [_fmt_thr(r), _fmt_per_dollar(r)]
                if has_power:
                    row += [_fmt_power(r), _fmt_per_joule(r)]
                row += [r["correctness"].get("gate", "—")]
        out.append("| " + " | ".join(row) + " |")

    pair = _select_headline_devices(devices)
    if pair is not None:
        out.append("")
        out.append(_render_ratio_section(layer_records, pair))

    out.append("")
    return "\n".join(out)


def _render_ratio_section(
    layer_records: list[dict[str, Any]], pair: tuple[str, str]
) -> str:
    """Render two cross-device ratio tables for the given GPU pair.

    1. **Best-of-device** (always shown): one row per (op_kind, shape),
       picking the best-throughput backend per device. Answers "which
       device delivers the most throughput at this shape, at any
       precision the device supports?" — the right headline when not
       every backend is implemented on every device yet (e.g., TT-LLK
       INT8 is still TODO in v1).

    2. **Per-matching-backend** (shown only if any matches exist): joins
       on exact backend_class, like-for-like precision. Useful once both
       sides have the same backend(s) implemented.

    Note: backend_class collapses device-specific names (cublaslt_int8,
    tt_llk_int8 → "int8"). FP64 is matched only against another FP64,
    BF16 against BF16, and so on; cross-precision comparisons stay in
    table 1.
    """
    d0, d1 = pair
    p0 = DEVICE_PRICES_USD.get(d0, 0.0)
    p1 = DEVICE_PRICES_USD.get(d1, 0.0)

    # Group records by (op_kind, shape) device-side, dropping nulls.
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in layer_records:
        if r["device"] not in pair:
            continue
        if r.get("throughput") is None:
            continue
        key = (r["useful_op_kind"], shape_key(r["shape"]))
        grouped[key][r["device"]].append(r)

    sorted_keys = sorted(grouped.keys(), key=lambda k: (k[0], _shape_sort_key(k[1])))

    out: list[str] = []
    out.append(f"**Headline ratios — {d0} vs {d1}, best backend per device** "
               f"(higher = {d0} wins):")
    out.append("")
    out.append(
        f"| op_kind | shape | {d0} best | {d1} best | {d0} / {d1} "
        f"| price-adjusted ({d0} /k$) / ({d1} /k$) |"
    )
    out.append("|---|---|---|---|---|---|")
    any_headline = False
    for (op_kind, shape) in sorted_keys:
        by_dev = grouped[(op_kind, shape)]
        recs0 = by_dev.get(d0, [])
        recs1 = by_dev.get(d1, [])
        if not recs0 or not recs1:
            continue
        r0 = max(recs0, key=lambda r: r["throughput"])
        r1 = max(recs1, key=lambda r: r["throughput"])
        ratio = r0["throughput"] / r1["throughput"]
        adj = "—"
        if p0 > 0 and p1 > 0:
            adj = f"{(r0['throughput'] / p0) / (r1['throughput'] / p1):.2f}×"
        bc0 = backend_class(r0["backend"])
        bc1 = backend_class(r1["backend"])
        out.append(
            f"| {op_kind} | {shape} | {bc0} ({_fmt_thr(r0)}) "
            f"| {bc1} ({_fmt_thr(r1)}) | {ratio:.2f}× | {adj} |"
        )
        any_headline = True
    if not any_headline:
        out.append(f"| _no overlapping records between {d0} and {d1}_ | | | | | |")

    # Per-backend-class table (only emitted if there is at least one match).
    matched_rows: list[str] = []
    for (op_kind, shape) in sorted_keys:
        by_dev = grouped[(op_kind, shape)]
        recs0 = by_dev.get(d0, [])
        recs1 = by_dev.get(d1, [])
        if not recs0 or not recs1:
            continue
        by_bc0 = {backend_class(r["backend"]): r for r in recs0}
        by_bc1 = {backend_class(r["backend"]): r for r in recs1}
        for bc in sorted(set(by_bc0) & set(by_bc1)):
            r0 = by_bc0[bc]
            r1 = by_bc1[bc]
            ratio = r0["throughput"] / r1["throughput"]
            adj = "—"
            if p0 > 0 and p1 > 0:
                adj = f"{(r0['throughput'] / p0) / (r1['throughput'] / p1):.2f}×"
            matched_rows.append(
                f"| {op_kind} | {shape} | {bc} | {ratio:.2f}× | {adj} |"
            )

    if matched_rows:
        out.append("")
        out.append(f"**Per-matching-backend ratios — {d0} vs {d1}** "
                   "(only shown where both devices have the backend implemented):")
        out.append("")
        out.append(
            f"| op_kind | shape | backend | {d0} / {d1} "
            f"| price-adjusted ({d0} /k$) / ({d1} /k$) |"
        )
        out.append("|---|---|---|---|---|")
        out.extend(matched_rows)

    return "\n".join(out)


def _render_plots_section() -> list[str]:
    """Reference the PNGs `scripts/plot_summary.py` writes to the same dir.

    Each link is wrapped in a check that gracefully degrades if the PNG
    isn't there (rendered to plain text — GitHub markdown handles this).
    The expected output directory is the same one this script writes to.
    """
    plots = [
        ("Layer D — KLSS-like inner product",
         "layer_d_klss_ip.png",
         "The actual FHE workload metric (useful MAC/s per second). Where TT KLSS results live."),
        ("Layer C — exact 36-bit modmul throughput",
         "layer_c_modmul.png",
         "Best backend per device. q36 only; q48 is in a separate plot."),
        ("Layer C — exact 36-bit modmul per dollar",
         "layer_c_modmul_per_dollar.png",
         "Same data, normalized by device MSRP. Answers the price-ratio question."),
        ("Layer C — q36 vs q48 modular product",
         "layer_c_q36_vs_q48.png",
         "Solid lines = q36, dashed = q48. Shows the cost of going to a 48-bit prime."),
        ("Layer B — raw GEMM throughput",
         "layer_b_throughput.png",
         "Per (device, backend), log-y. Shows the full precision-vs-throughput envelope."),
        ("Headline at 4096³",
         "headline_4096.png",
         "Side-by-side bar chart at one representative shape."),
    ]
    out: list[str] = []
    out.append("## Plots")
    out.append("")
    out.append(
        "Generated by `scripts/plot_summary.py` from the same JSONL "
        "files as this summary. Re-run after each bench refresh."
    )
    out.append("")
    for title, fname, caption in plots:
        out.append(f"### {title}")
        out.append("")
        out.append(f"![{title}]({fname})")
        out.append("")
        out.append(f"_{caption}_")
        out.append("")
    return out


def render_summary(records: list[dict[str, Any]]) -> str:
    if not records:
        return "# Bench summary\n\n_no records found_\n"

    devices = sorted({r["device"] for r in records})
    timestamps = sorted({r.get("timestamp", "") for r in records})
    git_shas = sorted({r.get("git_sha", "") for r in records})
    schema_versions = sorted({r.get("schema_version", "") for r in records})
    # v1↔v2 mixing is allowed: v2 added optional power_w_avg / joules_per_useful_op
    # fields with default None, so v1 records load cleanly. Anything beyond
    # {"1", "2"} signals an unsupported combination.
    unsupported = set(schema_versions) - {"1", "2"}
    if unsupported:
        raise RuntimeError(
            f"records have unsupported schema_versions: {sorted(unsupported)}"
        )

    lines: list[str] = []
    lines.append("# Bench summary — RTX 5090 vs TT Blackhole")
    lines.append("")
    lines.append(f"- **Schema versions present**: {', '.join(schema_versions)}")
    if len(schema_versions) > 1:
        lines.append(
            "  _Mixed schemas merged. v2 is a strict superset of v1 "
            "(adds Layer D, q36/q48 op_kind variants, energy fields); v1 records "
            "have v2-only fields rendered as `—`._"
        )
    lines.append(f"- **Devices**: {', '.join(devices)}")
    lines.append(f"- **Records**: {len(records)}")
    lines.append(f"- **Timestamps**: {timestamps[0]} – {timestamps[-1]}")
    lines.append(f"- **Git SHAs**: {', '.join(git_shas)}")
    lines.append("")

    prices = {d: DEVICE_PRICES_USD.get(d, 0.0) for d in devices}
    lines.append("## Device prices (declared, not detected)")
    lines.append("")
    lines.append("| device | price (USD) |")
    lines.append("|---|---|")
    for d in devices:
        p = prices[d]
        lines.append(f"| {d} | {p:.0f} |" if p > 0 else f"| {d} | (n/a) |")
    lines.append("")
    lines.append(
        "Update :data:`DEVICE_PRICES_USD` in `scripts/_bench_common.py` if these are stale."
    )
    lines.append("")

    lines.extend(_render_plots_section())

    lines.append("## Layer A — capability probe")
    lines.append(render_layer(records, "A") or "_no Layer A records_\n")
    lines.append("## Layer B — raw GEMM")
    lines.append(render_layer(records, "B") or "_no Layer B records_\n")
    lines.append("## Layer C — exact modular product (q36, q48)")
    lines.append(render_layer(records, "C") or "_no Layer C records_\n")
    lines.append("## Layer D — KLSS-style inner product (useful MAC view)")
    lines.append(render_layer(records, "D") or "_no Layer D records_\n")
    lines.append("## Layer E — SFPU INT32 microbench (Phase 8)")
    lines.append(render_layer(records, "E") or "_no Layer E records_\n")

    lines.append("## How to update")
    lines.append("")
    lines.append("```")
    lines.append("# NVIDIA host (this machine):")
    lines.append("uv run --extra bench python scripts/bench_nvidia.py \\")
    lines.append("    --out bench-results/nvidia_$(git rev-parse --short HEAD).jsonl")
    lines.append("")
    lines.append("# TT host (separate machine, see BENCHMARK_TT.md):")
    lines.append("python tt-llk-skeleton/bench_blackhole.py \\")
    lines.append("    --out bench-results/blackhole_$(git rev-parse --short HEAD).jsonl")
    lines.append("")
    lines.append("# Then back on this machine:")
    lines.append("python scripts/compare.py bench-results/*.jsonl --out bench-results/SUMMARY.md")
    lines.append("python scripts/plot_summary.py bench-results/*.jsonl --out-dir bench-results/")
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Merge bench JSONL into a markdown summary.")
    p.add_argument("inputs", nargs="*", type=Path, help="JSONL input files (globs OK).")
    p.add_argument("--out", type=Path, required=True, help="Markdown output path.")
    args = p.parse_args(argv)

    paths: list[Path] = []
    for inp in args.inputs:
        if "*" in str(inp):
            paths.extend(sorted(Path().glob(str(inp))))
        else:
            paths.append(inp)
    paths = [p for p in paths if p.is_file()]

    records = list(load_jsonl(paths))
    summary = render_summary(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(summary, encoding="utf-8")
    print(f"merged {len(records)} record(s) from {len(paths)} file(s) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
