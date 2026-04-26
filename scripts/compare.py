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
    "tt_llk_sfpu_fp32": "fp32",
    "cublaslt_fp64": "fp64",
    "cpu_int128": "cpu",
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

    out: list[str] = []
    out.append(f"### Layer {layer}")
    out.append("")
    header = ["op_kind", "shape", "backend"]
    for d in devices:
        header.append(f"{d} thr")
        header.append(f"{d} /k$")
        header.append(f"{d} gate")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")

    for key in sorted(rows.keys()):
        op_kind, shape, bclass = key
        row = [op_kind, shape, bclass]
        for d in devices:
            r = rows[key].get(d)
            if r is None:
                row += ["—", "—", "—"]
            else:
                row += [_fmt_thr(r), _fmt_per_dollar(r), r["correctness"].get("gate", "—")]
        out.append("| " + " | ".join(row) + " |")

    # Cross-device throughput ratio per row, only meaningful when ≥2 devices match.
    if len(devices) >= 2:
        out.append("")
        out.append("**Cross-device throughput ratios** (higher = first device wins):")
        out.append("")
        d0, d1 = devices[0], devices[1]
        out.append(
            f"| op_kind | shape | backend | {d0} / {d1} "
            f"| price-adjusted ({d0} /k$) / ({d1} /k$) |"
        )
        out.append("|---|---|---|---|---|")
        for key in sorted(rows.keys()):
            op_kind, shape, bclass = key
            r0 = rows[key].get(d0)
            r1 = rows[key].get(d1)
            t0 = r0.get("throughput") if r0 else None
            t1 = r1.get("throughput") if r1 else None
            if t0 is None or t1 is None:
                continue
            ratio = r0["throughput"] / r1["throughput"]
            p0 = DEVICE_PRICES_USD.get(d0, 0.0)
            p1 = DEVICE_PRICES_USD.get(d1, 0.0)
            adj_str = "—"
            if p0 > 0 and p1 > 0:
                adj = (r0["throughput"] / p0) / (r1["throughput"] / p1)
                adj_str = f"{adj:.2f}×"
            out.append(f"| {op_kind} | {shape} | {bclass} | {ratio:.2f}× | {adj_str} |")

    out.append("")
    return "\n".join(out)


def render_summary(records: list[dict[str, Any]]) -> str:
    if not records:
        return "# Bench summary\n\n_no records found_\n"

    devices = sorted({r["device"] for r in records})
    timestamps = sorted({r.get("timestamp", "") for r in records})
    git_shas = sorted({r.get("git_sha", "") for r in records})
    schema_versions = sorted({r.get("schema_version", "") for r in records})
    if len(schema_versions) > 1:
        raise RuntimeError(f"records have mismatched schema_versions: {schema_versions}")

    lines: list[str] = []
    lines.append("# Bench summary — RTX 5090 vs TT Blackhole")
    lines.append("")
    lines.append(f"- **Schema version**: {schema_versions[0]}")
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

    lines.append("## Layer A — capability probe")
    lines.append(render_layer(records, "A") or "_no Layer A records_\n")
    lines.append("## Layer B — raw GEMM")
    lines.append(render_layer(records, "B") or "_no Layer B records_\n")
    lines.append("## Layer C — exact 36-bit modular product")
    lines.append(render_layer(records, "C") or "_no Layer C records_\n")

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
