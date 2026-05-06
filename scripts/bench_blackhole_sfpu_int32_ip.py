"""Phase 8 (extension) — SFPU INT32 inner-product benchmark wrapper.

On-device:

    per_core_acc[ℓ]  =  Σ_{t<n_tiles}  A[t][ℓ] · B[t][ℓ]      ℓ ∈ [0, 1024)

over `n_cores` cores in parallel, each handling a disjoint slice of A/B.
The SFPU's int32 mul + add primitives carry a per-tile accumulator
through a small intermediate L1 CB; one accumulator tile per core is
written back to DRAM at the end.

Overflow: planned ahead in the host binary (see safe_input_bound() in
host_sfpu_ip/main.cpp). We bound  |a|, |b| ≤ ⌊√(2³⁰ / n_tiles)⌋  so the
per-lane partial sum N·B² stays below 2³⁰ — there is no post-hoc check.

Useful-ops counting (per core, per call):
    1 mul + 1 add per element per accumulating step, but the first tile
    has only the mul (no carry to add).
    ⇒  useful_ops_per_core  =  (2·n_tiles − 1) · 1024
    ⇒  useful_ops_total      =  ((2·n_tiles − 1) · 1024) · n_cores

`throughput` is reported in **GOPS** (giga integer ops/sec). One run
emits one record per `n_tiles_per_core` value in the sweep.

Pre-requisite: ``cd tt-llk-skeleton && make bench_sfpu_int32_inner_product``.

CLI::

    uv run python scripts/bench_blackhole_sfpu_int32_ip.py \\
        --out bench-results/blackhole_<sha>_sfpu_int32_ip.jsonl
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bench_common import (  # noqa: E402
    SCHEMA_VERSION,
    BenchResult,
    git_sha,
    now_iso,
    read_tt_power_w,
    write_results,
)

DEFAULT_SWEEP = (1, 4, 16, 64, 256, 1024, 4096)
DEFAULT_BIN = (
    _REPO_ROOT
    / "tt-llk-skeleton"
    / "build_sfpu_ip"
    / "bench_sfpu_int32_inner_product"
)


def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def _parse_csv_line(line: str) -> dict[str, str]:
    parts = line.rstrip("\n").split(",")
    if len(parts) < 7:
        raise ValueError(f"unexpected CSV row from binary: {line!r}")
    return {
        "median_ms": parts[0], "p10_ms": parts[1], "p90_ms": parts[2],
        "arch": parts[3], "n_cores": parts[4],
        "gate": parts[5], "err": ",".join(parts[6:]).strip(),
    }


def _run_one(binary: Path, n_tiles: int, warmup: int, iters: int) -> dict[str, str]:
    proc = subprocess.run(
        [str(binary), "--n-tiles", str(n_tiles),
         "--warmup", str(warmup), "--iters", str(iters)],
        capture_output=True, text=True, check=False,
        env={**os.environ, "TT_LOG_FILE_DEFAULT": "/tmp/tt_sfpu_int32_ip.log"},
    )
    if proc.returncode != 0 and not proc.stdout:
        return {
            "median_ms": "null", "p10_ms": "null", "p90_ms": "null",
            "arch": "?", "n_cores": "0", "gate": "skipped",
            "err": (proc.stderr.strip().splitlines() or ["binary error"])[-1],
        }
    csv_line = ""
    for line in reversed(proc.stdout.splitlines()):
        s = line.strip()
        if not s: continue
        head = s.split(",", 1)[0]
        if head == "null" or _is_float(head):
            csv_line = s; break
    if not csv_line:
        return {
            "median_ms": "null", "p10_ms": "null", "p90_ms": "null",
            "arch": "?", "n_cores": "0", "gate": "skipped",
            "err": "no CSV row in stdout",
        }
    return _parse_csv_line(csv_line)


def _safe_input_bound(n_tiles: int) -> int:
    """Mirrors host_sfpu_ip/main.cpp::safe_input_bound. Reported in stderr."""
    return max(1, int(math.floor(math.sqrt((1 << 30) / float(n_tiles)))))


def _to_record(
    csv_row: dict[str, str], n_tiles: int, warmup: int, iters: int,
    sha: str, ts: str, power_w_avg: float | None,
) -> BenchResult:
    n_cores = int(csv_row["n_cores"]) if csv_row["n_cores"].isdigit() else 0

    # First tile contributes one mul only; tiles 1..N-1 each contribute one
    # mul + one add. That is (2·N − 1) ops per lane, × 1024 lanes per core,
    # × n_cores cores in parallel.
    useful_ops_per_core = max(0, 2 * n_tiles - 1) * 1024
    useful_ops = useful_ops_per_core * n_cores

    def _f(s: str) -> float | None:
        return None if s == "null" else float(s)

    median_ms = _f(csv_row["median_ms"])
    p10_ms = _f(csv_row["p10_ms"])
    p90_ms = _f(csv_row["p90_ms"])

    throughput: float | None = None
    if median_ms is not None and median_ms > 0 and useful_ops > 0:
        throughput = useful_ops / (median_ms * 1e-3) / 1e9  # GOPS

    joules_per_op: float | None = None
    if (
        power_w_avg is not None
        and median_ms is not None
        and useful_ops > 0
    ):
        joules_per_op = power_w_avg * (median_ms * 1e-3) / float(useful_ops)

    return BenchResult(
        schema_version=SCHEMA_VERSION,
        device="Blackhole",
        device_detail={
            "name": "Tenstorrent Blackhole p150a",
            "harness": "tt-llk-skeleton/host_sfpu_ip/main.cpp",
            "kernel": "tt-llk-skeleton/kernels/compute_sfpu_int32_inner_product.cpp",
            "n_cores": n_cores,
            "n_tiles_per_core": n_tiles,
            "n_tiles_total": n_tiles * n_cores,
            "tile_bytes": 4096,
            "elements_per_tile": 1024,
            "ops_per_element_per_step": 2,  # 1 mul + 1 add (except first tile)
            "math_fidelity": "MathFidelity.HiFi4",
            "fp32_dest_acc_en": True,
            "input_bound_abs": _safe_input_bound(n_tiles),
            "input_bound_rationale": (
                "|a|,|b| ≤ ⌊√(2³⁰ / n_tiles)⌋ ⇒ per-lane partial sum "
                "n_tiles · B² ≤ 2³⁰ < 2³¹ (no post-hoc overflow check)."
            ),
            "note": (
                "SFPU per-lane inner product: each tile contributes one "
                "mul + one add (first tile only mul). 1024 lanes per "
                "core; the across-lane and across-core final reduction "
                "is not part of the timed device loop."
            ),
            "gate": csv_row.get("gate", "skipped"),
            "error": csv_row.get("err") or None,
        },
        layer="E",
        backend="tt_sfpu_int32_inner_product",
        # shape: M=1, K=n_cores · n_tiles · 1024 (the inner-dim length
        # summed across the device), N=1 (scalar-per-lane output).
        shape={
            "M": 1,
            "K": n_cores * n_tiles * 1024,
            "N": 1,
        },
        dtype_in="int32",
        dtype_acc="int32",
        iters=iters,
        warmup=warmup,
        median_ms=median_ms,
        p10_ms=p10_ms,
        p90_ms=p90_ms,
        useful_ops=useful_ops,
        useful_op_kind="int32_inner_product",
        throughput=throughput,
        throughput_unit="GOPS",
        correctness={"gate": csv_row.get("gate", "skipped")},
        host_overhead_ms=None,
        git_sha=sha,
        timestamp=ts,
        power_w_avg=power_w_avg,
        joules_per_useful_op=joules_per_op,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN,
                   help=f"Path to bench_sfpu_int32_inner_product (default: {DEFAULT_BIN}).")
    p.add_argument("--out", required=True, type=Path, help="JSONL output path.")
    p.add_argument(
        "--sweep", type=str, default=",".join(str(x) for x in DEFAULT_SWEEP),
        help=f"Comma-separated n_tiles_per_core list (default: {','.join(str(x) for x in DEFAULT_SWEEP)}).",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args(argv)

    if not args.bin.is_file():
        print(
            f"binary not found at {args.bin}. Build with: "
            "cd tt-llk-skeleton && make bench_sfpu_int32_inner_product",
            file=sys.stderr,
        )
        return 2

    sweep = tuple(int(x) for x in args.sweep.split(",") if x.strip())
    if not sweep:
        print("--sweep produced an empty list", file=sys.stderr)
        return 3

    sha, ts = git_sha(), now_iso()
    records: list[BenchResult] = []
    for n_tiles in sweep:
        bound = _safe_input_bound(n_tiles)
        print(
            f"  [plan] n_tiles={n_tiles:>5} → input bound ±{bound:>6} "
            f"(per-lane sum ≤ {n_tiles * bound * bound:.3e})",
            file=sys.stderr,
        )
        power_pre = read_tt_power_w()
        csv_row = _run_one(args.bin, n_tiles, args.warmup, args.iters)
        power_post = read_tt_power_w()
        power_w_avg = (
            (power_pre + power_post) / 2.0
            if power_pre is not None and power_post is not None else None
        )
        rec = _to_record(csv_row, n_tiles, args.warmup, args.iters,
                         sha, ts, power_w_avg)
        records.append(rec)
        gops = (
            f"{rec.throughput:.2f} GOPS"
            if rec.throughput is not None else "skipped"
        )
        gate = rec.correctness.get("gate", "?")
        print(
            f"  n_tiles={n_tiles:>5} → "
            f"median={rec.median_ms} ms, throughput={gops}, gate={gate}",
            file=sys.stderr,
        )

    write_results(records, args.out)
    print(f"wrote {len(records)} record(s) → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
