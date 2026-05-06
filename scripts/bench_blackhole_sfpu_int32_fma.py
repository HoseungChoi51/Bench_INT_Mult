"""Phase 8 — SFPU INT32 fused mul+add benchmark wrapper.

Subprocesses ``tt-llk-skeleton/build_sfpu/bench_sfpu_int32_fma`` once per
``--n-tiles`` value in the sweep, parses the binary's CSV stdout, and
emits one JSONL ``BenchResult`` per cell so the row joins the rest of the
v2 dataset in ``scripts/compare.py``.

Vector size: one tile = 32×32 INT32 = 1024 elements = 4 KiB. The sweep
is on ``n_tiles_per_core``; the total problem size is
``n_tiles_per_core × n_cores × 1024`` INT32 elements (≈4 KiB × n_cores
at 1 tile per core, scaling up).

Useful-ops accounting: 2 SFPU ops per element (one ``mul_int_tile`` +
one ``add_int_tile``). ``useful_ops = 2 × n_tiles_total × 1024``.
``throughput`` is reported in **GOPS** (giga integer ops/sec) — *not*
TFLOPS, since these are int32 ops.

Schema: layer = ``"E"`` (new in this phase: SFPU eltwise capability).
backend = ``tt_sfpu_int32_fma``. useful_op_kind = ``int32_fma_eltwise``.
shape = ``{"M": 1, "K": n_cores, "N": n_tiles_per_core * 1024}`` so the
sweep grid stays distinct in joins.

Pre-requisite: ``cd tt-llk-skeleton && make bench_sfpu_int32_fma``.

CLI::

    uv run python scripts/bench_blackhole_sfpu_int32_fma.py \\
        --out bench-results/blackhole_<sha>_sfpu_int32_fma.jsonl
"""

from __future__ import annotations

import argparse
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

# Vector-size sweep on tiles per core. At 1 tile / core: 1 KiB INT32 per
# core (110 KiB total on a full grid). At 4096 tiles / core: 16 MiB / core
# (1.7 GiB total) — DRAM-resident, far past the 1.5 MiB L1.
DEFAULT_SWEEP = (1, 4, 16, 64, 256, 1024, 4096)
DEFAULT_BIN = (
    _REPO_ROOT / "tt-llk-skeleton" / "build_sfpu" / "bench_sfpu_int32_fma"
)


def _parse_csv_line(line: str) -> dict[str, str]:
    """Parse one stdout CSV row.

    Format::

        median_ms,p10_ms,p90_ms,arch,n_cores,gate,err
    """
    parts = line.rstrip("\n").split(",")
    if len(parts) < 7:
        raise ValueError(f"unexpected CSV row from binary: {line!r}")
    return {
        "median_ms": parts[0],
        "p10_ms": parts[1],
        "p90_ms": parts[2],
        "arch": parts[3],
        "n_cores": parts[4],
        "gate": parts[5],
        "err": ",".join(parts[6:]).strip(),
    }


def _run_one(
    binary: Path,
    n_tiles_per_core: int,
    warmup: int,
    iters: int,
) -> dict[str, str]:
    cmd = [
        str(binary),
        "--n-tiles", str(n_tiles_per_core),
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "TT_LOG_FILE_DEFAULT": "/tmp/tt_sfpu_int32_fma.log"},
    )
    if proc.returncode not in (0,) and not proc.stdout:
        return {
            "median_ms": "null", "p10_ms": "null", "p90_ms": "null",
            "arch": "?", "n_cores": "0",
            "gate": "skipped",
            "err": (proc.stderr.strip().splitlines() or ["binary error"])[-1],
        }
    # tt-metalium can emit a few logger lines on stdout around device
    # close. Find the actual CSV line (first field is a float or "null").
    csv_line = ""
    for line in reversed(proc.stdout.splitlines()):
        s = line.strip()
        if not s:
            continue
        head = s.split(",", 1)[0]
        if head == "null" or _is_float(head):
            csv_line = s
            break
    if not csv_line:
        return {
            "median_ms": "null", "p10_ms": "null", "p90_ms": "null",
            "arch": "?", "n_cores": "0",
            "gate": "skipped",
            "err": "no CSV row in stdout",
        }
    return _parse_csv_line(csv_line)


def _is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def _to_record(
    csv_row: dict[str, str],
    n_tiles_per_core: int,
    warmup: int,
    iters: int,
    sha: str,
    ts: str,
    power_w_avg: float | None,
) -> BenchResult:
    n_cores = int(csv_row["n_cores"]) if csv_row["n_cores"].isdigit() else 0
    n_tiles_total = n_tiles_per_core * n_cores
    elements = n_tiles_total * 1024
    # 1 mul + 1 add per element.
    useful_ops = 2 * elements

    def _f(s: str) -> float | None:
        return None if s == "null" else float(s)

    median_ms = _f(csv_row["median_ms"])
    p10_ms = _f(csv_row["p10_ms"])
    p90_ms = _f(csv_row["p90_ms"])

    throughput: float | None = None
    if median_ms is not None and median_ms > 0 and useful_ops > 0:
        # GOPS = (ops / sec) / 1e9 ; sec = median_ms / 1000
        throughput = useful_ops / (median_ms * 1e-3) / 1e9

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
            "harness": "tt-llk-skeleton/host_sfpu/main.cpp",
            "kernel": "tt-llk-skeleton/kernels/compute_sfpu_int32_fma.cpp",
            "n_cores": n_cores,
            "n_tiles_per_core": n_tiles_per_core,
            "n_tiles_total": n_tiles_total,
            "tile_bytes": 4096,
            "elements_per_tile": 1024,
            "ops_per_element": 2,
            "math_fidelity": "MathFidelity.HiFi4",
            "fp32_dest_acc_en": True,
            "note": (
                "SFPU eltwise int32 fused mul+add (a*b+c). No NVIDIA "
                "counterpart in the v2 dataset; the row is TT-only by "
                "design (the matrix engine has no INT32 surface)."
            ),
            "gate": csv_row.get("gate", "skipped"),
            "error": csv_row.get("err") or None,
        },
        layer="E",
        backend="tt_sfpu_int32_fma",
        shape={"M": 1, "K": n_cores, "N": n_tiles_per_core * 1024},
        dtype_in="int32",
        dtype_acc="int32",
        iters=iters,
        warmup=warmup,
        median_ms=median_ms,
        p10_ms=p10_ms,
        p90_ms=p90_ms,
        useful_ops=useful_ops,
        useful_op_kind="int32_fma_eltwise",
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
    p.add_argument(
        "--bin", type=Path, default=DEFAULT_BIN,
        help=f"Path to bench_sfpu_int32_fma (default: {DEFAULT_BIN}).",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="JSONL output path.",
    )
    p.add_argument(
        "--sweep", type=str, default=",".join(str(x) for x in DEFAULT_SWEEP),
        help=(
            "Comma-separated list of n_tiles_per_core to test "
            f"(default: {','.join(str(x) for x in DEFAULT_SWEEP)})."
        ),
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args(argv)

    if not args.bin.is_file():
        print(
            f"binary not found at {args.bin}. Build with: "
            "cd tt-llk-skeleton && make bench_sfpu_int32_fma",
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
