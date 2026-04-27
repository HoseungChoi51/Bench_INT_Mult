"""Phase 7 — Reproduce TT-Metal's published GEMM_FLOPS numbers on Blackhole p150.

Two responsibilities:

1. **Reproduction wrapper.** Subprocess the upstream pytest at
   `$TT_METAL_HOME/tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf`
   with our small `tt_metal_extras/upstream_runner_conftest.py` plugin to
   strip the upstream `@pytest.mark.skip(...)` decorator. The pytest writes
   `$TT_METAL_HOME/generated/matmul_2d_host_perf_report.csv` (the file the
   GEMM_FLOPS report uses).

2. **CSV → JSONL converter.** Parse that CSV and emit JSONL records using
   the same `BenchResult` schema as the rest of the v2 bench so
   `scripts/compare.py` joins them alongside our existing Layer A/B/C/D
   numbers. Apply the user's row filter (BF16/HiFi4, BF16/HiFi2, FP32/HiFi4
   in full; one BF8/HiFi2 + one BF4/LoFi sanity row each).

Backend naming: ``tt_matmul_2d_<dtype>_<fidelity>[_traced]`` — drops the
`llk` prefix because TT-LLK is frozen and merged into tt-metal, and the
upstream benchmark uses the ttnn high-level API only (no LLK calls).

Usage::

    # one-shot: run + convert
    python scripts/bench_blackhole_ttnn_ref.py \
        --tt-metal-home /home/chs/TT/tt-metal \
        --out bench-results/blackhole_$(git rev-parse --short HEAD)_ttnn_ref.jsonl

    # convert an existing CSV without re-running pytest
    python scripts/bench_blackhole_ttnn_ref.py \
        --csv /home/chs/TT/tt-metal/generated/matmul_2d_host_perf_report.csv \
        --no-run \
        --out bench-results/blackhole_<sha>_ttnn_ref.jsonl
"""

from __future__ import annotations

import argparse
import csv
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

# Filter config per user direction (chat 2026-04-28): focus on
# high-precision INT-emulation candidates; one sanity sample each for the
# lower-precision rows just to confirm the script reaches them.
KEEP_FULL = {
    ("DataType.BFLOAT16", "MathFidelity.HiFi4"),
    ("DataType.BFLOAT16", "MathFidelity.HiFi2"),
    ("DataType.FLOAT32", "MathFidelity.HiFi4"),
}
SANITY_FIRST = {
    ("DataType.BFLOAT8_B", "MathFidelity.HiFi2"),
    ("DataType.BFLOAT4_B", "MathFidelity.LoFi"),
}


def _short_dtype(dtype: str) -> str:
    """`DataType.BFLOAT16` → `bf16` for the backend label."""
    last = dtype.rsplit(".", 1)[-1].lower()
    return {
        "bfloat16": "bf16",
        "bfloat8_b": "bf8",
        "bfloat4_b": "bf4",
        "float32": "fp32",
        "float16": "fp16",
    }.get(last, last)


def _short_fidelity(fid: str) -> str:
    """`MathFidelity.HiFi4` → `hifi4`."""
    return fid.rsplit(".", 1)[-1].lower()


def _backend_label(dtype: str, fid: str, use_trace: bool) -> str:
    suffix = "_traced" if use_trace else ""
    return f"tt_matmul_2d_{_short_dtype(dtype)}_{_short_fidelity(fid)}{suffix}"


def _filter_keep(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Apply user-direction config filter (skip BF8/BF4 except 1 sanity each)."""
    keep: list[dict[str, str]] = []
    sanity_seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["dtype"], r["math_fidelity"])
        if key in KEEP_FULL:
            keep.append(r)
        elif key in SANITY_FIRST and key not in sanity_seen:
            keep.append(r)
            sanity_seen.add(key)
    return keep


def _row_to_record(
    row: dict[str, str],
    sha: str,
    ts: str,
    power_w_avg: float | None,
) -> BenchResult:
    m = int(row["m"])
    k = int(row["k"])
    n = int(row["n"])
    use_trace = row["use_trace"].strip().lower() == "true"
    dtype = row["dtype"]
    fid = row["math_fidelity"]
    inference_ns = float(row["inference_time_avg [ns]"])
    median_ms = inference_ns / 1e6
    tflops = float(row["TFLOPs (avg)"])
    useful_ops = 2 * m * k * n
    backend = _backend_label(dtype, fid, use_trace)

    joules_per_op: float | None = None
    if power_w_avg is not None and useful_ops > 0:
        joules_per_op = power_w_avg * (median_ms * 1e-3) / float(useful_ops)

    return BenchResult(
        schema_version=SCHEMA_VERSION,
        device="Blackhole",
        device_detail={
            "name": "Tenstorrent Blackhole p150a (upstream tt-metal benchmark)",
            "source": "tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf",
            "grid_size": row.get("grid_size"),
            "in0_sharded": row.get("in0_sharded"),
            "out_sharded": row.get("out_sharded"),
            "in0_storage_type": row.get("in0_storage_type"),
            "in1_storage_type": row.get("in1_storage_type"),
            "out_storage_type": row.get("out_storage_type"),
            "use_trace": use_trace,
            "math_fidelity": fid,
            "dtype": dtype,
            "host_util_user_grid_pct": row.get(
                next(
                    (k for k in row if k.startswith("Host based utilization[%] (vs user")),
                    "",
                )
            ),
            "host_util_full_grid_pct": row.get(
                next(
                    (k for k in row if k.startswith("Host based utilization[%] (vs full")),
                    "",
                )
            ),
        },
        layer="B",
        backend=backend,
        shape={"M": m, "K": k, "N": n},
        dtype_in=_short_dtype(dtype),
        dtype_acc="fp32",
        iters=100,
        warmup=5,
        median_ms=median_ms,
        p10_ms=None,
        p90_ms=None,
        useful_ops=useful_ops,
        useful_op_kind="gemm_mac",
        throughput=tflops,
        throughput_unit="TFLOPS",
        correctness={"gate": "skipped"},
        host_overhead_ms=None,
        git_sha=sha,
        timestamp=ts,
        power_w_avg=power_w_avg,
        joules_per_useful_op=joules_per_op,
    )


def _run_upstream_pytest(tt_metal_home: Path, log_path: Path) -> int:
    env = os.environ.copy()
    env["TT_METAL_HOME"] = str(tt_metal_home)
    env["TT_METAL_RUNTIME_ROOT"] = str(tt_metal_home)
    env.setdefault("ARCH_NAME", "blackhole")
    env["PYTHONPATH"] = ":".join([
        str(_REPO_ROOT),
        str(tt_metal_home),
        str(tt_metal_home / "tests"),
        env.get("PYTHONPATH", ""),
    ])
    cmd = [
        str(Path("/home/chs/.tenstorrent-venv/bin/pytest")),
        "tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf",
        "-s",
        "--no-header",
        "-p", "no:cacheprovider",
        "-p", "tt_metal_extras.upstream_runner_conftest",
    ]
    print(f"Running upstream pytest (this is slow). Tee'ing log to {log_path}.")
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=tt_metal_home, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--tt-metal-home",
        default=os.environ.get("TT_METAL_HOME", "/home/chs/TT/tt-metal"),
        type=Path,
        help="TT-Metal source tree.",
    )
    p.add_argument(
        "--csv",
        type=Path,
        help="CSV path (default: <TT_METAL_HOME>/generated/matmul_2d_host_perf_report.csv).",
    )
    p.add_argument("--out", required=True, type=Path, help="JSONL output path.")
    p.add_argument(
        "--no-run", action="store_true",
        help="Skip the upstream pytest invocation; just convert an existing CSV.",
    )
    p.add_argument(
        "--no-filter", action="store_true",
        help="Emit every CSV row instead of applying the BF16/FP32 filter.",
    )
    p.add_argument(
        "--log",
        default="/tmp/upstream_bench.log",
        type=Path,
        help="Where to write the upstream pytest log when --no-run is unset.",
    )
    args = p.parse_args(argv)

    csv_path = args.csv or (args.tt_metal_home / "generated" / "matmul_2d_host_perf_report.csv")

    power_pre = read_tt_power_w()
    if not args.no_run:
        rc = _run_upstream_pytest(args.tt_metal_home, args.log)
        if rc != 0:
            print(f"Upstream pytest exited {rc}. Last 30 lines of log:", file=sys.stderr)
            try:
                tail = args.log.read_text(encoding="utf-8").splitlines()[-30:]
                for line in tail:
                    print(f"    {line}", file=sys.stderr)
            except OSError:
                pass
            return 1
    power_post = read_tt_power_w()
    power_w_avg = (
        (power_pre + power_post) / 2.0
        if power_pre is not None and power_post is not None else None
    )

    if not csv_path.is_file():
        print(f"CSV not found at {csv_path}", file=sys.stderr)
        return 2

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    kept = rows if args.no_filter else _filter_keep(rows)
    sha, ts = git_sha(), now_iso()
    records = [_row_to_record(r, sha, ts, power_w_avg) for r in kept]

    write_results(records, args.out)
    print(
        f"converted {len(rows)} CSV row(s) → kept {len(records)} → {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
