"""Phase 7 (INT8 mcast extension) — INT8 tuned matmul *with operand multicast*.

Companion to ``scripts/bench_blackhole_int8_tuned.py``: same shape sweep
and JSONL schema, distinct backend labels (``tt_matmul_2d_int8_mcast_*``)
so non-mcast and mcast rows sit side-by-side in SUMMARY.

Pre-requisite: ``cd tt-llk-skeleton && make bench_int8_tuned_mcast``.

CLI::

    uv run python scripts/bench_blackhole_int8_tuned_mcast.py \\
        --out bench-results/blackhole_<sha>_int8_tuned_mcast.jsonl
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

PER_CORE_SHAPES = [
    (128, 256, 256, 1, "small"),
    (256, 256, 256, 1, "medium"),
    (256, 384, 384, 1, "med-rect"),
    (384, 384, 384, 2, "large"),
    (384, 384, 512, 2, "large-rect"),
    (416, 320, 320, 1, "p150-square"),
    (512, 512, 512, 1, "xlarge"),
]
DEFAULT_FIDELITIES = ("HiFi4", "HiFi2")
DEFAULT_BIN = (
    _REPO_ROOT / "tt-llk-skeleton" / "build_int8_tuned_mcast"
    / "bench_int8_tuned_mcast"
)


def _is_float(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def _parse_csv(line: str) -> dict[str, str]:
    parts = line.rstrip("\n").split(",")
    if len(parts) < 7:
        raise ValueError(f"unexpected CSV row: {line!r}")
    return {
        "median_ms": parts[0], "p10_ms": parts[1], "p90_ms": parts[2],
        "arch": parts[3], "n_cores": parts[4],
        "gate": parts[5], "err": ",".join(parts[6:]).strip(),
    }


def _run_one(
    binary: Path, M: int, K: int, N: int, fidelity: str, in0_block_w: int,
    warmup: int, iters: int,
) -> dict[str, str]:
    cmd = [
        str(binary), "--M", str(M), "--K", str(K), "--N", str(N),
        "--fidelity", fidelity, "--in0-block-w", str(in0_block_w),
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
        env={**os.environ, "TT_LOG_FILE_DEFAULT": "/tmp/tt_int8_mcast.log"},
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
    return _parse_csv(csv_line)


def _to_record(
    csv_row: dict[str, str], M: int, K: int, N: int, fidelity: str,
    in0_block_w: int, label: str, warmup: int, iters: int,
    sha: str, ts: str, power_w_avg: float | None,
) -> BenchResult:
    n_cores = int(csv_row["n_cores"]) if csv_row["n_cores"].isdigit() else 0

    def _f(s: str) -> float | None:
        return None if s == "null" else float(s)
    median_ms = _f(csv_row["median_ms"])
    p10_ms = _f(csv_row["p10_ms"])
    p90_ms = _f(csv_row["p90_ms"])

    useful_ops = 2 * M * K * N
    throughput: float | None = None
    if median_ms is not None and median_ms > 0:
        throughput = useful_ops / (median_ms * 1e-3) / 1e12  # TOPS

    joules_per_op: float | None = None
    if power_w_avg is not None and median_ms is not None and useful_ops > 0:
        joules_per_op = power_w_avg * (median_ms * 1e-3) / float(useful_ops)

    backend = f"tt_matmul_2d_int8_mcast_{fidelity.lower()}"
    return BenchResult(
        schema_version=SCHEMA_VERSION,
        device="Blackhole",
        device_detail={
            "name": "Tenstorrent Blackhole p150a (block-tiled INT8 matmul + 2D operand mcast)",
            "harness": "tt-llk-skeleton/host_int8_tuned_mcast/main.cpp",
            "compute_kernel": (
                "tt_metal/programming_examples/matmul/matmul_common/"
                "kernels/compute/bmm_large_block_zm.cpp (upstream, unmodified)"
            ),
            "reader_kernels": (
                "tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/"
                "reader_bmm_tile_layout_in0_{sender,receiver}_in1_{sender,receiver}.cpp"
            ),
            "writer_kernel": (
                "tt_metal/programming_examples/matmul/matmul_common/kernels/dataflow/"
                "writer_bmm_tile_layout.cpp (split across 2 NoC processors)"
            ),
            "n_cores": n_cores,
            "math_fidelity": f"MathFidelity.{fidelity}",
            "fp32_dest_acc_en": True,
            "in0_block_w": in0_block_w,
            "shape_label": label,
            "mcast_topology": "2D: in0 along X (rows), in1 along Y (cols)",
            "note": (
                "INT8 block-tiled tuned matmul WITH operand multicast — "
                "closes the §4.7 'INT8 with mcast' open item. Each core "
                "row reads in0 from a single sender; each column reads "
                "in1 from a single sender. Joins cuBLASLt, the v1 "
                "reference, and the non-mcast tuned in the `int8` class."
            ),
            "gate": csv_row.get("gate", "skipped"),
            "error": csv_row.get("err") or None,
        },
        layer="B",
        backend=backend,
        shape={"M": M, "K": K, "N": N},
        dtype_in="int8",
        dtype_acc="int32",
        iters=iters,
        warmup=warmup,
        median_ms=median_ms,
        p10_ms=p10_ms,
        p90_ms=p90_ms,
        useful_ops=useful_ops,
        useful_op_kind="gemm_mac",
        throughput=throughput,
        throughput_unit="TOPS",
        correctness={"gate": csv_row.get("gate", "skipped")},
        host_overhead_ms=None,
        git_sha=sha,
        timestamp=ts,
        power_w_avg=power_w_avg,
        joules_per_useful_op=joules_per_op,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--fidelities", type=str, default=",".join(DEFAULT_FIDELITIES),
        help=f"Comma-separated MathFidelity list (default: {','.join(DEFAULT_FIDELITIES)}).",
    )
    p.add_argument("--grid-x", type=int, default=11)
    p.add_argument("--grid-y", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args(argv)

    if not args.bin.is_file():
        print(
            f"binary not found at {args.bin}. Build with: "
            "cd tt-llk-skeleton && make bench_int8_tuned_mcast",
            file=sys.stderr,
        )
        return 2

    fidelities = tuple(f for f in args.fidelities.split(",") if f.strip())
    sha, ts = git_sha(), now_iso()
    records: list[BenchResult] = []
    for fid in fidelities:
        for M_base, K_base, N_base, in0_block_w_div, label in PER_CORE_SHAPES:
            M = M_base * args.grid_y
            K = K_base * args.grid_x
            N = N_base * args.grid_x
            in0_block_w = max(1, (K // args.grid_x // 32) // in0_block_w_div)
            print(
                f"  [run] {label:>10} {fid:>5} M={M:>5} K={K:>5} N={N:>5} "
                f"in0_block_w={in0_block_w}",
                file=sys.stderr,
            )
            power_pre = read_tt_power_w()
            csv_row = _run_one(args.bin, M, K, N, fid, in0_block_w,
                               args.warmup, args.iters)
            power_post = read_tt_power_w()
            power_w_avg = (
                (power_pre + power_post) / 2.0
                if power_pre is not None and power_post is not None else None
            )
            rec = _to_record(csv_row, M, K, N, fid, in0_block_w, label,
                             args.warmup, args.iters, sha, ts, power_w_avg)
            records.append(rec)
            tflops = (
                f"{rec.throughput:.2f} TOPS"
                if rec.throughput is not None else "skipped"
            )
            print(
                f"     → median={rec.median_ms} ms, throughput={tflops}",
                file=sys.stderr,
            )

    write_results(records, args.out)
    print(f"wrote {len(records)} record(s) → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
