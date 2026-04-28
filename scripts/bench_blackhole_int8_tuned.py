"""Phase 7 (INT8 extension) — tuned INT8 matmul benchmark wrapper.

Closes one of the open items in `doc/04_phase7_tuned_matmul.md` §4.7:
upstream `test_matmul_2d_host_perf` doesn't iterate INT8, so we run our
own port of the upstream block-tiled `matmul_multicore_reuse` programming
example, adapted to INT8 (CB DataFormat::Int8 + sign-magnitude inputs +
fp32_dest_acc_en for the INT32 dst accumulator). The compute kernel
re-used unchanged is upstream's `bmm_large_block_zm.cpp` — same LLK
calls as our v1 reference, but with block / sub-block orchestration.

Sweep mirrors the upstream BF16 shape list (per-core base × grid_size).
For each (M, K, N) per-core triple in the upstream BF16 list:

    M_total = M_base × grid.y
    K_total = K_base × grid.x
    N_total = N_base × grid.x

For a 11×10 grid: per-core (256, 256, 256) → (2560, 2816, 2816). Compared
side-by-side with `tt_matmul_2d_bf16_hifi4` at the equivalent shape so
the BF16-vs-INT8 ratio falls out at the same problem size.

CLI::

    cd tt-llk-skeleton && make bench_int8_tuned
    uv run python scripts/bench_blackhole_int8_tuned.py \\
        --out bench-results/blackhole_<sha>_int8_tuned.jsonl
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

# Per-core (M, K, N) base shapes from upstream `matmul_shapes_bfloat8_b`
# (the BF16 list scales the same way; we use the BF8 list because INT8 is
# 1 byte/element like BF8 — same L1 footprint per tile). Each row scales
# by the grid: M *= grid_y, K/N *= grid_x. in0_block_w = K_base / 32
# / in0_block_w_div (the 6th column upstream).
#
# Trim to the same set the user used for Phase 7's BF16 in-full sweep
# (medium → large square shapes; small ones are dispatch-bound and don't
# show silicon throughput).
PER_CORE_SHAPES = [
    # (M_base, K_base, N_base, in0_block_w_div, label)
    (128, 256, 256, 1, "small"),
    (256, 256, 256, 1, "medium"),
    (256, 384, 384, 1, "med-rect"),
    (384, 384, 384, 2, "large"),
    (384, 384, 512, 2, "large-rect"),
    (416, 320, 320, 1, "p150-square"),
    (512, 512, 512, 1, "xlarge"),
]

# Two fidelities — both should produce the same throughput for INT8 since
# fidelity is fixed at HiFi4 internally for INT8 by the matrix engine,
# but reporting both confirms the equivalence. Default to HiFi4.
DEFAULT_FIDELITIES = ("HiFi4", "HiFi2")

DEFAULT_BIN = (
    _REPO_ROOT / "tt-llk-skeleton" / "build_int8_tuned" / "bench_int8_tuned"
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
        str(binary),
        "--M", str(M), "--K", str(K), "--N", str(N),
        "--fidelity", fidelity,
        "--in0-block-w", str(in0_block_w),
        "--warmup", str(warmup), "--iters", str(iters),
    ]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
        env={**os.environ, "TT_LOG_FILE_DEFAULT": "/tmp/tt_int8_tuned.log"},
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

    useful_ops = 2 * M * K * N  # GEMM_MAC convention: 1 mul + 1 add per MAC

    throughput: float | None = None
    if median_ms is not None and median_ms > 0:
        throughput = useful_ops / (median_ms * 1e-3) / 1e12  # TOPS

    joules_per_op: float | None = None
    if (
        power_w_avg is not None
        and median_ms is not None
        and useful_ops > 0
    ):
        joules_per_op = power_w_avg * (median_ms * 1e-3) / float(useful_ops)

    backend = f"tt_matmul_2d_int8_{fidelity.lower()}"
    return BenchResult(
        schema_version=SCHEMA_VERSION,
        device="Blackhole",
        device_detail={
            "name": "Tenstorrent Blackhole p150a (block-tiled INT8 matmul, no mcast)",
            "harness": "tt-llk-skeleton/host_int8_tuned/main.cpp",
            "compute_kernel": (
                "tt_metal/programming_examples/matmul/matmul_common/"
                "kernels/compute/bmm_large_block_zm.cpp (upstream, unmodified)"
            ),
            "n_cores": n_cores,
            "math_fidelity": f"MathFidelity.{fidelity}",
            "fp32_dest_acc_en": True,
            "in0_block_w": in0_block_w,
            "shape_label": label,
            "note": (
                "INT8 block-tiled tuned matmul — closes the §4.7 INT8 open "
                "item. Block reuse on, operand multicast off. Joins "
                "cuBLASLt INT8 and the v1 reference tt_llk_int8 in the "
                "`int8` BACKEND_CLASS so the three rows sit side-by-side."
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
    p.add_argument(
        "--grid-x", type=int, default=11,
        help="Compute grid X (cores along inner-K direction). 11 on harvested p150a.",
    )
    p.add_argument(
        "--grid-y", type=int, default=10,
        help="Compute grid Y (cores along outer-M direction). 10 on harvested p150a.",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=30)
    args = p.parse_args(argv)

    if not args.bin.is_file():
        print(
            f"binary not found at {args.bin}. Build with: "
            "cd tt-llk-skeleton && make bench_int8_tuned",
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
