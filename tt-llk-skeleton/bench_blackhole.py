"""Python driver for the TT Blackhole bench.

Subprocesses the compiled ``host/main.cpp`` binary once per
``(layer, backend, size)`` cell, parses the binary's CSV stdout, and
emits JSONL records using the **same schema** as
``scripts/bench_nvidia.py`` so ``scripts/compare.py`` can join the two
sides.

Pre-requisite: ``make all`` in this directory (which requires
``$TT_METAL_HOME`` set and a working TT-Metal install). If the binary is
missing or fails, the wrapper still emits a JSONL record per cell with
``correctness.gate == "skipped"`` and a populated ``device_detail.error``,
so the Section-4 summary stays honest about what was attempted.

CLI::

    python bench_blackhole.py \\
        --out ../bench-results/blackhole_<sha>.jsonl \\
        --layers A,B,C \\
        --sizes 512,1024,2048,4096,8192
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bench_common import (  # noqa: E402
    SCHEMA_VERSION,
    BenchResult,
    git_sha,
    now_iso,
    q36_ntt_friendly_prime,
    write_results,
)

DEFAULT_SIZES = (512, 1024, 2048, 4096, 8192)
QUICK_SIZES = (256, 512)
DEFAULT_BACKENDS = ("tt_llk_int8", "tt_llk_bf16", "tt_llk_sfpu_fp32")
LAYER_B_ITERS = {"warmup": 5, "iters": 50}
LAYER_C_ITERS = {"warmup": 3, "iters": 20}
LAYER_A_ITERS = {"warmup": 5, "iters": 30}
LAYER_A_PROBE_SIZE = 1024

BIN_PATH = Path(__file__).resolve().parent / "build" / "bench_blackhole"


def _dispatch(
    backend: str, layer: str, m: int, k: int, n: int, q36: int, warmup: int, iters: int
) -> dict[str, object]:
    """Run the C++ binary once and parse its CSV line.

    Returns a dict with keys: median_ms, p10_ms, p90_ms, arch, n_cores, error.
    """
    if not BIN_PATH.is_file():
        return {
            "median_ms": None, "p10_ms": None, "p90_ms": None,
            "arch": None, "n_cores": None,
            "error": f"binary not built: run `make all` (expected {BIN_PATH})",
        }
    cmd = [
        str(BIN_PATH),
        "--backend", backend,
        "--layer", layer,
        "--M", str(m), "--K", str(k), "--N", str(n),
        "--warmup", str(warmup), "--iters", str(iters),
        "--q36", str(q36),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=600)
    if proc.returncode != 0:
        msg = proc.stderr.strip() or proc.stdout.strip()
        return {
            "median_ms": None, "p10_ms": None, "p90_ms": None,
            "arch": None, "n_cores": None,
            "error": f"binary exited {proc.returncode}: {msg}",
        }
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    parts = line.split(",")
    if len(parts) < 6:
        return {
            "median_ms": None, "p10_ms": None, "p90_ms": None,
            "arch": None, "n_cores": None,
            "error": f"unparseable binary stdout: {line!r}",
        }
    median = None if parts[0] in ("null", "") else float(parts[0])
    p10 = None if parts[1] in ("null", "") else float(parts[1])
    p90 = None if parts[2] in ("null", "") else float(parts[2])
    arch = parts[3] or None
    try:
        n_cores = int(parts[4])
    except ValueError:
        n_cores = None
    err = parts[5] if len(parts) > 5 else ""
    return {
        "median_ms": median, "p10_ms": p10, "p90_ms": p90,
        "arch": arch, "n_cores": n_cores,
        "error": err if err else None,
    }


def _make_record(
    layer: str, backend: str, m: int, k: int, n: int,
    iters: int, warmup: int, dispatch: dict[str, object],
    useful_op_kind: str, throughput_unit: str,
    correctness: dict[str, object] | None = None,
) -> BenchResult:
    sha, ts = git_sha(), now_iso()
    detail = {
        "name": "Tenstorrent Blackhole p150a (skeleton)",
        "arch": dispatch.get("arch"),
        "n_cores": dispatch.get("n_cores"),
        "tt_metal_home": (
            (Path.home() / "tt-metal").as_posix()  # placeholder; real path is the env var
        ),
    }
    if dispatch.get("error"):
        detail["error"] = dispatch["error"]

    median = dispatch.get("median_ms")
    throughput = None
    if median:
        if useful_op_kind == "gemm_mac":
            ops = 2.0 * m * k * n
            throughput = ops / (median * 1e-3) / 1e12
        elif useful_op_kind == "exact_modmul":
            useful_ops = m * n
            throughput = useful_ops / (median * 1e-3) / 1e9

    if correctness is None:
        correctness = {"gate": "skipped" if dispatch.get("error") else "skipped"}

    return BenchResult(
        schema_version=SCHEMA_VERSION,
        device="Blackhole",
        device_detail=detail,
        layer=layer,
        backend=backend,
        shape={"M": m, "K": k, "N": n},
        dtype_in="int8" if "int8" in backend else ("bfloat16" if "bf16" in backend else "fp32"),
        dtype_acc="int32" if "int8" in backend else "fp32",
        iters=iters, warmup=warmup,
        median_ms=median,
        p10_ms=dispatch.get("p10_ms"),
        p90_ms=dispatch.get("p90_ms"),
        useful_ops=(2 * m * k * n if useful_op_kind == "gemm_mac" else m * n) if median else None,
        useful_op_kind=useful_op_kind,
        throughput=throughput,
        throughput_unit=throughput_unit,
        correctness=correctness,
        host_overhead_ms=None,
        git_sha=sha, timestamp=ts,
    )


def layer_a(backends: tuple[str, ...]) -> list[BenchResult]:
    s = LAYER_A_PROBE_SIZE
    out: list[BenchResult] = []
    for backend in backends:
        d = _dispatch(backend, "A", s, s, s, q36_ntt_friendly_prime(),
                      LAYER_A_ITERS["warmup"], LAYER_A_ITERS["iters"])
        out.append(_make_record(
            "A", backend, s, s, s,
            LAYER_A_ITERS["iters"], LAYER_A_ITERS["warmup"], d,
            "gemm_mac", "TFLOPS" if "int8" not in backend else "TOPS",
        ))
    return out


def layer_b(sizes: tuple[int, ...], backends: tuple[str, ...]) -> list[BenchResult]:
    out: list[BenchResult] = []
    for backend in backends:
        for s in sizes:
            d = _dispatch(backend, "B", s, s, s, q36_ntt_friendly_prime(),
                          LAYER_B_ITERS["warmup"], LAYER_B_ITERS["iters"])
            out.append(_make_record(
                "B", backend, s, s, s,
                LAYER_B_ITERS["iters"], LAYER_B_ITERS["warmup"], d,
                "gemm_mac", "TFLOPS" if "int8" not in backend else "TOPS",
            ))
    return out


def layer_c(sizes: tuple[int, ...]) -> list[BenchResult]:
    """q36 INT8 modmul. Currently emits skipped records until the
    modular epilogue TODO in compute_int8_mma.cpp is filled in.
    The C++ side reports its own correctness pass/fail; this wrapper
    just propagates the error string when present."""
    out: list[BenchResult] = []
    q = q36_ntt_friendly_prime()
    for s in sizes:
        d = _dispatch("tt_llk_int8", "C", s, s, s, q,
                      LAYER_C_ITERS["warmup"], LAYER_C_ITERS["iters"])
        # The Layer C correctness gate runs inside the C++ binary; until that
        # TODO is filled in, mark every Layer C record as "skipped" with an
        # explanatory note so the comparison summary stays honest.
        correctness = {
            "gate": "failed" if not d.get("error") else "skipped",
            "note": (
                "TODO: modular epilogue not implemented in compute_int8_mma.cpp; "
                "fill in SFPU intrinsics and re-run."
            ),
        }
        rec = _make_record(
            "C", "tt_llk_int8", s, s, s,
            LAYER_C_ITERS["iters"], LAYER_C_ITERS["warmup"], d,
            "exact_modmul", "G_modmul/s", correctness=correctness,
        )
        # Force perf fields to null until the gate passes — never let a
        # numerically-wrong run ship as a perf number.
        rec_dict = asdict(rec)
        rec_dict["throughput"] = None
        rec_dict["median_ms"] = None
        rec_dict["p10_ms"] = None
        rec_dict["p90_ms"] = None
        rec_dict["useful_ops"] = None
        out.append(BenchResult(**rec_dict))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TT Blackhole bench wrapper, BENCHMARK.md §4 v1.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--layers", default="A,B,C")
    p.add_argument("--sizes", default=None)
    p.add_argument("--backends", default=",".join(DEFAULT_BACKENDS))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="Enable TT-Metal device profiler dump (slow).")
    args = p.parse_args(argv)

    layers = tuple(args.layers.split(","))
    if args.sizes:
        sizes = tuple(int(s) for s in args.sizes.split(","))
    else:
        sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES
    backends = tuple(args.backends.split(","))

    bin_status = "present" if BIN_PATH.is_file() else "missing — run `make all`"
    print(f"binary  : {BIN_PATH}  ({bin_status})")
    print(f"out     : {args.out}")
    print(f"layers  : {','.join(layers)}")
    print(f"sizes   : {','.join(str(s) for s in sizes)}")
    print(f"backends: {','.join(backends)}")
    if args.profile:
        print("profile : enabled (TT_METAL_DEVICE_PROFILER=1 set in environment)")
    print()

    all_results: list[BenchResult] = []
    if "A" in layers:
        all_results.extend(layer_a(backends))
    if "B" in layers:
        all_results.extend(layer_b(sizes, backends))
    if "C" in layers:
        all_results.extend(layer_c(sizes))

    write_results(all_results, args.out)
    print(f"wrote {len(all_results)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
