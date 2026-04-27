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
so the §4 summary stays honest about what was attempted.

CLI::

    python bench_blackhole.py \\
        --out ../bench-results/blackhole_<sha>.jsonl \\
        --layers A,B,C \\
        --sizes 512,1024,2048,4096,8192
"""

from __future__ import annotations

import argparse
import os
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
    correctness_gate,
    git_sha,
    now_iso,
    q36_ntt_friendly_prime,
    q48_ntt_friendly_prime,
    write_results,
)

# BF16, INT8, and FP32-matrix backends have working compute kernels (real
# Blackhole matrix-engine matmul). SFPU FP32 has no LLK matmul/FMA tile
# primitive — it emits a "skipped" record via the C++ binary's early-skip,
# documented in BENCHMARK_TT.md §5(c).
DEFAULT_BACKENDS = (
    "tt_llk_bf16",
    "tt_llk_int8",
    "tt_llk_fp32_matrix",
    "tt_llk_sfpu_fp32",
)
DEFAULT_SIZES = (512, 1024, 2048, 4096, 8192)
QUICK_SIZES = (256, 512)
LAYER_B_ITERS = {"warmup": 5, "iters": 50}
LAYER_C_ITERS = {"warmup": 3, "iters": 20}
LAYER_A_ITERS = {"warmup": 5, "iters": 30}
LAYER_A_PROBE_SIZE = 1024
# Layer C primes and their byte-chunk decomposition counts. Each entry maps
# a prime size in bits to (prime, n_chunks_per_operand). One Layer C "modmul"
# dispatches n_chunks**2 INT8 GEMMs per timed iteration. The C++ binary loops
# internally based on the --n-gemms arg; these constants drive both that arg
# and the host bigint reference inside `_qn_int8_modmul_host`.
LAYER_C_PRIMES = {
    36: (q36_ntt_friendly_prime(), 5),  # 5x5 = 25 dispatches per modmul
    48: (q48_ntt_friendly_prime(), 6),  # 6x6 = 36 dispatches per modmul
}

BIN_PATH = Path(__file__).resolve().parent / "build" / "bench_blackhole"


def _qn_int8_modmul_host(
    a_list: list[int], b_list: list[int], q: int, n_chunks: int
) -> list[int]:
    """Host bigint reference for the n_chunks×n_chunks byte-chunk modmul recipe.

    The same algebra the on-device path *would* implement; we use this to
    drive the Layer C correctness gate while the INT8 perf path supplies the
    wall-clock cost. Bit-exact by construction (Python ``int``).
    """
    out: list[int] = []
    for a, b in zip(a_list, b_list, strict=True):
        a_chunks = [(a >> (8 * i)) & 0xFF for i in range(n_chunks)]
        b_chunks = [(b >> (8 * j)) & 0xFF for j in range(n_chunks)]
        acc = 0
        for i in range(n_chunks):
            for j in range(n_chunks):
                acc += a_chunks[i] * b_chunks[j] * pow(1 << (8 * (i + j)), 1, q)
        out.append(acc % q)
    return out


def _dispatch(
    backend: str, layer: str, m: int, k: int, n: int, q36: int, warmup: int, iters: int,
    n_gemms: int = 25,
) -> dict[str, object]:
    """Run the C++ binary once and parse its CSV line.

    ``n_gemms`` tunes Layer C's per-iteration dispatch count to match the
    byte-chunk recipe: 25 for q36 (5×5), 36 for q48 (6×6).

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
        "--n-gemms", str(n_gemms),
    ]
    # The C++ binary needs TT_METAL_HOME set so JIT can find runtime headers.
    env = os.environ.copy()
    env.setdefault("TT_METAL_HOME", str(Path.home() / "TT" / "tt-metal"))
    env.setdefault("ARCH_NAME", "blackhole")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, check=False, timeout=600, env=env
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        return {
            "median_ms": None, "p10_ms": None, "p90_ms": None,
            "arch": None, "n_cores": None,
            "error": f"binary exited {proc.returncode}: "
                     f"{(proc.stderr or '').strip()[:200] or 'no stderr'}",
        }
    # The binary may print log lines before the CSV; the CSV is always the
    # last line of stdout (emit_csv calls printf with a single line).
    line = ""
    for cand in reversed(proc.stdout.strip().splitlines()):
        if "," in cand and not cand.startswith(("20", "[", "0x")):
            line = cand
            break
    parts = line.split(",")
    if len(parts) < 6:
        return {
            "median_ms": None, "p10_ms": None, "p90_ms": None,
            "arch": None, "n_cores": None,
            "error": f"unparseable binary stdout (last line {line!r})",
        }
    median = None if parts[0] in ("null", "") else float(parts[0])
    p10 = None if parts[1] in ("null", "") else float(parts[1])
    p90 = None if parts[2] in ("null", "") else float(parts[2])
    arch = parts[3] or None
    try:
        n_cores = int(parts[4])
    except ValueError:
        n_cores = None
    err = ",".join(parts[5:]).strip()
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
    extra_detail: dict[str, object] | None = None,
) -> BenchResult:
    sha, ts = git_sha(), now_iso()
    detail: dict[str, object] = {
        "name": "Tenstorrent Blackhole p150a",
        "arch": dispatch.get("arch"),
        "n_cores": dispatch.get("n_cores"),
        "tt_metal_home": os.environ.get("TT_METAL_HOME"),
    }
    if dispatch.get("error"):
        detail["error"] = dispatch["error"]
    if extra_detail:
        detail.update(extra_detail)

    median = dispatch.get("median_ms")
    throughput: float | None = None
    if median is not None and median > 0:
        if useful_op_kind == "gemm_mac":
            ops = 2.0 * m * k * n
            throughput = ops / (float(median) * 1e-3) / 1e12
        elif useful_op_kind.startswith("exact_modmul"):
            # One "exact modmul" per output element; the binary's median
            # already includes the full n²-GEMM cascade.
            useful_ops = m * n
            throughput = useful_ops / (float(median) * 1e-3) / 1e9

    if correctness is None:
        correctness = {"gate": "skipped"}

    if useful_op_kind == "gemm_mac":
        useful_ops_val: int | None = int(2 * m * k * n) if median is not None else None
    else:
        useful_ops_val = m * n if median is not None else None

    dtype_in = (
        "int8" if "int8" in backend
        else ("bfloat16" if "bf16" in backend else "fp32")
    )
    dtype_acc = "int32" if "int8" in backend else "fp32"

    return BenchResult(
        schema_version=SCHEMA_VERSION,
        device="Blackhole",
        device_detail=detail,
        layer=layer,
        backend=backend,
        shape={"M": m, "K": k, "N": n},
        dtype_in=dtype_in,
        dtype_acc=dtype_acc,
        iters=iters, warmup=warmup,
        median_ms=median,
        p10_ms=dispatch.get("p10_ms"),
        p90_ms=dispatch.get("p90_ms"),
        useful_ops=useful_ops_val,
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


def layer_c(sizes: tuple[int, ...], primes: tuple[int, ...] = (36, 48)) -> list[BenchResult]:
    """Layer C — exact modular product across one or more prime sizes.

    Recipe correctness is gated on host (bigint reference). The on-device
    perf path is ``n_chunks**2 × INT8 matmul`` per timed iteration, matching
    the byte-decomposition cost shape used by the NVIDIA cuBLASLt INT8
    path. Per-prime ``useful_op_kind`` is ``exact_modmul_q36`` /
    ``exact_modmul_q48`` so compare.py renders one row per prime size.
    """
    out: list[BenchResult] = []
    for prime_bits in primes:
        if prime_bits not in LAYER_C_PRIMES:
            raise ValueError(f"unknown Layer C prime size {prime_bits}; "
                             f"add it to LAYER_C_PRIMES")
        q, n_chunks = LAYER_C_PRIMES[prime_bits]
        n_gemms = n_chunks * n_chunks
        op_kind = f"exact_modmul_q{prime_bits}"

        # Run the host-side gate once per prime; pure recipe check.
        gate = correctness_gate(
            lambda a, b, q=q, nc=n_chunks: _qn_int8_modmul_host(a, b, q, nc), q,
        )

        for s in sizes:
            d = _dispatch("tt_llk_int8", "C", s, s, s, q,
                          LAYER_C_ITERS["warmup"], LAYER_C_ITERS["iters"],
                          n_gemms=n_gemms)
            if gate.gate == "passed" and d.get("error") is None:
                corr = {
                    "gate": "passed",
                    "edge_cases": gate.edge_cases,
                    "edge_cases_passed": gate.edge_cases_passed,
                    "edge_cases_failed": gate.edge_cases_failed,
                    "note": (
                        f"host bigint reference for the {n_chunks}x{n_chunks} "
                        f"byte recipe; perf path is {n_gemms}× INT8→INT32-accum "
                        f"matmul on Blackhole (q{prime_bits})"
                    ),
                }
            else:
                corr = {
                    "gate": "skipped" if gate.gate == "passed" else "failed",
                    "edge_cases_passed": gate.edge_cases_passed,
                    "edge_cases_failed": gate.edge_cases_failed,
                    "note": (
                        "device dispatch error" if d.get("error") else
                        "host gate failed (would not happen with the bigint "
                        "reference; report a bug)"
                    ),
                }
            rec = _make_record(
                "C", "tt_llk_int8", s, s, s,
                LAYER_C_ITERS["iters"], LAYER_C_ITERS["warmup"], d,
                op_kind, "G_modmul/s", correctness=corr,
                extra_detail={
                    "n_int8_gemms_per_modmul": n_gemms,
                    "n_chunks": n_chunks,
                    "prime_bits": prime_bits,
                    "recipe": (
                        f"{n_chunks}x{n_chunks} byte decomposition; "
                        f"{n_gemms}× INT8→INT32-accum matmul cost proxy on "
                        f"Blackhole; reduction on host bigint"
                    ),
                    "q": q,
                },
            )
            # If anything went wrong with the device dispatch, force perf
            # fields to null so a wrong number can never ship.
            if d.get("error") or corr["gate"] == "failed":
                rec_dict = asdict(rec)
                for k in ("throughput", "median_ms", "p10_ms", "p90_ms", "useful_ops"):
                    rec_dict[k] = None
                rec = BenchResult(**rec_dict)
            out.append(rec)
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
    p.add_argument("--primes", default="36,48",
                   help="Comma-separated Layer C prime sizes in bits "
                        "(default: 36,48). Each entry must be a key in "
                        "LAYER_C_PRIMES.")
    args = p.parse_args(argv)

    layers = tuple(args.layers.split(","))
    if args.sizes:
        sizes = tuple(int(s) for s in args.sizes.split(","))
    else:
        sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES
    backends = tuple(args.backends.split(","))
    primes = tuple(int(p) for p in args.primes.split(","))

    bin_status = "present" if BIN_PATH.is_file() else "missing — run `make all`"
    print(f"binary  : {BIN_PATH}  ({bin_status})")
    print(f"out     : {args.out}")
    print(f"layers  : {','.join(layers)}")
    print(f"sizes   : {','.join(str(s) for s in sizes)}")
    print(f"backends: {','.join(backends)}")
    if args.profile:
        os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
        print("profile : enabled (TT_METAL_DEVICE_PROFILER=1)")
    print()

    all_results: list[BenchResult] = []
    if "A" in layers:
        print("Layer A (capability probe)…")
        all_results.extend(layer_a(backends))
    if "B" in layers:
        print("Layer B (raw GEMM)…")
        all_results.extend(layer_b(sizes, backends))
    if "C" in layers:
        prime_label = ",".join(f"q{pb}" for pb in primes)
        print(f"Layer C ({prime_label} exact modmul, n²-GEMM cost proxy)…")
        all_results.extend(layer_c(sizes, primes=primes))

    write_results(all_results, args.out)
    print(f"\nwrote {len(all_results)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
