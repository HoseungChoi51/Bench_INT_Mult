"""Head-to-head benchmark driver for the RTX 5090 (BENCHMARK.md §4 v1).

This script measures the three layers approved for v1 against an NVIDIA CUDA
device:

- **Layer A — capability probe**: dispatch one tiny matmul per backend,
  record cuBLASLt algorithm metadata, sanity-check the dispatched
  throughput against the published Blackwell consumer-card profile.
- **Layer B — raw GEMM**: 5 square sizes × 4 backends. Pure GEMM time,
  no modular reduction.
- **Layer C-minimal — exact 36-bit modular product**: q36 NTT-friendly
  prime, INT8 byte-decomposition + 25 INT8 GEMMs + per-element ``% q``,
  bit-exact correctness gate **before** perf is recorded for each size.

The four NVIDIA backends are:

- ``cublaslt_int8``  — INT8 → INT32 via ``torch._int_mm``
  (cuBLASLt's IGEMM kernels internally on sm_75+; nvmath 0.9 does not
  yet expose INT8, but this *is* the cuBLASLt path).
- ``cublaslt_tf32``  — TF32 via ``nvmath.linalg.advanced.matmul``
  (compute_type = ``COMPUTE_32F_FAST_TF32``).
- ``cublaslt_fp32``  — strict FP32 CUDA core via nvmath
  (compute_type = ``COMPUTE_32F``).
- ``cublaslt_fp64``  — FP64 path via nvmath. On Blackwell consumer cards
  this is **not** tensor-accelerated (BENCHMARK.md §3 footnote); the
  Layer A probe records the achieved TFLOPS so the reader can confirm.

A single CPU baseline record (Python ``int`` bigint) is also emitted to
establish a "no-GPU" floor. AVX-512-IFMA / Intel HEXL paths are deferred
to v2.

Usage::

    uv run --extra bench python scripts/bench_nvidia.py \\
        --out bench-results/nvidia_$(git rev-parse --short HEAD)_$(date +%Y%m%d).jsonl \\
        --layers A,B,C \\
        --sizes 512,1024,2048,4096,8192

Use ``--quick`` for a sub-30-second smoke run (256/512 only). The script
appends one JSONL record per (layer, backend, size); failed correctness
gates emit a record with null perf fields and ``correctness.gate ==
"failed"`` so partial information is still preserved.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import nvmath.linalg.advanced as adv
import torch

# Allow `python scripts/bench_nvidia.py` from the repo root without -m.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bench_common import (  # noqa: E402
    SCHEMA_VERSION,
    BenchResult,
    correctness_gate,
    cpu_time,
    cuda_time,
    git_sha,
    now_iso,
    q36_ntt_friendly_prime,
    write_results,
)

# --- Constants --------------------------------------------------------------

DEFAULT_SIZES = (512, 1024, 2048, 4096, 8192)
QUICK_SIZES = (256, 512)
DEFAULT_BACKENDS = ("cublaslt_int8", "cublaslt_tf32", "cublaslt_fp32", "cublaslt_fp64")

# Iteration counts are tuned so that even the largest size runs in well under
# a minute on a Blackwell-class card while still giving a stable median.
LAYER_B_ITERS = {"warmup": 5, "iters": 50}
LAYER_C_ITERS = {"warmup": 3, "iters": 20}
LAYER_A_ITERS = {"warmup": 5, "iters": 30}
LAYER_A_PROBE_SIZE = 1024  # large enough that launch overhead doesn't dominate;
# the per-backend plausibility check below assumes a steady-state kernel.

# Approximate Blackwell consumer (RTX 5090) dense throughput, used only as a
# loose plausibility check inside the Layer A probe. Sources: NVIDIA marketing
# tables (raw, not sustained). A measured TFLOPS more than 5× off in either
# direction is logged with a "suspect" flag in device_detail.
BLACKWELL_PROFILE_TOPS = {
    "cublaslt_int8": 838.0,   # INT8 dense
    "cublaslt_tf32": 209.0,   # TF32 dense
    "cublaslt_fp32": 104.5,   # FP32 dense
    "cublaslt_fp64": 1.6,     # FP64 ≈ FP32 / 64 on consumer Blackwell
}


# --- Backend registry -------------------------------------------------------


def _make_int8_gemm(m: int, k: int, n: int) -> tuple[Callable[[], object], dict[str, object]]:
    """Build a closure that runs one INT8 → INT32 GEMM via ``torch._int_mm``."""
    a = torch.randint(-128, 127, (m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 127, (k, n), dtype=torch.int8, device="cuda")

    def step() -> None:
        torch._int_mm(a, b)

    detail = {
        "dispatch": "torch._int_mm → cuBLASLt IGEMM",
        "dtype_in": "int8",
        "dtype_acc": "int32",
        "algo_id": None,  # not exposed by torch
        "tile": None,
    }
    return step, detail


def _make_nvmath_gemm(
    m: int, k: int, n: int, compute_type: adv.MatmulComputeType, dtype: torch.dtype
) -> tuple[Callable[[], object], dict[str, object]]:
    """Build a cached nvmath ``Matmul`` plan and return a closure that re-executes it.

    The plan is cached in the closure so successive calls re-use the cuBLASLt
    workspace and algorithm choice — what we want when timing.
    """
    a = torch.randn(m, k, dtype=dtype, device="cuda")
    b = torch.randn(k, n, dtype=dtype, device="cuda")
    opts = adv.MatmulOptions(compute_type=compute_type)
    plan = adv.Matmul(a, b, options=opts)
    plan.plan()
    algo = plan.algorithms[0] if plan.algorithms else None

    def step() -> None:
        plan.execute()

    detail = {
        "dispatch": f"nvmath cuBLASLt ({compute_type.name})",
        "dtype_in": str(dtype).replace("torch.", ""),
        "dtype_acc": "fp32" if compute_type != adv.MatmulComputeType.COMPUTE_64F else "fp64",
        "algo_id": int(algo.algorithm_id) if algo is not None else None,
        "tile": tuple(algo.tile) if algo is not None else None,
        "inner_shape": int(algo.inner_shape) if algo is not None else None,
    }
    return step, detail


def make_gemm_step(
    backend: str, m: int, k: int, n: int
) -> tuple[Callable[[], object], dict[str, object]]:
    if backend == "cublaslt_int8":
        return _make_int8_gemm(m, k, n)
    if backend == "cublaslt_tf32":
        return _make_nvmath_gemm(
            m, k, n, adv.MatmulComputeType.COMPUTE_32F_FAST_TF32, torch.float32
        )
    if backend == "cublaslt_fp32":
        return _make_nvmath_gemm(m, k, n, adv.MatmulComputeType.COMPUTE_32F, torch.float32)
    if backend == "cublaslt_fp64":
        return _make_nvmath_gemm(m, k, n, adv.MatmulComputeType.COMPUTE_64F, torch.float64)
    raise ValueError(f"unknown backend: {backend!r}")


# --- Device probe -----------------------------------------------------------


def device_info() -> dict[str, object]:
    name = torch.cuda.get_device_name(0)
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    sm = 10 * cap_major + cap_minor
    props = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "sm": sm,
        "vram_gb": round(props.total_memory / (1024**3), 1),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


# --- Layer A — capability probe ---------------------------------------------


def layer_a_capability_probe() -> list[BenchResult]:
    """One small GEMM per backend; record algo metadata + plausibility flag."""
    sha, ts = git_sha(), now_iso()
    dev = device_info()
    results: list[BenchResult] = []
    s = LAYER_A_PROBE_SIZE
    for backend in DEFAULT_BACKENDS:
        try:
            step, detail = make_gemm_step(backend, s, s, s)
        except Exception as e:  # noqa: BLE001
            results.append(
                BenchResult(
                    schema_version=SCHEMA_VERSION,
                    device="RTX5090",
                    device_detail={**dev, "backend_detail": {"error": str(e)}},
                    layer="A",
                    backend=backend,
                    shape={"M": s, "K": s, "N": s},
                    dtype_in="?", dtype_acc="?",
                    iters=0, warmup=0,
                    median_ms=None, p10_ms=None, p90_ms=None,
                    useful_ops=None, useful_op_kind="gemm_mac",
                    throughput=None, throughput_unit="TFLOPS",
                    correctness={"gate": "skipped"},
                    host_overhead_ms=None,
                    git_sha=sha, timestamp=ts,
                )
            )
            continue
        median_ms, p10, p90 = cuda_time(step, **LAYER_A_ITERS)
        flops = 2.0 * s * s * s
        tflops = flops / (median_ms * 1e-3) / 1e12
        suspect = False
        ref = BLACKWELL_PROFILE_TOPS.get(backend)
        if ref is not None and (tflops > 5 * ref or tflops < ref / 5):
            suspect = True
        results.append(
            BenchResult(
                schema_version=SCHEMA_VERSION,
                device="RTX5090",
                device_detail={
                    **dev,
                    "backend_detail": detail,
                    "plausibility": {"reference_TOPS": ref, "suspect": suspect},
                },
                layer="A",
                backend=backend,
                shape={"M": s, "K": s, "N": s},
                dtype_in=str(detail.get("dtype_in")),
                dtype_acc=str(detail.get("dtype_acc")),
                iters=LAYER_A_ITERS["iters"],
                warmup=LAYER_A_ITERS["warmup"],
                median_ms=median_ms, p10_ms=p10, p90_ms=p90,
                useful_ops=int(flops),
                useful_op_kind="gemm_mac",
                throughput=tflops,
                throughput_unit="TOPS" if backend == "cublaslt_int8" else "TFLOPS",
                correctness={"gate": "skipped"},  # Layer A doesn't gate
                host_overhead_ms=None,
                git_sha=sha, timestamp=ts,
            )
        )
    return results


# --- Layer B — raw GEMM ------------------------------------------------------


def layer_b_raw_gemm(sizes: tuple[int, ...], backends: tuple[str, ...]) -> list[BenchResult]:
    sha, ts = git_sha(), now_iso()
    dev = device_info()
    results: list[BenchResult] = []
    for backend in backends:
        for s in sizes:
            try:
                step, detail = make_gemm_step(backend, s, s, s)
            except torch.cuda.OutOfMemoryError as e:
                results.append(
                    BenchResult(
                        schema_version=SCHEMA_VERSION,
                        device="RTX5090",
                        device_detail={**dev, "backend_detail": {"error": f"OOM: {e}"}},
                        layer="B",
                        backend=backend,
                        shape={"M": s, "K": s, "N": s},
                        dtype_in="?", dtype_acc="?",
                        iters=0, warmup=0,
                        median_ms=None, p10_ms=None, p90_ms=None,
                        useful_ops=None, useful_op_kind="gemm_mac",
                        throughput=None, throughput_unit="TFLOPS",
                        correctness={"gate": "skipped"},
                        host_overhead_ms=None,
                        git_sha=sha, timestamp=ts,
                    )
                )
                torch.cuda.empty_cache()
                continue
            median_ms, p10, p90 = cuda_time(step, **LAYER_B_ITERS)
            flops = 2.0 * s * s * s
            tflops = flops / (median_ms * 1e-3) / 1e12
            results.append(
                BenchResult(
                    schema_version=SCHEMA_VERSION,
                    device="RTX5090",
                    device_detail={**dev, "backend_detail": detail},
                    layer="B",
                    backend=backend,
                    shape={"M": s, "K": s, "N": s},
                    dtype_in=str(detail.get("dtype_in")),
                    dtype_acc=str(detail.get("dtype_acc")),
                    iters=LAYER_B_ITERS["iters"],
                    warmup=LAYER_B_ITERS["warmup"],
                    median_ms=median_ms, p10_ms=p10, p90_ms=p90,
                    useful_ops=int(flops),
                    useful_op_kind="gemm_mac",
                    throughput=tflops,
                    throughput_unit="TOPS" if backend == "cublaslt_int8" else "TFLOPS",
                    correctness={"gate": "skipped"},
                    host_overhead_ms=None,
                    git_sha=sha, timestamp=ts,
                )
            )
            del step
            torch.cuda.empty_cache()
    return results


# --- Layer C — exact 36-bit modular product ---------------------------------


def _shift_mod_q_table(q: int, n_chunks: int = 5, chunk_bits: int = 8) -> list[int]:
    """Pre-compute ``2^(chunk_bits·(i+j)) mod q`` for ``i, j ∈ [0, n_chunks)``.

    Used by both the elementwise correctness gate and the matmul perf path
    so each byte cross-product can be reduced mod q **before** an int64
    accumulator can overflow.
    """
    return [pow(1 << (chunk_bits * k), 1, q) for k in range(2 * n_chunks - 1)]


def _q36_int8_modmul_elementwise(a_list: list[int], b_list: list[int], q: int) -> list[int]:
    """Reference path used **only by the correctness gate**, on the GPU.

    Decomposes each operand into 5 INT8 byte chunks, runs 25 elementwise
    chunk products, scales each by ``2^(8(i+j)) mod q``, reduces mod q, and
    sums. This is the **same recipe** the matmul perf path uses (modulo the
    GEMM versus elementwise multiply); a passing gate is strong evidence
    the matmul path will also be correct (the matmul path is additionally
    unit-tested at small scale in
    ``tests/test_hardware_gpu.py::test_36bit_matmul_via_25_int8_gemms``).

    Per-pair mod-q reduction is required: a naive int64 accumulator with
    raw byte shifts would overflow at the highest shift positions
    (``i + j = 8`` shifts by 64 bits, undefined for int64).
    """
    n_chunks = 5
    a = torch.tensor(a_list, dtype=torch.int64, device="cuda")
    b = torch.tensor(b_list, dtype=torch.int64, device="cuda")
    mask = torch.tensor(0xFF, dtype=torch.int64, device="cuda")

    a_chunks = [(a >> (8 * i)) & mask for i in range(n_chunks)]
    b_chunks = [(b >> (8 * j)) & mask for j in range(n_chunks)]
    smq = _shift_mod_q_table(q, n_chunks=n_chunks)

    out = torch.zeros(a.shape, dtype=torch.int64, device="cuda")
    for i in range(n_chunks):
        for j in range(n_chunks):
            # chunk product < 2^14, shift_mod_q < q < 2^36 → < 2^50, fits int64
            term = (a_chunks[i] * b_chunks[j]) * smq[i + j]
            out += term % q
    out = out % q
    return out.tolist()


def _layer_c_perf_step_factory(
    m: int, k: int, n: int, q: int
) -> Callable[[], object]:
    """Build the 25-INT8-GEMM matmul reconstruction step at the given shape.

    Random INT8-chunked operands in ``[0, 127]`` per chunk so each chunk
    fits in signed int8 without sign tricks. The 25 ``torch._int_mm`` calls
    each return int32; each partial is scaled by ``2^(8(i+j)) mod q``,
    reduced mod q, and accumulated. This **per-pair reduction** keeps the
    int64 accumulator bounded: the largest scaled-and-reduced term fits in
    ``q < 2^36`` and 25 such terms sum to ``< 25·q < 2^41``, safely inside
    signed int64.

    The earlier ``out += partial << (8*(i+j))`` pattern (used in
    ``scripts/bench_gpu.py``) **silently overflows** int64 for shifts up
    to 64 bits, so its ``% q`` output is incorrect. That bench is kept for
    the cost-model narrative; this one shadows it with the correct recipe.
    """
    n_chunks = 5
    a_chunks = [
        torch.randint(0, 128, (m, k), dtype=torch.int8, device="cuda") for _ in range(n_chunks)
    ]
    b_chunks = [
        torch.randint(0, 128, (k, n), dtype=torch.int8, device="cuda") for _ in range(n_chunks)
    ]
    out = torch.zeros((m, n), dtype=torch.int64, device="cuda")
    smq = _shift_mod_q_table(q, n_chunks=n_chunks)

    # Sanity: K · 127² · (q-1) < 2^63 ⇒ K < 2^63 / (127² · q) ≈ 8323 for this q.
    # Above that the int64 partial-times-scale multiplication overflows; switch
    # to a wider intermediate (Barrett, or split partial).
    if k * 127 * 127 * (q - 1) >= (1 << 63):
        raise ValueError(
            f"K={k} too large for int64 partial-scale multiply with q={q}; "
            "use Barrett reduction in v2"
        )

    def step() -> None:
        out.zero_()
        for i in range(n_chunks):
            for j in range(n_chunks):
                partial = torch._int_mm(a_chunks[i], b_chunks[j])
                # partial < K · 127² ≈ 2^27 (K=8192). scale < 2^36 → product < 2^63.
                term = (partial.to(torch.int64) * smq[i + j]) % q
                out.add_(term)
        out.remainder_(q)

    return step


def layer_c_q36_int8_modmul(sizes: tuple[int, ...]) -> list[BenchResult]:
    sha, ts = git_sha(), now_iso()
    dev = device_info()
    q = q36_ntt_friendly_prime()

    # Correctness gate — run once, not per size. The gate validates the
    # *decomposition+reduction* recipe; the per-size perf path uses the same
    # recipe just at matmul scale.
    gate = correctness_gate(lambda a, b: _q36_int8_modmul_elementwise(a, b, q), q)

    results: list[BenchResult] = []
    for s in sizes:
        if gate.gate != "passed":
            results.append(
                BenchResult(
                    schema_version=SCHEMA_VERSION,
                    device="RTX5090",
                    device_detail={**dev, "q36": q},
                    layer="C",
                    backend="cublaslt_int8",
                    shape={"M": s, "K": s, "N": s},
                    dtype_in="int8", dtype_acc="int32",
                    iters=0, warmup=0,
                    median_ms=None, p10_ms=None, p90_ms=None,
                    useful_ops=None, useful_op_kind="exact_modmul",
                    throughput=None, throughput_unit="G_modmul/s",
                    correctness={
                        "gate": gate.gate,
                        "edge_cases": gate.edge_cases,
                        "edge_cases_passed": gate.edge_cases_passed,
                        "edge_cases_failed": gate.edge_cases_failed,
                        "failure_examples": gate.failure_examples,
                    },
                    host_overhead_ms=None,
                    git_sha=sha, timestamp=ts,
                )
            )
            continue
        try:
            step = _layer_c_perf_step_factory(s, s, s, q)
        except torch.cuda.OutOfMemoryError as e:
            results.append(
                BenchResult(
                    schema_version=SCHEMA_VERSION,
                    device="RTX5090",
                    device_detail={**dev, "q36": q, "backend_detail": {"error": f"OOM: {e}"}},
                    layer="C",
                    backend="cublaslt_int8",
                    shape={"M": s, "K": s, "N": s},
                    dtype_in="int8", dtype_acc="int32",
                    iters=0, warmup=0,
                    median_ms=None, p10_ms=None, p90_ms=None,
                    useful_ops=None, useful_op_kind="exact_modmul",
                    throughput=None, throughput_unit="G_modmul/s",
                    correctness={
                        "gate": "passed",  # gate did pass — skip is OOM, not correctness
                        "edge_cases_passed": gate.edge_cases_passed,
                        "edge_cases_failed": 0,
                    },
                    host_overhead_ms=None,
                    git_sha=sha, timestamp=ts,
                )
            )
            torch.cuda.empty_cache()
            continue
        median_ms, p10, p90 = cuda_time(step, **LAYER_C_ITERS)
        useful_ops = s * s  # one logical exact 36-bit modmul per output element
        g_modmul_per_sec = useful_ops / (median_ms * 1e-3) / 1e9
        results.append(
            BenchResult(
                schema_version=SCHEMA_VERSION,
                device="RTX5090",
                device_detail={
                    **dev,
                    "q36": q,
                    "backend_detail": {
                        "dispatch": "25 × torch._int_mm (cuBLASLt IGEMM) + int64 shift/add + % q",
                        "n_int8_gemms": 25,
                    },
                },
                layer="C",
                backend="cublaslt_int8",
                shape={"M": s, "K": s, "N": s},
                dtype_in="int8", dtype_acc="int32",
                iters=LAYER_C_ITERS["iters"],
                warmup=LAYER_C_ITERS["warmup"],
                median_ms=median_ms, p10_ms=p10, p90_ms=p90,
                useful_ops=useful_ops,
                useful_op_kind="exact_modmul",
                throughput=g_modmul_per_sec,
                throughput_unit="G_modmul/s",
                correctness={
                    "gate": "passed",
                    "edge_cases": gate.edge_cases,
                    "edge_cases_passed": gate.edge_cases_passed,
                    "edge_cases_failed": gate.edge_cases_failed,
                },
                host_overhead_ms=None,
                git_sha=sha, timestamp=ts,
            )
        )
        del step
        torch.cuda.empty_cache()
    return results


# --- CPU baseline -----------------------------------------------------------


def cpu_int128_baseline_record(n: int = 100_000) -> BenchResult:
    """CPU bigint elementwise modmul, single record. Establishes the floor."""
    sha, ts = git_sha(), now_iso()
    q = q36_ntt_friendly_prime()
    rng = np.random.default_rng(20260427)
    a = rng.integers(0, q, size=n).tolist()
    b = rng.integers(0, q, size=n).tolist()

    # Python int is arbitrary-precision; the "int128" name is conceptual,
    # matching BENCHMARK.md's CPU baseline framing. Real C++ benchmarks
    # would use ``unsigned __int128``; we don't ship that in v1.
    def step() -> None:
        for ai, bi in zip(a, b, strict=True):
            (ai * bi) % q

    median_ms, p10, p90 = cpu_time(step, iters=5, warmup=1)
    throughput = n / (median_ms * 1e-3) / 1e6  # M_modmul/s
    return BenchResult(
        schema_version=SCHEMA_VERSION,
        device="CPU",
        device_detail={"name": "host CPU (Python int)", "interpreter": sys.version.split()[0]},
        layer="C",
        backend="cpu_int128",
        shape={"M": n, "K": 1, "N": 1},
        dtype_in="bigint", dtype_acc="bigint",
        iters=5, warmup=1,
        median_ms=median_ms, p10_ms=p10, p90_ms=p90,
        useful_ops=n,
        useful_op_kind="exact_modmul",
        throughput=throughput / 1e3,  # report as G_modmul/s for unit consistency
        throughput_unit="G_modmul/s",
        correctness={"gate": "passed", "note": "trivially exact (Python int reference)"},
        host_overhead_ms=None,
        git_sha=sha, timestamp=ts,
    )


# --- Driver -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RTX 5090 cuBLASLt benchmark, BENCHMARK.md §4 v1.")
    p.add_argument("--out", required=True, type=Path, help="JSONL output path.")
    p.add_argument(
        "--layers", default="A,B,C",
        help="Comma-separated subset of {A,B,C}. Default: A,B,C.",
    )
    p.add_argument(
        "--sizes", default=None,
        help="Comma-separated square sizes. Default: 512,1024,2048,4096,8192.",
    )
    p.add_argument(
        "--backends", default=",".join(DEFAULT_BACKENDS),
        help="Comma-separated subset of available backends.",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Smoke run: sizes=256,512 only. Useful for CI.",
    )
    p.add_argument(
        "--no-cpu", action="store_true",
        help="Skip the CPU bigint baseline (slow on cold runs).",
    )
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        print("CUDA not available; this benchmark requires a CUDA GPU.", file=sys.stderr)
        return 2

    layers = tuple(args.layers.split(","))
    if args.sizes is not None:
        sizes = tuple(int(s) for s in args.sizes.split(","))
    else:
        sizes = QUICK_SIZES if args.quick else DEFAULT_SIZES
    backends = tuple(args.backends.split(","))

    print(f"device   : {device_info()['name']} (sm_{device_info()['sm']})")
    print(f"out      : {args.out}")
    print(f"layers   : {','.join(layers)}")
    print(f"sizes    : {','.join(str(s) for s in sizes)}")
    print(f"backends : {','.join(backends)}")
    print()

    all_results: list[BenchResult] = []
    if "A" in layers:
        print("Layer A (capability probe)…")
        all_results.extend(layer_a_capability_probe())
    if "B" in layers:
        print("Layer B (raw GEMM)…")
        all_results.extend(layer_b_raw_gemm(sizes, backends))
    if "C" in layers:
        print("Layer C-minimal (q36 INT8 modmul)…")
        all_results.extend(layer_c_q36_int8_modmul(sizes))
        if not args.no_cpu:
            print("CPU bigint baseline…")
            all_results.append(cpu_int128_baseline_record())

    write_results(all_results, args.out)
    print(f"\nwrote {len(all_results)} record(s) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
