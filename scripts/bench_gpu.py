"""GPU microbenchmark for the notebook series cost models.

Times four GPU paths that correspond to the central narratives of the
educational notebooks:

1. ``int64`` elementwise ``(a * b) % q`` — the GPU analogue of the CPU
   31-bit RNS path (Notebook 3 §3.3 Method 1), running on CUDA cores.
2. BF16 Tensor-Core GEMM — floating-point baseline. Establishes the
   *natural* throughput shape of the device's Tensor Cores.
3. INT8 → INT32 Tensor-Core GEMM via ``torch._int_mm`` — the hardware
   primitive Notebook 4 builds on.
4. 36-bit-equivalent matmul reconstructed as 25 INT8 GEMMs — the actual
   recipe for exact 36-bit residue matmul on Tensor Cores
   (Notebook 4 §4.3).

The headline number reported is the ratio of (3) to (4): how close the
empirical 36-bit-via-INT8 path comes to the cost model's predicted 25×
overhead. PLAN.md §4.5 argues INT8 chunking only wins when the device's
INT8 advantage exceeds 25× — this script measures exactly that ratio
on the host hardware.

Run with::

    uv run --extra bench python scripts/bench_gpu.py

Throughput is reported using CUDA-event timing with warm-up; the median
of multiple iterations is reported to mitigate clock-throttling jitter.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BenchResult:
    """One row of the benchmark output table."""

    name: str
    problem_size: str
    iters: int
    median_ms: float
    throughput: float
    throughput_unit: str


def cuda_time(fn: Callable[[], object], iters: int, warmup: int = 5) -> list[float]:
    """Return per-iteration GPU time in milliseconds, sorted ascending.

    Uses CUDA events for true device-side timing and runs ``warmup``
    untimed iterations first to stabilize clocks and JIT caches.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))


def bench_int64_rns_mulmod(n: int = 100_000_000, iters: int = 30) -> BenchResult:
    """31-bit RNS analogue: ``(a * b) % q`` over int64 on CUDA cores."""
    q = (1 << 31) - 1
    rng = torch.Generator(device="cpu").manual_seed(20260427)
    a = torch.randint(0, q, (n,), dtype=torch.int64, generator=rng).cuda()
    b = torch.randint(0, q, (n,), dtype=torch.int64, generator=rng).cuda()
    out = torch.empty_like(a)

    def step() -> None:
        torch.remainder(a * b, q, out=out)

    times = cuda_time(step, iters)
    median_ms = statistics.median(times)
    gops = n / (median_ms * 1e-3) / 1e9
    return BenchResult(
        name="int64_31bit_rns_mulmod",
        problem_size=f"n={n:,}",
        iters=iters,
        median_ms=median_ms,
        throughput=gops,
        throughput_unit="Gops/s",
    )


def bench_bf16_gemm(m: int = 4096, k: int = 4096, n: int = 4096, iters: int = 50) -> BenchResult:
    """BF16 Tensor-Core GEMM as a floating-point baseline."""
    a = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(k, n, dtype=torch.bfloat16, device="cuda")

    def step() -> None:
        torch.matmul(a, b)

    times = cuda_time(step, iters)
    median_ms = statistics.median(times)
    flops = 2.0 * m * k * n
    tflops = flops / (median_ms * 1e-3) / 1e12
    return BenchResult(
        name="bf16_tensor_core_gemm",
        problem_size=f"{m}x{k}x{n}",
        iters=iters,
        median_ms=median_ms,
        throughput=tflops,
        throughput_unit="TFLOPS",
    )


def bench_int8_gemm(m: int = 4096, k: int = 4096, n: int = 4096, iters: int = 50) -> BenchResult:
    """INT8 → INT32 Tensor-Core GEMM via ``torch._int_mm``."""
    a = torch.randint(-128, 127, (m, k), dtype=torch.int8, device="cuda")
    b = torch.randint(-128, 127, (k, n), dtype=torch.int8, device="cuda")

    def step() -> None:
        torch._int_mm(a, b)

    times = cuda_time(step, iters)
    median_ms = statistics.median(times)
    ops = 2.0 * m * k * n
    tops = ops / (median_ms * 1e-3) / 1e12
    return BenchResult(
        name="int8_int32_tensor_core_gemm",
        problem_size=f"{m}x{k}x{n}",
        iters=iters,
        median_ms=median_ms,
        throughput=tops,
        throughput_unit="TOPS",
    )


def bench_36bit_via_25_int8_gemms(
    m: int = 4096, k: int = 4096, n: int = 4096, iters: int = 20
) -> BenchResult:
    """Reconstruct an exact 36-bit-equivalent matmul as 25 INT8 GEMMs.

    Throughput is reported as if it were a single ``m × k × n`` matmul
    (i.e. the conceptual workload is one 36-bit-residue matmul), so the
    number is directly comparable to the raw INT8 GEMM result. The ratio
    of those two numbers is the empirical version of Notebook 4's 25×
    cost-model overhead.
    """
    n_chunks = 5  # 36 bits / 8 bits-per-chunk → 5 byte chunks
    a_chunks = [
        torch.randint(0, 128, (m, k), dtype=torch.int8, device="cuda") for _ in range(n_chunks)
    ]
    b_chunks = [
        torch.randint(0, 128, (k, n), dtype=torch.int8, device="cuda") for _ in range(n_chunks)
    ]
    out = torch.zeros((m, n), dtype=torch.int64, device="cuda")

    def step() -> None:
        out.zero_()
        for i in range(n_chunks):
            for j in range(n_chunks):
                partial = torch._int_mm(a_chunks[i], b_chunks[j])
                out.add_(partial.to(torch.int64) << (8 * (i + j)))

    times = cuda_time(step, iters)
    median_ms = statistics.median(times)
    ops = 2.0 * m * k * n  # one *logical* 36-bit-MAC matmul
    teff = ops / (median_ms * 1e-3) / 1e12
    return BenchResult(
        name="36bit_matmul_via_25_int8_gemms",
        problem_size=f"{m}x{k}x{n}",
        iters=iters,
        median_ms=median_ms,
        throughput=teff,
        throughput_unit="T(36b-MAC)/s",
    )


def _print_header() -> None:
    name = torch.cuda.get_device_name(0)
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    sm = 10 * cap_major + cap_minor
    print(f"Device   : {name} (sm_{sm})")
    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : runtime {torch.version.cuda}")
    print()


def _print_table(results: list[BenchResult]) -> None:
    name_w = max(len(r.name) for r in results)
    size_w = max(len(r.problem_size) for r in results)
    header = (
        f"{'name':<{name_w}}  {'size':<{size_w}}  "
        f"{'iters':>5}  {'median_ms':>10}  {'throughput':>16}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        thr = f"{r.throughput:>9.2f} {r.throughput_unit}"
        print(
            f"{r.name:<{name_w}}  {r.problem_size:<{size_w}}  "
            f"{r.iters:>5}  {r.median_ms:>10.3f}  {thr:>16}"
        )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this benchmark requires a CUDA GPU.")
    _print_header()

    results = [
        bench_int64_rns_mulmod(),
        bench_bf16_gemm(),
        bench_int8_gemm(),
        bench_36bit_via_25_int8_gemms(),
    ]
    _print_table(results)

    int8 = next(r for r in results if r.name == "int8_int32_tensor_core_gemm")
    bit36 = next(r for r in results if r.name == "36bit_matmul_via_25_int8_gemms")
    ratio = int8.throughput / bit36.throughput
    predicted = 25.0  # PLAN.md §4.5: 5×5 byte cross-products
    print()
    print(
        f"INT8 raw / 36-bit-equivalent throughput ratio: {ratio:.2f}x  "
        f"(cost-model prediction: {predicted:.0f}x; "
        f"empirical overhead {ratio / predicted:.2f}x of model)"
    )


if __name__ == "__main__":
    main()
