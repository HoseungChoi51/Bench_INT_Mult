"""Shared infrastructure for the head-to-head bench scripts.

The two bench drivers — ``bench_nvidia.py`` (RTX 5090, cuBLASLt via nvmath +
``torch._int_mm``) and the TT-side ``bench_blackhole.py`` (TT-LLK) — must
emit the **same JSON Lines schema** so ``compare.py`` can join their outputs
on ``(layer, useful_op_kind, shape)``. This module owns:

- the ``BenchResult`` dataclass (the schema, frozen)
- a JSONL writer
- timing helpers (CUDA-event median for the GPU, ``perf_counter`` for the host)
- the bit-exact correctness gate utility for Layer C
- a small device-price registry used by ``compare.py`` for the price-adjusted
  ratio columns (declared, not detected)

See ``BENCHMARK.md`` Section 4 for the layer specifications and
``/home/chs/.claude/plans/i-have-added-benchmark-md-happy-galaxy.md`` for
the v1 plan this file was written against.
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = "2"

# Schema history:
#   v1 (initial)   — A/B/C layers, gemm_mac / exact_modmul / klss_mac op_kinds.
#   v2 (TT-side)   — adds Layer D (klss_ip_modmul); splits exact_modmul into
#                    q36 / q48 variants; adds joules/power telemetry. v2 is a
#                    strict superset of v1, so old records are still valid.
Layer = Literal["A", "B", "C", "D"]
UsefulOpKind = Literal[
    "gemm_mac",
    "exact_modmul",       # legacy alias; v1 records use this for q36
    "exact_modmul_q36",   # 36-bit prime → 25 INT8 GEMMs
    "exact_modmul_q48",   # 48-bit prime → 36 INT8 GEMMs (v2 Phase 2)
    "klss_mac",
    "klss_ip_modmul",
    "klss_ip_modmul_q36",
    "klss_ip_modmul_q48",
]
GateOutcome = Literal["passed", "skipped", "failed"]


@dataclass(frozen=True)
class CorrectnessReport:
    """Result of running an adversarial test set against a reference."""

    gate: GateOutcome
    edge_cases: list[str] = field(default_factory=list)
    edge_cases_passed: int = 0
    edge_cases_failed: int = 0
    failure_examples: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BenchResult:
    """One JSONL record. Schema is frozen at ``SCHEMA_VERSION``.

    All fields whose value is unknown / not-applicable should be ``None`` so
    consumers can distinguish "not measured" from "measured to zero". The
    JSON serializer preserves nulls.
    """

    schema_version: str
    device: str
    device_detail: dict[str, Any]
    layer: Layer
    backend: str
    shape: dict[str, int]
    dtype_in: str
    dtype_acc: str
    iters: int
    warmup: int
    median_ms: float | None
    p10_ms: float | None
    p90_ms: float | None
    useful_ops: int | None
    useful_op_kind: UsefulOpKind
    throughput: float | None
    throughput_unit: str
    correctness: dict[str, Any]
    host_overhead_ms: float | None
    git_sha: str
    timestamp: str
    # v2 additions (optional — older v1 records leave these as None).
    # power_w_avg: pre/post snapshot averaged board power in watts during
    # the timed loop. None when telemetry was unavailable (no GPU NVML,
    # no tt-smi, etc.). joules_per_useful_op is the derived metric
    # (power_w_avg * median_s / useful_ops).
    power_w_avg: float | None = None
    joules_per_useful_op: float | None = None

    def to_jsonl_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def write_results(results: Iterable[BenchResult], path: Path) -> None:
    """Append-style writer. Creates parent dir; one record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r.to_jsonl_line() + "\n")


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or "unknown"
    except FileNotFoundError:
        return "unknown"


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# --- Timing -----------------------------------------------------------------


def cuda_time(
    fn: Callable[[], object],
    iters: int,
    warmup: int = 5,
) -> tuple[float, float, float]:
    """Time ``fn`` on the GPU via CUDA events; return ``(median, p10, p90)`` ms.

    ``fn`` must enqueue work on the current CUDA stream. Warmup runs are
    untimed; ``torch.cuda.synchronize()`` is called before timing starts.
    """
    import torch  # local import: scripts that don't need CUDA shouldn't pay this cost

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

    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends, strict=True))
    return _percentiles(times)


def cpu_time(
    fn: Callable[[], object],
    iters: int,
    warmup: int = 1,
) -> tuple[float, float, float]:
    """Time ``fn`` on the host via ``perf_counter``; return ``(median, p10, p90)`` ms."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return _percentiles(samples)


def _percentiles(sorted_times_ms: Sequence[float]) -> tuple[float, float, float]:
    n = len(sorted_times_ms)
    if n == 0:
        raise ValueError("no samples")
    median = statistics.median(sorted_times_ms)
    p10 = sorted_times_ms[max(0, int(0.10 * n) - 1)]
    p90 = sorted_times_ms[min(n - 1, int(0.90 * n))]
    return median, p10, p90


# --- Correctness gate -------------------------------------------------------


def adversarial_modmul_inputs(
    q: int, n_random: int = 1000, seed: int = 20260427
) -> list[tuple[int, int, str]]:
    """Build the Section-4 Layer-C adversarial input set for ``c = (a*b) mod q``.

    Returns a list of ``(a, b, label)`` triples covering:
    - ``0, 1, q-1, q-2``
    - ``2^k ± 1`` for ``k`` near the operand bit width
    - ``n_random`` random pairs in ``[0, q)``
    - 100 near-quotient-boundary pairs (products near a multiple of ``q``)

    The test harness must produce ``(a*b) mod q`` bit-exactly against the
    Python ``int`` reference for all of these before any perf record is
    emitted for that backend/size.
    """
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")
    bit_width = q.bit_length()

    out: list[tuple[int, int, str]] = []
    boundary_vals = [0, 1, q - 1, q - 2]
    for a in boundary_vals:
        for b in boundary_vals:
            out.append((a, b, f"boundary({a},{b})"))

    for k in (bit_width - 6, bit_width - 4, bit_width - 2, bit_width - 1):
        if k <= 0:
            continue
        for sign in (-1, +1):
            v = (1 << k) + sign
            if 0 <= v < q:
                out.append((v, v, f"2^{k}{'+' if sign>0 else '-'}1_squared"))
                out.append((v, q - 1, f"2^{k}{'+' if sign>0 else '-'}1_x_q-1"))

    import random  # local: only correctness gate uses RNG
    rng = random.Random(seed)
    for i in range(n_random):
        out.append((rng.randrange(0, q), rng.randrange(0, q), f"random[{i}]"))

    # Near-quotient-boundary: choose m such that m*q ± δ has the right shape,
    # then factor approximately by picking a near sqrt(m*q).
    for i in range(100):
        m = rng.randrange(1, 1 << 16)
        delta = rng.randrange(-1024, 1024)
        target = m * q + delta
        if target <= 0:
            continue
        a = max(1, min(q - 1, int(target**0.5)))
        b = max(1, min(q - 1, target // max(a, 1)))
        out.append((a, b, f"near_quotient[{i},m={m},δ={delta}]"))

    return out


def correctness_gate(
    impl_fn: Callable[[list[int], list[int]], list[int]],
    q: int,
    n_random: int = 1000,
) -> CorrectnessReport:
    """Run the adversarial set through ``impl_fn`` and compare against bigint truth.

    ``impl_fn(a_list, b_list) -> c_list`` must compute ``(a*b) mod q``
    elementwise. This wrapper batches all adversarial pairs into one call so
    GPU implementations can amortize launch overhead, but the per-element
    comparison is bit-exact.
    """
    cases = adversarial_modmul_inputs(q, n_random=n_random)
    a_in = [a for a, _, _ in cases]
    b_in = [b for _, b, _ in cases]

    c_out = impl_fn(a_in, b_in)
    if len(c_out) != len(cases):
        raise RuntimeError(f"impl_fn returned {len(c_out)} values, expected {len(cases)}")

    failures: list[dict[str, Any]] = []
    for (a, b, label), c in zip(cases, c_out, strict=True):
        truth = (a * b) % q
        if int(c) != truth and len(failures) < 8:  # cap the noise; the count is what matters
            failures.append({"label": label, "a": a, "b": b, "got": int(c), "want": truth})

    n_passed = len(cases) - len(failures)
    edge_labels_seen = sorted({lbl.split("[")[0].split("(")[0] for _, _, lbl in cases})
    return CorrectnessReport(
        gate="passed" if not failures else "failed",
        edge_cases=edge_labels_seen,
        edge_cases_passed=n_passed,
        edge_cases_failed=len(cases) - n_passed,
        failure_examples=failures,
    )


# --- Headline-prime selection ----------------------------------------------


def q36_ntt_friendly_prime() -> int:
    """The largest prime ``q < 2^36`` with ``q ≡ 1 (mod 2^16)``.

    The congruence guarantees a primitive 2^16-th root of unity mod ``q``,
    the typical ring-degree alignment for CKKS NTT. Any backend hitting
    this prime in Layer C must produce bit-exact results for the
    adversarial set in :func:`adversarial_modmul_inputs`.

    Value: ``0xFFFF00001 = 68_718_428_161 = 1_048_560 · 2^16 + 1``,
    36 bits wide, primality verified by deterministic Miller-Rabin.
    """
    return 0xFFFF00001


def q48_ntt_friendly_prime() -> int:
    """The largest prime ``q < 2^48`` with ``q ≡ 1 (mod 2^16)``.

    Same NTT-friendliness guarantee as :func:`q36_ntt_friendly_prime`;
    used by the v2 Layer C extension to measure the cost of a wider
    operand. 48-bit operands decompose into 6 bytes, so a single modmul
    becomes 6×6 = 36 INT8 GEMMs (vs 25 for q36 — a 1.44× cost factor).

    Value: ``0xFFFFFFFA0001 = 281_474_976_317_441 = 4_294_967_290 · 2^16 + 1``,
    48 bits wide, primality verified by deterministic Miller-Rabin.
    """
    return 0xFFFFFFFA0001


# --- Power telemetry --------------------------------------------------------


def read_tt_power_w() -> float | None:
    """Snapshot board power (W) from tt-smi. Returns None if unavailable.

    `tt-smi -s` dumps the SMBus telemetry block as JSON to stdout. The TDP
    field is a hex-encoded watts value (e.g. ``"0x2d" → 45 W``). We parse
    the first device's TDP. Cheap (~30 ms invocation cost), accurate
    enough for pre/post-loop snapshots that span seconds.
    """
    try:
        out = subprocess.run(
            ["tt-smi", "-s"], capture_output=True, text=True, check=False, timeout=10
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        data = json.loads(out.stdout)
        devices = data.get("device_info") or []
        if not devices:
            return None
        tdp_hex = devices[0].get("smbus_telem", {}).get("TDP")
        if tdp_hex is None:
            return None
        return float(int(tdp_hex, 16))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return None


def read_nvidia_power_w(device_index: int = 0) -> float | None:
    """Snapshot board power (W) from NVML. Returns None if pynvml unavailable.

    nvmlDeviceGetPowerUsage returns mW, we convert to W. Identical
    pre/post snapshot model as :func:`read_tt_power_w`. Falls back to
    `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits`
    if pynvml isn't importable (more portable on minimal CUDA installs).
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            return float(pynvml.nvmlDeviceGetPowerUsage(h)) / 1000.0
        finally:
            pynvml.nvmlShutdown()
    except (ImportError, Exception):  # noqa: BLE001 — NVMLError is dynamic
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return float(out.stdout.strip().split()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


# --- Device price registry -------------------------------------------------

DEVICE_PRICES_USD: dict[str, float] = {
    # Declared, not detected. Keep these as MSRP-at-time-of-bench so the
    # comparison is reproducible. Update with a comment if pricing changes.
    "RTX5090": 1999.0,        # NVIDIA GeForce RTX 5090 launch MSRP, 2025
    "Blackhole": 999.0,       # Tenstorrent Blackhole p150a, 2025 list price
    "CPU": 0.0,               # CPU baseline is "free" (already in machine);
                              # ratio columns involving CPU are nonsense
                              # and compare.py blanks them.
}
