# `doc/` — narrative companion to the bench

The artifacts in this repo come in three layers:

- [`PLAN.md`](../PLAN.md) and [`BENCHMARK.md`](../BENCHMARK.md) — design
  documents. What we set out to build and why.
- [`scripts/`](../scripts/), [`tt-llk-skeleton/`](../tt-llk-skeleton/),
  [`bench-results/SUMMARY.md`](../bench-results/SUMMARY.md), and the
  PNG plots — the executable artefacts and the raw numbers.
- This directory — prose explanations in plain English. Reading order:

  1. [`01_rationale.md`](01_rationale.md) — why this benchmark exists.
     What FHE actually does on the hardware, why 36-bit modular
     multiplication is the bottleneck, why the comparison is between
     RTX 5090 and TT Blackhole specifically, and what each of the five
     layers (A–E) measures.
  2. [`02_findings.md`](02_findings.md) — the numbers we have, and
     what they mean. Per-layer interpretation, the price-adjusted
     comparison, and the small set of headline conclusions.
  3. [`03_caveats.md`](03_caveats.md) — what's missing, what's
     deferred to v2, and the methodology limits a reader should keep
     in mind before quoting any of these numbers.
  4. [`04_phase7_tuned_matmul.md`](04_phase7_tuned_matmul.md) — the
     Phase 7 reproduction of TT-Metal's upstream GEMM_FLOPS benchmark.
     Why our v1+v2 reference Layer B numbers were ~37–70× under the
     matrix engine's actual capability, what it took to unblock the
     reproduction (firmware ↔ tt-metal ↔ ttnn three-way mismatch and
     the side-by-side rebuild that fixed it), and how the tuned
     numbers reframe every cross-device ratio in `02_findings.md`.

The [`discussions/`](discussions/) subdirectory holds **open analytical
sidebars** — caveats, asymmetries, follow-up experiments — that aren't
yet part of the main numbered narrative but should travel with the
benchmark. Each entry is a self-contained markdown file with a "Status"
line at the top:

- [`discussions/nvidia_cublaslt_int8_utilization.md`](discussions/nvidia_cublaslt_int8_utilization.md)
  — why the §4.5.4 RTX 5090 INT8 number (215 TOPS) sits at only ~51 %
  of dense Tensor Core peak, why that's the expected `torch._int_mm`
  utilization, and the asymmetry vs the TT mcast number sitting at
  ~83 % of its matrix-engine ceiling. Decides between re-running with
  CUTLASS vs documenting the caveat in §4.5.4.

The numbers cited here are reproducible from the JSONL files in
[`bench-results/`](../bench-results/). When the data is refreshed (new
runs, new backends), the prose should be re-checked against
[`SUMMARY.md`](../bench-results/SUMMARY.md) — that file is the
authoritative scoreboard, this directory is the explanation.
