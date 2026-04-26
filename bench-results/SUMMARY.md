# Bench summary — RTX 5090 vs TT Blackhole

- **Schema version**: 1
- **Devices**: CPU, RTX5090
- **Records**: 30
- **Timestamps**: 2026-04-26T17:45:10+00:00 – 2026-04-26T17:46:03+00:00
- **Git SHAs**: fa2e772

## Device prices (declared, not detected)

| device | price (USD) |
|---|---|
| CPU | (n/a) |
| RTX5090 | 1999 |

Update :data:`DEVICE_PRICES_USD` in `scripts/_bench_common.py` if these are stale.

## Layer A — capability probe
### Layer A

| op_kind | shape | backend | RTX5090 thr | RTX5090 /k$ | RTX5090 gate |
|---|---|---|---|---|---|
| gemm_mac | 1024×1024×1024 | fp32 | 48.19 TFLOPS | 24.109 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | fp64 | 1.49 TFLOPS | 0.747 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | int8 | 76.35 TOPS | 38.193 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | tf32 | 76.87 TFLOPS | 38.455 /k$ | skipped |

## Layer B — raw GEMM
### Layer B

| op_kind | shape | backend | RTX5090 thr | RTX5090 /k$ | RTX5090 gate |
|---|---|---|---|---|---|
| gemm_mac | 1024×1024×1024 | fp32 | 48.06 TFLOPS | 24.040 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | fp64 | 1.45 TFLOPS | 0.727 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | int8 | 76.43 TOPS | 38.236 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | tf32 | 76.96 TFLOPS | 38.499 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | fp32 | 67.75 TFLOPS | 33.893 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | fp64 | 1.38 TFLOPS | 0.689 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | int8 | 179.74 TOPS | 89.913 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | tf32 | 89.43 TFLOPS | 44.736 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | fp32 | 58.57 TFLOPS | 29.299 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | fp64 | 1.49 TFLOPS | 0.747 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | int8 | 214.53 TOPS | 107.321 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | tf32 | 102.59 TFLOPS | 51.319 /k$ | skipped |
| gemm_mac | 512×512×512 | fp32 | 8.90 TFLOPS | 4.450 /k$ | skipped |
| gemm_mac | 512×512×512 | fp64 | 1.02 TFLOPS | 0.510 /k$ | skipped |
| gemm_mac | 512×512×512 | int8 | 17.15 TOPS | 8.582 /k$ | skipped |
| gemm_mac | 512×512×512 | tf32 | 9.75 TFLOPS | 4.880 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | fp32 | 61.69 TFLOPS | 30.862 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | fp64 | 1.58 TFLOPS | 0.792 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | int8 | 210.11 TOPS | 105.105 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | tf32 | 100.87 TFLOPS | 50.462 /k$ | skipped |

## Layer C — exact 36-bit modular product
### Layer C

| op_kind | shape | backend | CPU thr | CPU /k$ | CPU gate | RTX5090 thr | RTX5090 /k$ | RTX5090 gate |
|---|---|---|---|---|---|---|---|---|
| exact_modmul | 100000×1×1 | cpu | 0.01 G_modmul/s | — | passed | — | — | — |
| exact_modmul | 1024×1024×1024 | int8 | — | — | — | 0.84 G_modmul/s | 0.422 /k$ | passed |
| exact_modmul | 2048×2048×2048 | int8 | — | — | — | 0.85 G_modmul/s | 0.426 /k$ | passed |
| exact_modmul | 4096×4096×4096 | int8 | — | — | — | 0.45 G_modmul/s | 0.224 /k$ | passed |
| exact_modmul | 512×512×512 | int8 | — | — | — | 0.34 G_modmul/s | 0.169 /k$ | passed |
| exact_modmul | 8192×8192×8192 | int8 | — | — | — | 0.33 G_modmul/s | 0.163 /k$ | passed |

**Cross-device throughput ratios** (higher = first device wins):

| op_kind | shape | backend | CPU / RTX5090 | price-adjusted (CPU /k$) / (RTX5090 /k$) |
|---|---|---|---|---|

## How to update

```
# NVIDIA host (this machine):
uv run --extra bench python scripts/bench_nvidia.py \
    --out bench-results/nvidia_$(git rev-parse --short HEAD).jsonl

# TT host (separate machine, see BENCHMARK_TT.md):
python tt-llk-skeleton/bench_blackhole.py \
    --out bench-results/blackhole_$(git rev-parse --short HEAD).jsonl

# Then back on this machine:
python scripts/compare.py bench-results/*.jsonl --out bench-results/SUMMARY.md
```
