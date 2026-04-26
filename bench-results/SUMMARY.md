# Bench summary — RTX 5090 vs TT Blackhole

- **Schema version**: 1
- **Devices**: Blackhole, CPU, RTX5090
- **Records**: 53
- **Timestamps**: 2026-04-26T17:45:10+00:00 – 2026-04-26T19:10:26+00:00
- **Git SHAs**: dbe3cc1, fa2e772

## Device prices (declared, not detected)

| device | price (USD) |
|---|---|
| Blackhole | 999 |
| CPU | (n/a) |
| RTX5090 | 1999 |

Update :data:`DEVICE_PRICES_USD` in `scripts/_bench_common.py` if these are stale.

## Layer A — capability probe
### Layer A

| op_kind | shape | backend | Blackhole thr | Blackhole /k$ | Blackhole gate | RTX5090 thr | RTX5090 /k$ | RTX5090 gate |
|---|---|---|---|---|---|---|---|---|
| gemm_mac | 1024×1024×1024 | bf16 | 3.65 TFLOPS | 3.658 /k$ | skipped | — | — | — |
| gemm_mac | 1024×1024×1024 | fp32 | — | — | skipped | 48.19 TFLOPS | 24.109 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | fp64 | — | — | — | 1.49 TFLOPS | 0.747 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | int8 | 6.48 TOPS | 6.490 /k$ | skipped | 76.35 TOPS | 38.193 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | tf32 | — | — | — | 76.87 TFLOPS | 38.455 /k$ | skipped |

**Cross-device throughput ratios** (higher = first device wins):

| op_kind | shape | backend | Blackhole / RTX5090 | price-adjusted (Blackhole /k$) / (RTX5090 /k$) |
|---|---|---|---|---|
| gemm_mac | 1024×1024×1024 | int8 | 0.08× | 0.17× |

## Layer B — raw GEMM
### Layer B

| op_kind | shape | backend | Blackhole thr | Blackhole /k$ | Blackhole gate | RTX5090 thr | RTX5090 /k$ | RTX5090 gate |
|---|---|---|---|---|---|---|---|---|
| gemm_mac | 1024×1024×1024 | bf16 | 3.66 TFLOPS | 3.659 /k$ | skipped | — | — | — |
| gemm_mac | 1024×1024×1024 | fp32 | — | — | skipped | 48.06 TFLOPS | 24.040 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | fp64 | — | — | — | 1.45 TFLOPS | 0.727 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | int8 | 6.48 TOPS | 6.489 /k$ | skipped | 76.43 TOPS | 38.236 /k$ | skipped |
| gemm_mac | 1024×1024×1024 | tf32 | — | — | — | 76.96 TFLOPS | 38.499 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | bf16 | 3.84 TFLOPS | 3.845 /k$ | skipped | — | — | — |
| gemm_mac | 2048×2048×2048 | fp32 | — | — | skipped | 67.75 TFLOPS | 33.893 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | fp64 | — | — | — | 1.38 TFLOPS | 0.689 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | int8 | 7.32 TOPS | 7.331 /k$ | skipped | 179.74 TOPS | 89.913 /k$ | skipped |
| gemm_mac | 2048×2048×2048 | tf32 | — | — | — | 89.43 TFLOPS | 44.736 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | bf16 | 3.91 TFLOPS | 3.914 /k$ | skipped | — | — | — |
| gemm_mac | 4096×4096×4096 | fp32 | — | — | skipped | 58.57 TFLOPS | 29.299 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | fp64 | — | — | — | 1.49 TFLOPS | 0.747 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | int8 | 7.57 TOPS | 7.580 /k$ | skipped | 214.53 TOPS | 107.321 /k$ | skipped |
| gemm_mac | 4096×4096×4096 | tf32 | — | — | — | 102.59 TFLOPS | 51.319 /k$ | skipped |
| gemm_mac | 512×512×512 | bf16 | 2.48 TFLOPS | 2.480 /k$ | skipped | — | — | — |
| gemm_mac | 512×512×512 | fp32 | — | — | skipped | 8.90 TFLOPS | 4.450 /k$ | skipped |
| gemm_mac | 512×512×512 | fp64 | — | — | — | 1.02 TFLOPS | 0.510 /k$ | skipped |
| gemm_mac | 512×512×512 | int8 | 3.66 TOPS | 3.666 /k$ | skipped | 17.15 TOPS | 8.582 /k$ | skipped |
| gemm_mac | 512×512×512 | tf32 | — | — | — | 9.75 TFLOPS | 4.880 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | bf16 | 3.89 TFLOPS | 3.894 /k$ | skipped | — | — | — |
| gemm_mac | 8192×8192×8192 | fp32 | — | — | skipped | 61.69 TFLOPS | 30.862 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | fp64 | — | — | — | 1.58 TFLOPS | 0.792 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | int8 | 7.39 TOPS | 7.396 /k$ | skipped | 210.11 TOPS | 105.105 /k$ | skipped |
| gemm_mac | 8192×8192×8192 | tf32 | — | — | — | 100.87 TFLOPS | 50.462 /k$ | skipped |

**Cross-device throughput ratios** (higher = first device wins):

| op_kind | shape | backend | Blackhole / RTX5090 | price-adjusted (Blackhole /k$) / (RTX5090 /k$) |
|---|---|---|---|---|
| gemm_mac | 1024×1024×1024 | int8 | 0.08× | 0.17× |
| gemm_mac | 2048×2048×2048 | int8 | 0.04× | 0.08× |
| gemm_mac | 4096×4096×4096 | int8 | 0.04× | 0.07× |
| gemm_mac | 512×512×512 | int8 | 0.21× | 0.43× |
| gemm_mac | 8192×8192×8192 | int8 | 0.04× | 0.07× |

## Layer C — exact 36-bit modular product
### Layer C

| op_kind | shape | backend | Blackhole thr | Blackhole /k$ | Blackhole gate | CPU thr | CPU /k$ | CPU gate | RTX5090 thr | RTX5090 /k$ | RTX5090 gate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_modmul | 100000×1×1 | cpu | — | — | — | 0.01 G_modmul/s | — | passed | — | — | — |
| exact_modmul | 1024×1024×1024 | int8 | 0.14 G_modmul/s | 0.136 /k$ | passed | — | — | — | 0.84 G_modmul/s | 0.422 /k$ | passed |
| exact_modmul | 2048×2048×2048 | int8 | 0.07 G_modmul/s | 0.073 /k$ | passed | — | — | — | 0.85 G_modmul/s | 0.426 /k$ | passed |
| exact_modmul | 4096×4096×4096 | int8 | 0.04 G_modmul/s | 0.037 /k$ | passed | — | — | — | 0.45 G_modmul/s | 0.224 /k$ | passed |
| exact_modmul | 512×512×512 | int8 | 0.21 G_modmul/s | 0.208 /k$ | passed | — | — | — | 0.34 G_modmul/s | 0.169 /k$ | passed |
| exact_modmul | 8192×8192×8192 | int8 | 0.02 G_modmul/s | 0.018 /k$ | passed | — | — | — | 0.33 G_modmul/s | 0.163 /k$ | passed |

**Cross-device throughput ratios** (higher = first device wins):

| op_kind | shape | backend | Blackhole / CPU | price-adjusted (Blackhole /k$) / (CPU /k$) |
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
