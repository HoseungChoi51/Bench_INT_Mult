"""Hardware tests for the Tensor-engine side of the cost-model story (Tenstorrent Blackhole).

Notebook 4 makes a concrete claim about INT8 / Tensor-Core-style hardware:

* A 36-bit operand decomposes into 5 bytes, so an exact 36-bit operand-pair
  multiplication needs ``5 x 5 = 25`` byte-level cross-products.
* Recomposition by byte-shift index ``(i + j)`` then yields the exact
  72-bit product, which can finally be reduced modulo a 36-bit prime.

These tests exercise that claim on the *actual* Tenstorrent Blackhole card
in the host. They run when:

* the ``ttnn`` Python module is importable in the current interpreter
  (it ships with ``tt-metal``; install via the project's Tenstorrent venv,
  e.g. ``/home/chs/.tenstorrent-venv``), and
* a Blackhole device can be opened *and* a trivial elementwise op can be
  built and dispatched. If kernel build fails (e.g. SFPI toolchain
  problem) the whole module skips with a clear message; it does not fail
  the suite.

To run::

    /home/chs/.tenstorrent-venv/bin/pytest tests/test_blackhole_ttnn.py -v

Note: ``ttnn`` currently requires CPython 3.12, while this project pins
3.13. From the project venv these tests will skip via
``pytest.importorskip``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

ttnn = pytest.importorskip("ttnn", reason="Tenstorrent ttnn not installed in this interpreter")
torch = pytest.importorskip("torch", reason="ttnn requires PyTorch")

from rns_arithmetic.limb_decompose import chunk_product_count  # noqa: E402

# 36-bit prime used in the notebook narrative.
Q36: int = (1 << 36) - 5
SEED: int = 20260426

# Blackhole TILE_LAYOUT requires shapes that are multiples of the 32x32 tile.
TILE: int = 32


@pytest.fixture(scope="module")
def device() -> Iterator[object]:
    """Open the Blackhole and validate that kernels can build and run."""
    try:
        dev = ttnn.open_device(device_id=0)
    except Exception as exc:  # pragma: no cover - hardware/driver dependent
        pytest.skip(f"could not open Tenstorrent device: {exc}")

    # Smoke check: an elementwise multiply forces kernel compilation. If the
    # SFPI toolchain (or any other piece of the dispatch path) is broken we
    # would rather skip the suite than report misleading test failures.
    try:
        sample = torch.ones(TILE, TILE, dtype=torch.float32)
        ts = ttnn.from_torch(sample, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        _ = ttnn.to_torch(ttnn.multiply(ts, ts))
    except Exception as exc:  # pragma: no cover - hardware/driver dependent
        ttnn.close_device(dev)
        pytest.skip(f"Blackhole kernel build/dispatch smoke test failed: {exc}")

    try:
        yield dev
    finally:
        ttnn.close_device(dev)


def _to_device(t: torch.Tensor, device: object, dtype=ttnn.float32) -> object:
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def test_blackhole_arch_is_blackhole(device: object) -> None:
    """The opened device should self-report as Blackhole on this host."""
    arch = device.arch()
    # ttnn.device.Arch is an enum; its repr/name contains "BLACKHOLE" on a p150a.
    assert "BLACKHOLE" in str(arch).upper(), f"unexpected device arch: {arch}"


def test_bf16_matmul_identity(device: object) -> None:
    """A bfloat16 matmul against the identity reproduces the input on Blackhole.

    Sanity check that the matrix engine is functional and that the Python
    binding round-trips tensors correctly. bfloat16 is exact for the small
    integer values used here.
    """
    a = torch.arange(TILE * TILE, dtype=torch.float32).reshape(TILE, TILE) % 64.0
    eye = torch.eye(TILE, dtype=torch.float32)

    ta = _to_device(a, device, dtype=ttnn.bfloat16)
    te = _to_device(eye, device, dtype=ttnn.bfloat16)
    tc = ttnn.matmul(ta, te)
    c = ttnn.to_torch(tc).to(torch.float32)

    # bfloat16 holds integers up to 2**8 exactly; values are < 64 so this is exact.
    assert torch.equal(c, a), f"max diff = {(c - a).abs().max().item()}"


def test_int8_chunk_decomposition_reconstructs_36_bit_products(device: object) -> None:
    """The 25-INT8-chunk decomposition reconstructs exact 36-bit modular products.

    For a batch of ``TILE x TILE`` independent 36-bit residue pairs, decompose
    each operand into its 5 bytes, run all ``5 x 5 = 25`` byte cross-products
    elementwise on the Blackhole, then accumulate the byte-shifted partials.
    The final ``(a * b) % Q36`` must equal the bigint reference.

    This is the empirical version of the headline number in PLAN.md §4.5:
    ``chunk_product_count(36, 8) == 25``.
    """
    assert chunk_product_count(36, 8) == 25  # sanity link to the cost model

    rng = np.random.default_rng(SEED)
    a_np = rng.integers(0, Q36, size=(TILE, TILE), dtype=np.uint64)
    b_np = rng.integers(0, Q36, size=(TILE, TILE), dtype=np.uint64)

    # Byte decomposition (LSB-first). Each chunk is a uint16 in [0, 256).
    n_chunks = math.ceil(36 / 8)
    a_bytes = [((a_np >> (8 * i)) & 0xFF).astype(np.float32) for i in range(n_chunks)]
    b_bytes = [((b_np >> (8 * i)) & 0xFF).astype(np.float32) for i in range(n_chunks)]

    # Push each byte-plane to the Blackhole as float32 (exact for [0, 256)).
    a_dev = [_to_device(torch.from_numpy(x), device) for x in a_bytes]
    b_dev = [_to_device(torch.from_numpy(x), device) for x in b_bytes]

    # 25 elementwise byte cross-products on the device.
    cross_products: dict[tuple[int, int], np.ndarray] = {}
    n_ops = 0
    for i in range(n_chunks):
        for j in range(n_chunks):
            prod_dev = ttnn.multiply(a_dev[i], b_dev[j])
            prod = ttnn.to_torch(prod_dev).to(torch.int64).numpy()
            # Each product <= 255*255 = 65025, fits losslessly in float32 -> int64.
            assert prod.max() <= 255 * 255, "byte product exceeded INT8 envelope"
            cross_products[(i, j)] = prod.astype(object)  # bigint accumulation
            n_ops += 1

    assert n_ops == 25, "expected exactly 25 byte cross-products for 36-bit / INT8"

    # Bigint accumulation of weighted partials. The accumulator must be Python
    # int (object dtype) because the full product is up to 72 bits.
    full_product = np.zeros((TILE, TILE), dtype=object)
    for (i, j), c_ij in cross_products.items():
        full_product = full_product + (c_ij << (8 * (i + j)))

    actual = np.array(
        [[int(full_product[r, c]) % Q36 for c in range(TILE)] for r in range(TILE)],
        dtype=np.uint64,
    )
    expected = np.array(
        [[(int(a_np[r, c]) * int(b_np[r, c])) % Q36 for c in range(TILE)] for r in range(TILE)],
        dtype=np.uint64,
    )
    assert np.array_equal(actual, expected), (
        f"byte-chunk reconstruction disagrees with bigint reference at "
        f"{int(np.sum(actual != expected))} of {TILE * TILE} positions"
    )


def test_byte_chunk_gemm_reconstructs_36_bit_matmul(device: object) -> None:
    """The 25-byte-GEMM decomposition reconstructs an exact 36-bit-residue matmul.

    This is the matrix-shaped variant of the previous test, and the form
    that maps directly onto Tensor-Core-style hardware: each of the 25
    byte products becomes a small INT8 (here float32 with byte values)
    GEMM. We use ``K = TILE = 32``; the per-element matmul accumulator is
    bounded by ``K * 255**2 = 2_080_800``, well within float32's exact
    integer range (``2**24 = 16_777_216``).
    """
    M = K = N = TILE  # one tile each
    rng = np.random.default_rng(SEED + 1)
    a_np = rng.integers(0, Q36, size=(M, K), dtype=np.uint64)
    b_np = rng.integers(0, Q36, size=(K, N), dtype=np.uint64)

    n_chunks = 5
    assert K * 255 * 255 < (1 << 24), "K too large for exact float32 integer accumulation"

    a_planes = [((a_np >> (8 * i)) & 0xFF).astype(np.float32) for i in range(n_chunks)]
    b_planes = [((b_np >> (8 * j)) & 0xFF).astype(np.float32) for j in range(n_chunks)]

    a_dev = [_to_device(torch.from_numpy(x), device) for x in a_planes]
    b_dev = [_to_device(torch.from_numpy(x), device) for x in b_planes]

    # 25 INT8-shaped GEMMs on the Blackhole.
    full = np.zeros((M, N), dtype=object)
    for i in range(n_chunks):
        for j in range(n_chunks):
            c_ij = ttnn.to_torch(ttnn.matmul(a_dev[i], b_dev[j])).to(torch.int64).numpy()
            full = full + (c_ij.astype(object) << (8 * (i + j)))

    actual = np.array(
        [[int(full[r, c]) % Q36 for c in range(N)] for r in range(M)],
        dtype=np.uint64,
    )
    # Reference: arbitrary-precision Python matmul mod Q36.
    expected = np.zeros((M, N), dtype=np.uint64)
    for r in range(M):
        for c in range(N):
            acc = 0
            for k in range(K):
                acc += int(a_np[r, k]) * int(b_np[k, c])
            expected[r, c] = acc % Q36

    assert np.array_equal(actual, expected), (
        f"25-GEMM reconstruction disagrees with bigint matmul reference at "
        f"{int(np.sum(actual != expected))} of {M * N} positions"
    )
