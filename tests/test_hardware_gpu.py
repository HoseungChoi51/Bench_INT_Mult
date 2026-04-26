"""Hardware-specific tests for NVIDIA CUDA GPU (target: RTX 5090, sm_120).

These tests pin Notebook 4's core claim onto an actual Tensor-Core pipeline:

- INT8 → INT32 GEMM via ``torch._int_mm`` is exact for small inputs. This
  is the path through which PyTorch dispatches to the integer Tensor-Core
  MMA instructions on Ampere (sm_80) and later devices, including
  Blackwell sm_120.
- A 36-bit-residue matrix product ``A @ B`` can be reconstructed exactly
  from the :math:`5 \\times 5 = 25` byte-chunk GEMMs predicted by
  :func:`rns_arithmetic.tensor_int8_model.byte_grid` (Notebook 4 §4.3).
- 31-bit RNS modular multiplication on the GPU using ``int64`` is exact —
  the GPU analogue of the CPU uint64 path, executed on CUDA cores rather
  than Tensor Cores.

The tests skip cleanly when torch is missing (the ``bench`` extra is
optional) or when no CUDA-capable GPU is present.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE,
    reason="CUDA-capable GPU required",
)

from rns_arithmetic.tensor_int8_model import byte_grid  # noqa: E402

DEVICE = "cuda" if CUDA_AVAILABLE else "cpu"


def test_gpu_supports_int8_tensor_cores() -> None:
    """Compute capability gate: INT8 tensor cores require sm_75+ (Turing).

    Also surfaces the device name so the suite records which GPU it ran on.
    Blackwell (sm_120, e.g. RTX 5090) and Ada/Ampere/Turing all pass.
    """
    cap_major, cap_minor = torch.cuda.get_device_capability(0)
    sm = 10 * cap_major + cap_minor
    name = torch.cuda.get_device_name(0)
    assert sm >= 75, f"INT8 tensor cores require sm_75+; got {name} (sm_{sm})"


def test_int8_gemm_matches_int32_reference() -> None:
    """``torch._int_mm`` produces an exact INT8 → INT32 matmul on this device.

    Constraints (PyTorch 2.x): both operands must be 2-D ``int8`` tensors
    on CUDA, and ``M`` (the first dimension of ``a``) must be > 16. Output
    is ``int32``. The reference is the same matmul promoted to ``int32``
    on the host.
    """
    rng = np.random.default_rng(20260426)
    M, K, N = 64, 128, 64
    a_np = rng.integers(-128, 128, size=(M, K), dtype=np.int8)
    b_np = rng.integers(-128, 128, size=(K, N), dtype=np.int8)
    a = torch.from_numpy(a_np)
    b = torch.from_numpy(b_np)

    c_gpu = torch._int_mm(a.to(DEVICE), b.to(DEVICE))
    c_ref = a.to(torch.int32) @ b.to(torch.int32)

    assert c_gpu.shape == (M, N)
    assert c_gpu.dtype == torch.int32
    assert torch.equal(c_gpu.cpu(), c_ref)


def test_36bit_matmul_via_25_int8_gemms() -> None:
    """5×5 = 25 INT8 GEMMs reconstruct an exact 36-bit-residue matrix product.

    For matrices of 36-bit-shaped residues, decompose each into 5 byte
    chunks :math:`A = \\sum_i A_i 2^{8i}`, :math:`B = \\sum_j B_j 2^{8j}`.
    Then

    .. math::
        A B = \\sum_{i, j} (A_i B_j) \\cdot 2^{8(i + j)},

    a 5×5 grid of byte-chunk GEMMs whose byte-shift indices are exactly
    :func:`byte_grid` ``(40, 8)``. The reference is the same product
    computed in Python big ints.

    To keep every byte chunk inside the signed-int8 range without sign
    tricks, each chunk is drawn from ``[0, 127]``. Operands are then
    bounded by :math:`127 \\cdot (1 + 2^8 + 2^{16} + 2^{24} + 2^{32})
    < 2^{39}`, which exercises the same 5-chunk decomposition shape as
    36-bit operands.
    """
    rng = np.random.default_rng(0xCAFE)
    M, K, N = 32, 32, 32  # all > 16 to satisfy torch._int_mm

    grid = byte_grid(40, 8)  # 5×5; G[i, j] = i + j is the byte-shift index
    n_chunks = grid.shape[0]
    assert n_chunks == 5
    assert grid.size == 25  # the headline number from Notebook 4

    a_chunks = rng.integers(0, 128, size=(n_chunks, M, K), dtype=np.int8)
    b_chunks = rng.integers(0, 128, size=(n_chunks, K, N), dtype=np.int8)

    # Build the operand and reference product in Python big ints (object dtype).
    A_obj = np.zeros((M, K), dtype=object)
    B_obj = np.zeros((K, N), dtype=object)
    for i in range(n_chunks):
        A_obj += a_chunks[i].astype(object) * (1 << (8 * i))
        B_obj += b_chunks[i].astype(object) * (1 << (8 * i))
    C_ref = A_obj @ B_obj

    # Compute the same product on the GPU via 25 INT8 GEMMs.
    a_dev = [torch.from_numpy(c).to(DEVICE) for c in a_chunks]
    b_dev = [torch.from_numpy(c).to(DEVICE) for c in b_chunks]

    C_total = np.zeros((M, N), dtype=object)
    n_gemms = 0
    for i in range(n_chunks):
        for j in range(n_chunks):
            partial_int32 = torch._int_mm(a_dev[i], b_dev[j])
            partial_host = partial_int32.cpu().numpy().astype(np.int64).astype(object)
            C_total += partial_host * (1 << (8 * int(grid[i, j])))
            n_gemms += 1
    assert n_gemms == 25

    assert (C_total == C_ref).all()


def test_31bit_rns_mulmod_on_gpu() -> None:
    """Per-element 31-bit RNS multiply on the GPU using ``torch.int64``.

    With ``q ~ 2**31 - 1`` the raw product fits in int64 with ample margin.
    This is the GPU analogue of the CPU uint64 path; no Tensor-Core
    involvement, plain elementwise multiply through CUDA cores.
    """
    rng = np.random.default_rng(42)
    q = (1 << 31) - 1
    n = 1_000_000
    a_np = rng.integers(0, q, size=n, dtype=np.int64)
    b_np = rng.integers(0, q, size=n, dtype=np.int64)
    a = torch.from_numpy(a_np).to(DEVICE)
    b = torch.from_numpy(b_np).to(DEVICE)
    result = (a * b) % q

    idx = rng.choice(n, size=1000, replace=False)
    a_h = a_np[idx]
    b_h = b_np[idx]
    r_h = result.cpu().numpy()[idx]
    for ai, bi, ri in zip(a_h.tolist(), b_h.tolist(), r_h.tolist(), strict=True):
        assert int(ri) == (int(ai) * int(bi)) % q
