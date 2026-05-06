"""Pytest plugin: strip the upstream `@pytest.mark.skip` from the GEMM_FLOPS
benchmarks so they run in our reproduction.

The upstream tests at `tt-metal/tests/ttnn/unit_tests/benchmarks/test_benchmark.py`
are marked with `@pytest.mark.skip(reason="...not intended to be run as part
of CI...")`. We want to invoke them unmodified — same code, same
configuration — but override that one CI-safety gate. This plugin removes
the skip marker for exactly the two GEMM_FLOPS benchmark entries and
nothing else.

Usage::

    pytest tests/ttnn/unit_tests/benchmarks/test_benchmark.py::test_matmul_2d_host_perf \\
        -p no:cacheprovider -s \\
        --rootdir=/home/chs/TT/tt-metal \\
        -p tt_metal_extras.upstream_runner_conftest

(or set PYTHONPATH so this module is importable, then `-p
tt_metal_extras.upstream_runner_conftest`).
"""

from __future__ import annotations

UNGATE_TEST_NAMES = {
    "test_matmul_2d_host_perf",
    "test_matmul_2d_host_perf_out_of_box",
}


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    for item in items:
        if item.originalname not in UNGATE_TEST_NAMES:
            continue
        # Drop the `@pytest.mark.skip(...)` marker so the test actually runs.
        item.own_markers = [m for m in item.own_markers if m.name != "skip"]
