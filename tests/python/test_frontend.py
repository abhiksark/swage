# tests/python/test_frontend.py
"""LLVM-free tests for the public compile-only frontend API."""

import sys

import pytest
import swage as sw
import swage.language as sl


def test_importing_swage_does_not_import_native_bindings():
    """Keep the public package usable without the native build tree."""
    assert "mlir_swage" not in sys.modules


@pytest.mark.parametrize(
    "symbolic_call",
    [
        lambda: sl.program_id(0),
        lambda: sl.arange(0, 1),
        lambda: sl.load(None),
        lambda: sl.store(None, None),
    ],
)
def test_symbolic_language_calls_fail_outside_jit(symbolic_call):
    """Reject symbolic operations instead of pretending to execute them."""
    with pytest.raises(RuntimeError, match="only available inside @swage.jit"):
        symbolic_call()


def test_decorating_a_kernel_does_not_execute_its_body():
    """Capture source without running arbitrary user code."""
    calls = []

    @sw.jit
    def kernel():
        calls.append("executed")

    assert calls == []
    assert kernel.__name__ == "kernel"


def test_calling_a_decorated_kernel_reports_execution_unavailable():
    """Keep compile-only kernels non-executing until M3."""
    @sw.jit
    def kernel():
        return

    with pytest.raises(RuntimeError, match="execution is unavailable until M3"):
        kernel()
