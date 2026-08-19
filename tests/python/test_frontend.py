# tests/python/test_frontend.py
"""LLVM-free tests for the public compile-only frontend API."""

import subprocess
import sys

import pytest
import swage as sw
import swage.language as sl


def test_importing_swage_does_not_import_native_bindings():
    """Keep the public package usable without the native build tree."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import swage\n"
            "assert 'mlir_swage' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


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


def test_stacked_decorator_is_rejected():
    """Reject decorator semantics that the frontend would otherwise ignore."""
    def passthrough(function):
        return function

    with pytest.raises(
        sw.CompilationError,
        match="only @swage.jit may decorate a kernel",
    ):
        @sw.jit
        @passthrough
        def kernel():
            return


@pytest.mark.parametrize(
    ("value", "reason"),
    [
        (True, "constexpr 'VALUE' must be an integer"),
        (1.5, "constexpr 'VALUE' must be an integer"),
        (1 << 63, "constexpr 'VALUE' must fit signed 64-bit"),
        (-(1 << 63) - 1, "constexpr 'VALUE' must fit signed 64-bit"),
    ],
)
def test_constexpr_values_are_validated_before_native_import(value, reason):
    """Keep unsupported constexpr values inside the diagnostic boundary."""
    @sw.jit
    def kernel(VALUE: sl.constexpr):
        return

    with pytest.raises(sw.CompilationError, match=reason):
        kernel.emit_mlir(signature={}, constexprs={"VALUE": value})


def test_calling_a_decorated_kernel_reports_execution_unavailable():
    """Keep compile-only kernels non-executing until M3."""
    @sw.jit
    def kernel():
        return

    with pytest.raises(RuntimeError, match="execution is unavailable until M3"):
        kernel()
