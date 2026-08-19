# tests/python/test_frontend.py
"""LLVM-free tests for the public compile-only frontend API."""

import subprocess
import sys
import types
from unittest import mock

import pytest
import swage as sw
import swage.language as sl


def test_importing_swage_does_not_import_optional_dependencies():
    """Keep the public package usable without native or PyTorch packages."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import swage\n"
            "assert 'mlir_swage' not in sys.modules\n"
            "assert 'torch' not in sys.modules",
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


@pytest.mark.parametrize(
    ("keywords", "reason"),
    [
        ({}, "exactly one of signature or arguments is required"),
        (
            {"signature": {}, "arguments": {}},
            "exactly one of signature or arguments is required",
        ),
    ],
)
def test_emit_requires_one_runtime_input_mode(keywords, reason):
    """Reject ambiguous runtime type inputs before native imports."""
    @sw.jit
    def kernel():
        return

    with pytest.raises(sw.CompilationError, match=reason):
        kernel.emit_mlir(constexprs={}, **keywords)


def test_argument_keys_are_validated_before_importing_pytorch():
    """Report mapping mistakes without requiring the optional dependency."""
    @sw.jit
    def kernel(value):
        return

    with mock.patch("builtins.__import__", side_effect=ImportError("missing")):
        with pytest.raises(
            sw.CompilationError,
            match=(
                "arguments keys must match runtime parameters; "
                "missing: value"
            ),
        ):
            kernel.emit_mlir(arguments={}, constexprs={})


def test_missing_pytorch_has_an_installation_hint():
    """Keep optional-dependency failures inside the diagnostic boundary."""
    @sw.jit
    def kernel(value):
        return

    real_import = __import__

    def import_without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=import_without_torch):
        with pytest.raises(sw.CompilationError) as caught:
            kernel.emit_mlir(arguments={"value": 1}, constexprs={})

    message = str(caught.value)
    assert message.startswith(f"{__file__}:")
    assert message.endswith(
        "kernel: PyTorch metadata inference requires "
        "'swage-compiler[pytorch]'"
    )


def test_pytorch_metadata_failures_have_an_installation_hint(monkeypatch):
    """Translate dependency metadata errors into stable diagnostics."""
    class Tensor:
        @property
        def layout(self):
            raise RuntimeError("broken metadata")

    fake_torch = types.SimpleNamespace(
        Tensor=Tensor,
        float32=object(),
        strided=object(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    @sw.jit
    def kernel(value):
        return

    with pytest.raises(sw.CompilationError) as caught:
        kernel.emit_mlir(arguments={"value": Tensor()}, constexprs={})

    assert str(caught.value).endswith(
        "kernel: could not read PyTorch metadata for parameter 'value'; "
        "install 'swage-compiler[pytorch]'"
    )


@pytest.mark.parametrize(
    "value",
    [True, -(1 << 31) - 1, 1 << 31, 1.5, object()],
)
def test_inferred_scalars_reject_unsupported_values(value):
    """Accept only non-boolean Python integers in the signed i32 range."""
    @sw.jit
    def kernel(runtime_value):
        return

    with pytest.raises(
        sw.CompilationError,
        match="unsupported argument for parameter 'runtime_value'",
    ):
        kernel.emit_mlir(
            arguments={"runtime_value": value}, constexprs={}
        )


def test_calling_a_decorated_kernel_reports_execution_unavailable():
    """Keep compile-only kernels non-executing until M3."""
    @sw.jit
    def kernel():
        return

    with pytest.raises(RuntimeError, match="execution is unavailable until M3"):
        kernel()
