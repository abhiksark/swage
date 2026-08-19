# python/tests/mlir/test_frontend.py
"""Native binding tests for the compile-only Python AST frontend."""

import inspect

import pytest
import swage as sw
import swage.language as sl
from mlir_swage import ir


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


SIGNATURE = {
    "x_ptr": sl.pointer(sl.float32),
    "y_ptr": sl.pointer(sl.float32),
    "output_ptr": sl.pointer(sl.float32),
    "n": sl.int32,
}


def _emit(kernel=add_kernel, signature=SIGNATURE, constexprs=None):
    """Emit one native module with the standard test signature."""
    if constexprs is None:
        constexprs = {"BLOCK": 128}
    return kernel.emit_mlir(signature=signature, constexprs=constexprs)


def test_vector_add_emits_a_deterministic_live_module():
    """Build the required vector-add structure without textual round trips."""
    first = _emit()
    second = _emit()

    assert isinstance(first, ir.Module)
    assert first.operation.verify()
    first_asm = first.operation.get_asm(enable_debug_info=False)
    second_asm = second.operation.get_asm(enable_debug_info=False)
    assert first_asm == second_asm
    assert "swage.program_id" in first_asm
    assert "vector.step" in first_asm
    assert first_asm.count("vector.gather") == 2
    assert "arith.addf" in first_asm
    assert "vector.scatter" in first_asm


def test_vector_add_preserves_kernel_and_python_source_locations():
    """Expose kernel-name and file-line locations in debug assembly."""
    debug_asm = _emit().operation.get_asm(enable_debug_info=True)
    source_line = inspect.getsourcelines(add_kernel.python_function)[1]

    assert 'loc("add_kernel"' in debug_asm
    assert __file__ in debug_asm
    assert f":{source_line + 1}:" in debug_asm


def test_arbitrary_python_call_is_rejected_without_invoking_it():
    """Reject calls outside the symbolic subset without executing Python."""
    calls = []

    def arbitrary_call():
        calls.append("executed")

    @sw.jit
    def bad_kernel():
        arbitrary_call()

    with pytest.raises(sw.CompilationError, match="unsupported call"):
        bad_kernel.emit_mlir(signature={}, constexprs={})
    assert calls == []


def test_control_flow_has_a_stable_source_diagnostic():
    """Reject unsupported control flow at its Python source location."""
    @sw.jit
    def bad_kernel():
        if True:
            return

    function_line = inspect.getsourcelines(bad_kernel.python_function)[1] + 1
    with pytest.raises(sw.CompilationError) as caught:
        bad_kernel.emit_mlir(signature={}, constexprs={})

    expected = (
        f"{__file__}:{function_line + 1}:5: bad_kernel: "
        "unsupported statement 'If'"
    )
    assert str(caught.value) == expected


@pytest.mark.parametrize(
    ("signature", "constexprs", "reason"),
    [
        (
            {key: value for key, value in SIGNATURE.items() if key != "n"},
            {"BLOCK": 128},
            "signature keys must match runtime parameters; missing: n",
        ),
        (
            {**SIGNATURE, "extra": sl.int32},
            {"BLOCK": 128},
            "signature keys must match runtime parameters; extra: extra",
        ),
        (
            {**SIGNATURE, "BLOCK": sl.int32},
            {},
            "constexpr parameter 'BLOCK' must be passed in constexprs",
        ),
        (
            {key: value for key, value in SIGNATURE.items() if key != "n"},
            {"BLOCK": 128, "n": 32},
            "runtime parameter 'n' must be passed in signature",
        ),
    ],
)
def test_signature_partition_errors_are_stable(signature, constexprs, reason):
    """Require runtime and constexpr arguments in their declared mappings."""
    with pytest.raises(sw.CompilationError, match=reason):
        _emit(signature=signature, constexprs=constexprs)


@pytest.mark.parametrize("block", [0, -1, True, 1.5])
def test_block_must_be_a_positive_integer(block):
    """Reject invalid static vector widths before building MLIR types."""
    with pytest.raises(
        sw.CompilationError,
        match="constexpr 'BLOCK' must be a positive integer",
    ):
        _emit(constexprs={"BLOCK": block})


@pytest.mark.parametrize(
    "bad_type",
    [sl.float32, sl.pointer(sl.int32), object()],
)
def test_unsupported_runtime_types_have_stable_diagnostics(bad_type):
    """Accept only i32 scalars and dynamic rank-one f32 pointers."""
    with pytest.raises(
        sw.CompilationError,
        match="unsupported type for parameter 'n'",
    ):
        _emit(signature={**SIGNATURE, "n": bad_type})
