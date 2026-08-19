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


@sw.jit
def rebound_block_kernel(BLOCK: sl.constexpr):  # noqa: D103
    BLOCK = 4
    pid = sl.program_id(0)
    _ = pid * BLOCK + sl.arange(0, BLOCK)


@sw.jit
def nested_address_kernel(x_ptr, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    _ = x_ptr + (x_ptr + offsets)


@sw.jit
def oversized_axis_kernel():  # noqa: D103
    _ = sl.program_id(2147483648)


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
        f"{__file__}:{function_line + 1}:9: bad_kernel: "
        "unsupported statement 'If'"
    )
    assert str(caught.value) == expected


def test_nested_kernel_mlir_location_uses_real_source_column():
    """Restore indentation removed before parsing to emitted locations."""
    @sw.jit
    def nested_kernel():
        pid = sl.program_id(0)  # noqa: F841

    source_line = inspect.getsourcelines(nested_kernel.python_function)[1]
    debug_asm = nested_kernel.emit_mlir(
        signature={}, constexprs={}
    ).operation.get_asm(enable_debug_info=True)

    assert f'{__file__}":{source_line + 2}:15' in debug_asm


def test_rejects_every_unsupported_parameter_kind():
    """Never silently omit Python parameter kinds from the MLIR ABI."""
    @sw.jit
    def positional_only(value, /):
        return

    @sw.jit
    def keyword_only(*, value):
        return

    @sw.jit
    def variadic(*values):
        return

    @sw.jit
    def keyword_variadic(**values):
        return

    cases = [
        (
            positional_only,
            "value",
            "positional-only parameters are unsupported",
        ),
        (
            keyword_only,
            "value",
            "keyword-only parameters are unsupported",
        ),
        (
            variadic,
            "values",
            "variadic positional parameters are unsupported",
        ),
        (
            keyword_variadic,
            "values",
            "variadic keyword parameters are unsupported",
        ),
    ]
    for kernel, parameter, reason in cases:
        source, source_line = inspect.getsourcelines(kernel.python_function)
        parameter_line = next(
            offset for offset, line in enumerate(source) if parameter in line
        )
        column = source[parameter_line].index(parameter) + 1
        with pytest.raises(sw.CompilationError) as caught:
            kernel.emit_mlir(signature={}, constexprs={})

        expected = (
            f"{__file__}:{source_line + parameter_line}:{column}: "
            f"{kernel.__name__}: {reason}"
        )
        assert str(caught.value) == expected


def test_store_is_rejected_on_an_assignment_rhs():
    """Permit the effectful store only in expression-statement position."""
    @sw.jit
    def bad_kernel(output_ptr, n, BLOCK: sl.constexpr):
        pid = sl.program_id(0)
        offsets = pid * BLOCK + sl.arange(0, BLOCK)
        mask = offsets < n
        value = sl.load(output_ptr + offsets, mask=mask, other=0.0)
        _ = sl.store(output_ptr + offsets, value, mask=mask)

    with pytest.raises(
        sw.CompilationError,
        match="sl.store is only supported as an expression statement",
    ):
        bad_kernel.emit_mlir(
            signature={
                "output_ptr": sl.pointer(sl.float32),
                "n": sl.int32,
            },
            constexprs={"BLOCK": 32},
        )


def test_empty_return_must_be_the_final_statement():
    """Reject statements after return before constructing an invalid block."""
    @sw.jit
    def bad_kernel():
        return
        sl.program_id(0)

    with pytest.raises(
        sw.CompilationError,
        match="empty return must be the final statement",
    ):
        bad_kernel.emit_mlir(signature={}, constexprs={})


def test_unsupported_types_are_reported_in_source_parameter_order():
    """Choose the first invalid parameter independently of set ordering."""
    @sw.jit
    def alpha_first(alpha, beta):
        return

    @sw.jit
    def beta_first(beta, alpha):
        return

    signature = {"alpha": sl.float32, "beta": sl.float32}
    for kernel, expected_name in (
        (alpha_first, "alpha"),
        (beta_first, "beta"),
    ):
        with pytest.raises(sw.CompilationError) as caught:
            kernel.emit_mlir(signature=signature, constexprs={})

        assert str(caught.value).endswith(
            f"unsupported type for parameter '{expected_name}'"
        )


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


def test_constexpr_parameter_cannot_be_rebound():
    """Keep constexpr arithmetic consistent with the emitted vector width."""
    with pytest.raises(sw.CompilationError) as caught:
        rebound_block_kernel.emit_mlir(
            signature={}, constexprs={"BLOCK": 8}
        )

    assert str(caught.value).endswith(
        "rebound_block_kernel: cannot assign to constexpr parameter 'BLOCK'"
    )


@pytest.mark.parametrize(
    ("kernel_name", "emit", "reason"),
    [
        (
            "add_kernel",
            lambda: _emit(signature={1: sl.int32}),
            "signature keys must be strings",
        ),
        (
            "nested_address_kernel",
            lambda: nested_address_kernel.emit_mlir(
                signature={"x_ptr": sl.pointer(sl.float32)},
                constexprs={"BLOCK": 8},
            ),
            "pointers support only addition with offsets",
        ),
        (
            "oversized_axis_kernel",
            lambda: oversized_axis_kernel.emit_mlir(
                signature={}, constexprs={}
            ),
            "program_id axis must fit signed i32",
        ),
        (
            "add_kernel",
            lambda: _emit(constexprs={"BLOCK": 1 << 63}),
            "constexpr 'BLOCK' must fit a signed 64-bit MLIR dimension",
        ),
    ],
    ids=["mapping-key", "nested-address", "axis-range", "block-range"],
)
def test_malformed_supported_inputs_have_stable_diagnostics(
    kernel_name, emit, reason
):
    """Translate malformed trust-boundary inputs to compilation errors."""
    with pytest.raises(sw.CompilationError) as caught:
        emit()

    message = str(caught.value)
    assert message.startswith(f"{__file__}:")
    assert message.endswith(f"{kernel_name}: {reason}")


def test_internal_mlir_verification_error_is_source_located(monkeypatch):
    """Verify emitted modules and translate only native verification errors."""
    native_module = ir.Module

    class FailingOperation:
        """Operation proxy that reports a native verifier failure."""

        @staticmethod
        def verify():
            raise ir.MLIRError("forced verification failure", [])

    class FailingModule:
        """Module proxy retaining the real body used by operation builders."""

        def __init__(self, module):
            self.body = module.body
            self.operation = FailingOperation()

        @classmethod
        def create(cls, location):
            return cls(native_module.create(location))

    monkeypatch.setattr(ir, "Module", FailingModule)

    with pytest.raises(sw.CompilationError) as caught:
        _emit()

    assert str(caught.value).endswith(
        "add_kernel: emitted MLIR failed verification"
    )
