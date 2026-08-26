# python/tests/mlir/test_frontend.py
"""Native binding tests for the compile-only Python AST frontend."""

import gc
import inspect
import weakref

import pytest
import swage as sw
import swage.language as sl
import torch
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


@sw.jit
def address_compare_left_kernel(x_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    _ = (x_ptr + offsets) < n


@sw.jit
def address_compare_right_kernel(x_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    _ = offsets < (x_ptr + offsets)


@sw.jit
def address_load_mask_kernel(x_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    _ = sl.load(
        x_ptr + offsets,
        mask=x_ptr + offsets,
        other=0.0,
    )


@sw.jit
def address_store_value_kernel(x_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    sl.store(x_ptr + offsets, x_ptr + offsets, mask=mask)


@sw.jit
def address_store_mask_kernel(x_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    offsets = sl.program_id(0) * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    value = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    sl.store(x_ptr + offsets, value, mask=x_ptr + offsets)


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


def _arguments(device="cpu"):
    """Create metadata inputs for the standard vector-add kernel."""
    return {
        "x_ptr": torch.empty(8, device=device),
        "y_ptr": torch.empty(8, device=device),
        "output_ptr": torch.empty(8, device=device),
        "n": 8,
    }


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


def test_inferred_and_explicit_signatures_emit_identical_mlir():
    """Route inferred descriptors through the existing emitter unchanged."""
    explicit = _emit().operation.get_asm(enable_debug_info=False)
    first = add_kernel.emit_mlir(
        arguments=_arguments(), constexprs={"BLOCK": 128}
    )
    second = add_kernel.emit_mlir(
        arguments=_arguments(), constexprs={"BLOCK": 128}
    )

    assert first.operation.verify()
    assert first.operation.get_asm(enable_debug_info=False) == explicit
    assert second.operation.get_asm(enable_debug_info=False) == explicit


def test_tensor_inference_never_reads_data_pointers(monkeypatch):
    """Infer from metadata without entering the future runtime boundary."""
    def fail_data_ptr(self):
        raise AssertionError("data_ptr must not be called")

    monkeypatch.setattr(torch.Tensor, "data_ptr", fail_data_ptr)

    module = add_kernel.emit_mlir(
        arguments=_arguments(), constexprs={"BLOCK": 128}
    )
    assert module.operation.verify()


@pytest.mark.parametrize(
    "value",
    [-(1 << 31), (1 << 31) - 1],
)
def test_inferred_i32_boundaries_are_accepted(value):
    """Accept both signed i32 scalar endpoints."""
    arguments = _arguments()
    arguments["n"] = value
    module = add_kernel.emit_mlir(
        arguments=arguments, constexprs={"BLOCK": 128}
    )
    assert module.operation.verify()


@pytest.mark.parametrize(
    ("tensor", "reason"),
    [
        (torch.empty(8, dtype=torch.float64), "dtype torch.float64"),
        (torch.empty(2, 4), "rank 2"),
        (torch.empty(16)[::2], "non-contiguous"),
        (
            torch.empty(8).to_sparse(),
            "layout torch.sparse_coo",
        ),
        (torch.empty(8, device="meta"), "device type 'meta'"),
    ],
)
def test_unsupported_tensor_metadata_has_stable_diagnostics(tensor, reason):
    """Reject tensor metadata outside the compile-only pointer contract."""
    with pytest.raises(sw.CompilationError) as caught:
        add_kernel.emit_mlir(
            arguments={**_arguments(), "x_ptr": tensor},
            constexprs={"BLOCK": 128},
        )

    assert str(caught.value).endswith(
        f"add_kernel: unsupported argument for parameter 'x_ptr': {reason}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_tensor_metadata_is_accepted():
    """Accept CUDA tensors as metadata providers without launching work."""
    module = add_kernel.emit_mlir(
        arguments=_arguments("cuda"), constexprs={"BLOCK": 128}
    )
    assert module.operation.verify()


def test_inferred_emission_does_not_retain_arguments():
    """Discard metadata providers before returning the live module."""
    def emit_with_local_tensor():
        tensor = torch.empty(8)
        reference = weakref.ref(tensor)
        module = add_kernel.emit_mlir(
            arguments={
                "x_ptr": tensor,
                "y_ptr": tensor,
                "output_ptr": tensor,
                "n": 8,
            },
            constexprs={"BLOCK": 128},
        )
        return module, reference

    module, reference = emit_with_local_tensor()
    gc.collect()

    assert module.operation.verify()
    assert reference() is None


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


@pytest.mark.parametrize(
    ("kernel", "reason"),
    [
        (
            address_compare_left_kernel,
            "comparison requires index offsets and i32",
        ),
        (
            address_compare_right_kernel,
            "comparison requires index offsets and i32",
        ),
        (address_load_mask_kernel, "sl.load mask must be a vector"),
        (
            address_store_value_kernel,
            "sl.store requires float values and a mask",
        ),
        (
            address_store_mask_kernel,
            "sl.store requires float values and a mask",
        ),
    ],
)
def test_addresses_in_value_positions_have_stable_diagnostics(kernel, reason):
    """Reject transient addresses before reading value-only type facts."""
    with pytest.raises(sw.CompilationError, match=reason):
        kernel.emit_mlir(
            signature={
                "x_ptr": sl.pointer(sl.float32),
                "n": sl.int32,
            },
            constexprs={"BLOCK": 8},
        )


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


@sw.jit
def oversized_literal_kernel(x_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    offsets = offsets + 18446744073709551617
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    sl.store(x_ptr + offsets, x, mask=mask)


def test_oversized_integer_literals_have_source_located_diagnostics():
    """Bound body literals like constexprs instead of leaking a TypeError."""
    with pytest.raises(
        sw.CompilationError, match="integer literal must fit signed 64-bit"
    ) as caught:
        oversized_literal_kernel.emit_mlir(
            signature={"x_ptr": sl.pointer(sl.float32), "n": sl.int32},
            constexprs={"BLOCK": 8},
        )

    line = inspect.getsourcelines(
        oversized_literal_kernel.python_function
    )[1] + 4
    assert f"{__file__}:{line}:" in str(caught.value)
