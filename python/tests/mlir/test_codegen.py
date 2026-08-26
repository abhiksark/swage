# python/tests/mlir/test_codegen.py
"""Native tests for fixed-block NVPTX compilation."""

import pytest
import swage as sw
import swage.language as sl
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _emit():
    return add_kernel.emit_mlir(
        signature={
            "x_ptr": sl.pointer(sl.float32),
            "y_ptr": sl.pointer(sl.float32),
            "output_ptr": sl.pointer(sl.float32),
            "n": sl.int32,
        },
        constexprs={"BLOCK": 128},
    )


def test_compiles_fixed_vector_add_to_deterministic_ptx():
    """Lower the fixed vector add without mutating its semantic module."""
    module = _emit()
    original = module.operation.get_asm(enable_debug_info=False)

    first = native_swage._compile_ptx(
        module, kernel_name="add_kernel", block_size=128, target="sm_80"
    )
    second = native_swage._compile_ptx(
        module, kernel_name="add_kernel", block_size=128, target="sm_80"
    )

    assert first == second
    lowered, ptx = first
    assert "swage." not in lowered
    assert "vector." not in lowered
    assert "llvm.func @add_kernel" in lowered
    assert '#nvvm.target<chip = "sm_80">' in lowered
    assert ".target sm_80" in ptx
    assert ".entry add_kernel" in ptx
    assert "ld.global.b32" in ptx
    assert "st.global.b32" in ptx
    assert module.operation.get_asm(enable_debug_info=False) == original


@pytest.mark.parametrize("target", ["sm_8", "compute_80", "sm_79", "sm_999"])
def test_rejects_invalid_nvptx_targets(target):
    """Fail closed instead of silently choosing another architecture."""
    with pytest.raises(
        ValueError,
        match="target must match sm_<major><minor> and be sm_80 or newer",
    ):
        native_swage._compile_ptx(
            _emit(),
            kernel_name="add_kernel",
            block_size=128,
            target=target,
        )


@pytest.mark.parametrize("target", ["sm_85", "sm_99", "sm_119"])
def test_rejects_sm_values_the_pinned_llvm_does_not_support(target):
    """Fail at admission instead of emitting PTX no driver can load."""
    with pytest.raises(
        ValueError, match="not a processor supported by the pinned LLVM"
    ):
        native_swage._compile_ptx(
            _emit(),
            kernel_name="add_kernel",
            block_size=128,
            target=target,
        )


def test_fixed_kernels_pin_their_launch_width_with_reqntid():
    """Make a mismatched blockDim a launch error, not a wrong result."""
    _, ptx = native_swage._compile_ptx(
        _emit(), kernel_name="add_kernel", block_size=128, target="sm_80"
    )
    assert ".reqntid 128, 1, 1" in ptx


def test_rejects_a_block_size_above_the_hardware_limit():
    """1024 is the CUDA block ceiling; anything larger can never launch."""
    with pytest.raises(ValueError, match="at most 1024"):
        native_swage._compile_ptx(
            _emit(),
            kernel_name="add_kernel",
            block_size=2048,
            target="sm_80",
        )


def test_compiles_for_the_newest_supported_sm():
    """Keep the admission edge chips compiling, not just sm_80."""
    _, ptx = native_swage._compile_ptx(
        _emit(), kernel_name="add_kernel", block_size=128, target="sm_121"
    )
    assert ".target sm_121" in ptx


def test_rejects_a_block_size_that_differs_from_the_vector_shape():
    """Keep the launch block and semantic vector width identical."""
    with pytest.raises(ValueError, match="vector width 128 does not match"):
        native_swage._compile_ptx(
            _emit(),
            kernel_name="add_kernel",
            block_size=64,
            target="sm_80",
        )


def test_rejects_function_symbols_ptx_cannot_represent():
    """Fail with a diagnostic instead of aborting in the NVPTX printer."""
    from mlir_swage import ir
    from mlir_swage.dialects import swage as swage_dialect

    source = _emit().operation.get_asm(enable_debug_info=False)
    renamed = source.replace("@add_kernel", '@"añadir"')
    with ir.Context() as context:
        swage_dialect.register_dialects(context)
        module = ir.Module.parse(renamed)
        with pytest.raises(
            ValueError, match="not a valid PTX identifier"
        ):
            native_swage._compile_ptx(
                module,
                kernel_name="añadir",
                block_size=128,
                target="sm_80",
            )
