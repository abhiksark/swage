# python/tests/mlir/test_target_compile.py
"""Compile-only coverage of every admitted NVPTX processor.

No test here launches a kernel, so the file needs the native bindings and no
GPU. Compiling for a processor shows that the pinned backend emits PTX for
it. It does not show that the kernel runs correctly there: only `sm_86` has
execution evidence, and the other processors are admitted, not qualified.
"""

import pytest
import swage as sw
import swage.language as sl
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage
from reduction_programs import reduction_module

# The processors the code generation C API admits. The last test in this
# file fails when the native list and this one disagree in either direction,
# so a newly admitted processor cannot skip the compile tests.
_ADMITTED = (80, 86, 87, 88, 89, 90, 100, 101, 103, 110, 120, 121)

# The range of `sm_` values the target syntax accepts.
_SYNTACTIC = range(80, 130)

# Every private segmented entry point: the native function, its options, and
# the suffix it gives the kernel name in the PTX.
_SEGMENTED = {
    "direct": (
        "_compile_segmented_reduction_ptx",
        {"block_size": 128},
        "",
    ),
    "task-ids": (
        "_compile_segmented_reduction_ptx",
        {"block_size": 32, "use_task_ids": True},
        "",
    ),
    "fused": ("_compile_fused_segmented_reduction_ptx", {}, ""),
    "persistent": ("_compile_persistent_segmented_reduction_ptx", {}, ""),
    "split-partial": (
        "_compile_split_partial_reduction_ptx",
        {},
        "__partial",
    ),
    "split-merge": ("_compile_split_merge_reduction_ptx", {}, "__merge"),
}


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):  # noqa: D103
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


def _emit_public_kernel():
    return add_kernel.emit_mlir(
        signature={
            "x_ptr": sl.pointer(sl.float32),
            "y_ptr": sl.pointer(sl.float32),
            "output_ptr": sl.pointer(sl.float32),
            "n": sl.int32,
        },
        constexprs={"BLOCK": 128},
    )


@pytest.mark.parametrize("sm", _ADMITTED)
def test_public_kernel_compiles_for_every_admitted_processor(sm):
    """Emit PTX for the one launchable kernel on each admitted processor."""
    lowered, ptx, _ = native_swage._compile_ptx(
        _emit_public_kernel(),
        kernel_name="add_kernel",
        block_size=128,
        target=f"sm_{sm}",
    )

    assert f'#nvvm.target<chip = "sm_{sm}">' in lowered
    assert f".target sm_{sm}\n" in ptx
    assert ".entry add_kernel(" in ptx


@pytest.mark.parametrize("sm", _ADMITTED)
@pytest.mark.parametrize("schedule", _SEGMENTED)
def test_segmented_kernels_compile_for_every_admitted_processor(schedule, sm):
    """Emit PTX for each private segmented kernel on each processor."""
    compiler, options, suffix = _SEGMENTED[schedule]
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(reduction_module("sum", "identity"))

        lowered, ptx, _ = getattr(native_swage, compiler)(
            module,
            kernel_name="segmented_sum",
            target=f"sm_{sm}",
            **options,
        )

    assert f'#nvvm.target<chip = "sm_{sm}">' in lowered
    assert f".target sm_{sm}\n" in ptx
    assert f".entry segmented_sum{suffix}(" in ptx


def test_every_other_processor_is_refused():
    """Keep the admitted list and the compile tests above in step."""
    module = _emit_public_kernel()
    admitted = []
    for sm in _SYNTACTIC:
        try:
            native_swage._compile_ptx(
                module,
                kernel_name="add_kernel",
                block_size=128,
                target=f"sm_{sm}",
            )
        except ValueError as error:
            assert "not a processor supported by the pinned LLVM" in str(error)
        else:
            admitted.append(sm)

    assert tuple(admitted) == _ADMITTED
