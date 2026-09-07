# python/tests/mlir/test_codegen.py
"""Native tests for fixed-block NVPTX compilation."""

from pathlib import Path

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
    lowered, ptx, contract = first
    assert "swage." not in lowered
    assert "vector." not in lowered
    assert "llvm.func @add_kernel" in lowered
    assert '#nvvm.target<chip = "sm_80">' in lowered
    assert ".target sm_80" in ptx
    assert ".entry add_kernel" in ptx
    assert "ld.global.b32" in ptx
    assert "st.global.b32" in ptx
    assert contract == (
        '{"version":2,"backend":"cuda","entry":"add_kernel",'
        '"launch":{"model":"spmd-grid","block":[128,1,1]},'
        '"arguments":[{"kind":"ptr","origin":"user","source_index":0,'
        '"access":"read"},{"kind":"ptr","origin":"user","source_index":1,'
        '"access":"read"},{"kind":"ptr","origin":"user","source_index":2,'
        '"access":"write"},{"kind":"i32","origin":"user","source_index":3}]}'
    )
    assert ptx.count(".param .u64") == 3
    assert ptx.count(".param .u32") == 1
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
    _, ptx, _ = native_swage._compile_ptx(
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
    _, ptx, _ = native_swage._compile_ptx(
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
        with pytest.raises(ValueError, match="not a valid PTX identifier"):
            native_swage._compile_ptx(
                module,
                kernel_name="añadir",
                block_size=128,
                target="sm_80",
            )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", ["f32", "f16", "f8E4M3FN", "f8E5M2"])
def test_multiply_codegen_preserves_source_and_launch_contract(backend, dtype):
    """Share add's ABI and rounding path without changing the source."""
    from mlir_swage import ir
    from mlir_swage.dialects import swage as swage_dialect

    source = _multiply_source().replace("f32", dtype)
    with ir.Context() as context:
        swage_dialect.register_dialects(context)
        module = ir.Module.parse(source)
        original = str(module)
        compiler = (
            native_swage._compile_fixed_host
            if backend == "cpu"
            else native_swage._compile_ptx
        )
        options = {"kernel_name": "multiply_kernel", "block_size": 128}
        if backend == "cuda":
            options["target"] = "sm_80"
        first = compiler(module, **options)
        second = compiler(module, **options)
        assert first[0] == second[0]
        assert first[2] == second[2]
        assert "llvm.fmul" in first[0]
        assert "vector." not in first[0]
        if backend == "cuda":
            assert first[1] == second[1]
            assert "mul.rn.f32" in first[1]
            assert ".reqntid 128, 1, 1" in first[1]
        else:
            assert first[1].entry == "multiply_kernel"
        add_module = ir.Module.parse(source.replace("arith.mulf", "arith.addf"))
        assert compiler(add_module, **options)[2] == first[2]
        assert str(module) == original


def _multiply_source():
    fixture = (
        Path(__file__).resolve().parents[3]
        / "test/Conversion/SwageToCPU/fixed-vector-multiply.mlir"
    )
    return fixture.read_text().split("// HOST-NOT:", 1)[0]


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "before,after,diagnostic",
    [
        ("%lhs, %rhs : vector", "%lhs, %lhs : vector", "do not form"),
        ("%mask, %product\n", "%mask, %lhs\n", "scatter value must"),
        ("%y[%c0]", "%x[%c0]", "do not form"),
        (
            "    %lhs = vector.gather %x[%c0] [%offsets], %mask, %passthrough",
            "    %other_mask = arith.cmpi ult, %offsets, %offsets "
            ": vector<128xindex>\n"
            "    %lhs = vector.gather %x[%c0] [%offsets], %other_mask, "
            "%passthrough",
            "do not form",
        ),
        (
            "    %rhs = vector.gather %y[%c0] [%offsets], %mask, %passthrough",
            "    %other_mask = arith.cmpi ult, %offsets, %offsets "
            ": vector<128xindex>\n"
            "    %rhs = vector.gather %y[%c0] [%offsets], %other_mask, "
            "%passthrough",
            "do not form",
        ),
        (
            "    vector.scatter %output[%c0] [%offsets], %mask, %product",
            "    %other_mask = arith.cmpi ult, %offsets, %offsets "
            ": vector<128xindex>\n"
            "    vector.scatter %output[%c0] [%offsets], %other_mask, "
            "%product",
            "do not form",
        ),
        ("%x[%c0] [%offsets]", "%x[%c0] [%lane]", "do not form"),
        ("%y[%c0] [%offsets]", "%y[%c0] [%lane]", "do not form"),
        ("%output[%c0] [%offsets]", "%output[%c0] [%lane]", "do not form"),
        ("arith.cmpi slt", "arith.cmpi sge", "canonical program offsets"),
        (
            "constant 0 : index",
            "constant 1 : index",
            "canonical program offsets",
        ),
        ("arith.mulf", "arith.divf", "unsupported"),
        (
            "    vector.scatter",
            "    %unused = arith.addf %lhs, %rhs : vector<128xf32>\n"
            "    vector.scatter",
            "one floating-point add or multiply",
        ),
        (
            "    vector.scatter",
            "    %unused = arith.mulf %zero, %zero : f32\n    vector.scatter",
            "one floating-point add or multiply",
        ),
        (
            "    vector.scatter",
            "    %chained = arith.addf %product, %rhs : vector<128xf32>\n"
            "    vector.scatter",
            "one floating-point add or multiply",
        ),
        (
            "    vector.scatter",
            "    %chained = arith.mulf %product, %rhs : vector<128xf32>\n"
            "    vector.scatter",
            "one floating-point add or multiply",
        ),
    ],
)
def test_malformed_multiply_fails_before_source_mutation(
    backend, before, after, diagnostic
):
    """Reject sibling invalid paths through the shared CPU/CUDA admission."""
    from mlir_swage import ir
    from mlir_swage.dialects import swage as swage_dialect

    source = _multiply_source()
    assert before in source
    source = source.replace(before, after)
    if "%chained" in source:
        source = source.replace("%mask, %product\n", "%mask, %chained\n")
    with ir.Context() as context:
        swage_dialect.register_dialects(context)
        module = ir.Module.parse(source)
        assert module.operation.verify()
        original = str(module)
        compiler = (
            native_swage._compile_fixed_host
            if backend == "cpu"
            else native_swage._compile_ptx
        )
        options = {"kernel_name": "multiply_kernel", "block_size": 128}
        if backend == "cuda":
            options["target"] = "sm_80"
        with pytest.raises(ValueError, match=diagnostic):
            compiler(module, **options)
        assert str(module) == original
