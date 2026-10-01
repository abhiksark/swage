# python/tests/mlir/test_bindings.py
"""Integration tests for constructing Swage IR with Python bindings."""

import pathlib

import pytest
from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import arith, builtin, func, math, swage


def test_native_module_reports_the_pinned_llvm_version():
    """Expose the linked LLVM version and keep it equal to the pin."""
    repo_root = pathlib.Path(__file__).resolve().parents[3]
    pin = (repo_root / "cmake" / "llvm-version.txt").read_text().strip()

    assert native_swage.__llvm_version__ == pin.removeprefix("llvmorg-")


def test_builds_and_round_trips_every_swage_operation():
    """Build, verify, print, and reparse all Swage operations."""
    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        f32 = ir.F32Type.get()
        index = ir.IndexType.get()
        segment = swage.SegmentType.get(f32)
        dynamic = ir.ShapedType.get_dynamic_size()
        values = ir.MemRefType.get([dynamic], f32)
        offsets = ir.MemRefType.get(
            [dynamic], ir.IntegerType.get_signless(32)
        )
        module = builtin.ModuleOp()

        with ir.InsertionPoint(module.body):
            kernel = func.FuncOp(
                "kernel", ([values, offsets, values, f32], [])
            )
        with ir.InsertionPoint(kernel.add_entry_block()):
            values_arg, offsets_arg, output, scale = kernel.arguments
            swage.ProgramIdOp(index, 0)
            segment_id = swage.SegmentIdOp(index, 0).result
            input_segment = swage.MakeSegmentOp(
                segment, values_arg, offsets_arg, segment_id
            ).result
            swage.ExtentOp(index, input_segment)

            mapped = swage.MapOp(segment, input_segment, [scale])
            mapped.body.blocks.append(f32, f32)
            with ir.InsertionPoint(mapped.body.blocks[0]):
                scaled = arith.MulFOp(
                    mapped.body.blocks[0].arguments[0],
                    mapped.body.blocks[0].arguments[1],
                )
                swage.YieldOp(math.ExpOp(scaled.result).result)

            reductions = []
            for kind in ("sum", "max", "min"):
                reduction = swage.ReduceOp(
                    f32,
                    mapped.result,
                    [scale],
                    ir.Attribute.parse(f"#swage.reduction_kind<{kind}>"),
                )
                reduction.body.blocks.append(f32, f32)
                with ir.InsertionPoint(reduction.body.blocks[0]):
                    scaled = arith.MulFOp(
                        reduction.body.blocks[0].arguments[0],
                        reduction.body.blocks[0].arguments[1],
                    )
                    swage.YieldOp(scaled.result)
                reductions.append(reduction.result)

            store = swage.MapStoreOp(mapped.result, output, reductions)
            store.body.blocks.append(f32, f32, f32, f32)
            with ir.InsertionPoint(store.body.blocks[0]):
                numerator = math.ExpOp(store.body.blocks[0].arguments[0])
                denominator = arith.AddFOp(
                    store.body.blocks[0].arguments[1],
                    store.body.blocks[0].arguments[2],
                )
                denominator = arith.AddFOp(
                    denominator.result, store.body.blocks[0].arguments[3]
                )
                quotient = arith.DivFOp(numerator.result, denominator.result)
                swage.YieldOp(quotient.result)
            func.ReturnOp([])

        assert module.operation.verify()
        reparsed = ir.Module.parse(str(module))
        assert reparsed.operation.verify()


def test_embeds_memref_and_vector_dialects():
    """Build upstream memref and vector operations from the pinned package."""
    from mlir_swage.dialects import memref, vector

    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        f32 = ir.F32Type.get()
        buffer = ir.MemRefType.get([4], f32)
        tile = ir.VectorType.get([4], f32)
        module = builtin.ModuleOp()

        with ir.InsertionPoint(module.body):
            kernel = func.FuncOp("kernel", ([], []))
        with ir.InsertionPoint(kernel.add_entry_block()):
            memref.AllocOp(buffer, [], [])
            scalar = arith.ConstantOp(f32, 0.0)
            vector.BroadcastOp(tile, scalar.result)
            func.ReturnOp([])

        assert module.operation.verify()


def test_rejects_make_segment_element_type_mismatch():
    """Reject a segment whose element type differs from the values buffer."""
    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        with pytest.raises(
            ir.MLIRError,
            match=(
                "values element type 'f32' does not match segment element type "
                "'i32'"
            ),
        ):
            ir.Module.parse(
                """
                module {
                  func.func @bad(
                      %values: memref<?xf32>, %offsets: memref<?xi32>,
                      %segment_id: index) {
                    %segment = swage.make_segment %values, %offsets, %segment_id
                        : memref<?xf32>, memref<?xi32>, index
                        -> !swage.segment<i32>
                    return
                  }
                }
                """
            )


def test_segment_type_constructor_builds_the_parsed_type():
    """Build `!swage.segment<T>` without going through the parser."""
    with ir.Context() as context:
        swage.register_dialects(context)
        f32 = ir.F32Type.get()
        parsed = ir.Type.parse("!swage.segment<f32>")

        built = swage.SegmentType.get(f32)

        assert built == parsed
        assert str(built) == "!swage.segment<f32>"
        assert built.element_type == f32
        assert isinstance(built, swage.SegmentType)
        # A parsed type arrives as the same class, through its type ID.
        assert isinstance(parsed, swage.SegmentType)
        assert swage.SegmentType.isinstance(parsed)
        assert not swage.SegmentType.isinstance(f32)
        with pytest.raises(ValueError, match="Cannot cast type to SegmentType"):
            swage.SegmentType(f32)


def test_segment_type_constructor_rejects_a_non_scalar_element():
    """Report the dialect's own diagnostic instead of asserting."""
    with ir.Context() as context:
        swage.register_dialects(context)
        buffer = ir.Type.parse("memref<2xf32>")

        with pytest.raises(
            ValueError,
            match="segment element type must be an integer or float type, "
            "got 'memref<2xf32>'",
        ):
            swage.SegmentType.get(buffer)


def test_segment_type_constructor_needs_the_dialect_loaded():
    """Fail with a message where MLIR itself would abort the process."""
    with ir.Context():
        with pytest.raises(
            ValueError,
            match="the swage dialect is not loaded in this context",
        ):
            swage.SegmentType.get(ir.F32Type.get())
