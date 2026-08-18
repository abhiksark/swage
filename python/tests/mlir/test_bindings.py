# python/tests/mlir/test_bindings.py
"""Integration tests for constructing Swage IR with Python bindings."""

import pytest
from mlir_swage import ir
from mlir_swage.dialects import arith, builtin, func, math, swage


def test_builds_and_round_trips_every_swage_operation():
    """Build, verify, print, and reparse all Swage operations."""
    with ir.Context() as context, ir.Location.unknown():
        swage.register_dialects(context)
        f32 = ir.F32Type.get()
        index = ir.IndexType.get()
        segment = ir.Type.parse("!swage.segment<f32>")
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
