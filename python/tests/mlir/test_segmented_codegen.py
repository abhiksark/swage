# python/tests/mlir/test_segmented_codegen.py
"""Native tests for segmented-reduction NVPTX compilation."""

from mlir_swage import ir
from mlir_swage._mlir_libs._swageDialectsNanobind import swage as native_swage
from mlir_swage.dialects import swage

SEGMENTED_SUM = """
module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
"""


def test_compiles_segmented_sum_to_deterministic_ptx():
    """Preserve the source module while emitting one CTA reduction."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_SUM)
        original = module.operation.get_asm(enable_debug_info=False)

        first = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_80",
        )
        second = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_80",
        )

        assert first == second
        lowered, ptx = first
        assert "swage." not in lowered
        assert "gpu.all_reduce" not in lowered
        assert "llvm.func @segmented_sum" in lowered
        assert '#nvvm.target<chip = "sm_80">' in lowered
        assert ".target sm_80" in ptx
        assert ".entry segmented_sum" in ptx
        assert "ld.global.b32" in ptx
        assert "st.global.b32" in ptx
        assert module.operation.get_asm(enable_debug_info=False) == original


SEGMENTED_EXPONENTIAL_SUM = """
module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %log2e = arith.constant 1.44269504 : f32
      %scaled = arith.mulf %value, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
"""


def test_region_exponential_compiles_without_libdevice():
    """Emit a native ex2 rather than an unresolvable libdevice call."""
    with ir.Context() as context:
        swage.register_dialects(context)
        module = ir.Module.parse(SEGMENTED_EXPONENTIAL_SUM)

        lowered, ptx = native_swage._compile_segmented_reduction_ptx(
            module,
            kernel_name="segmented_sum",
            block_size=128,
            target="sm_80",
        )

        assert "llvm.intr.exp2" in lowered
        assert "__nv_exp2f" not in lowered
        assert ".extern .func" not in ptx
        assert "ex2.approx.f32" in ptx
