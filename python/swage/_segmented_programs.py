# python/swage/_segmented_programs.py
"""Canonical private MLIR programs for segmented qualification."""


def _semantic_module(kind):
    """Return the canonical private qualification module."""
    if kind not in {"sum", "max"}:
        raise ValueError("reduction kind must be 'sum' or 'max'")
    return f"""
module {{
  func.func @segmented_{kind}(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>) {{
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<{kind}>
        : !swage.segment<f32> -> f32 {{
    ^bb0(%value: f32):
      swage.yield %value : f32
    }}
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }}
}}
"""


_SOFTMAX_MODULE = """
module {
  func.func @ragged_softmax(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32, %m: f32, %t: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }
    return
  }
}
"""
