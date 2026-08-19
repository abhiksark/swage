// RUN: swage-opt %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of the core segment operations.

// CHECK-LABEL: func.func @program_id_roundtrip
func.func @program_id_roundtrip() -> index {
  // CHECK: %[[PID:.*]] = swage.program_id 0
  %pid = swage.program_id 0
  // CHECK: return %[[PID]] : index
  return %pid : index
}

// CHECK-LABEL: func.func @segment_roundtrip
func.func @segment_roundtrip(%values: memref<?xf32>, %offsets: memref<?xi32>) -> index {
  // CHECK: %[[SID:.*]] = swage.segment_id 0
  %sid = swage.segment_id 0
  // CHECK: %[[SEG:.*]] = swage.make_segment %{{.*}}, %{{.*}}, %[[SID]] : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %seg = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  // CHECK: %[[N:.*]] = swage.extent %[[SEG]] : !swage.segment<f32>
  %n = swage.extent %seg : !swage.segment<f32>
  // CHECK: return %[[N]] : index
  return %n : index
}

// CHECK-LABEL: func.func @integer_segment
func.func @integer_segment(%values: memref<?xi32>, %offsets: memref<?xi64>, %sid: index) -> !swage.segment<i32> {
  // CHECK: swage.make_segment %{{.*}}, %{{.*}}, %{{.*}} : memref<?xi32>, memref<?xi64>, index -> !swage.segment<i32>
  %seg = swage.make_segment %values, %offsets, %sid : memref<?xi32>, memref<?xi64>, index -> !swage.segment<i32>
  return %seg : !swage.segment<i32>
}

// CHECK-LABEL: func.func @half_precision_segments
func.func @half_precision_segments(%v16: memref<?xf16>, %vbf: memref<?xbf16>, %offsets: memref<?xi32>, %sid: index) -> (index, index) {
  // CHECK: swage.make_segment %{{.*}}, %{{.*}}, %{{.*}} : memref<?xf16>, memref<?xi32>, index -> !swage.segment<f16>
  %s16 = swage.make_segment %v16, %offsets, %sid : memref<?xf16>, memref<?xi32>, index -> !swage.segment<f16>
  // CHECK: swage.make_segment %{{.*}}, %{{.*}}, %{{.*}} : memref<?xbf16>, memref<?xi32>, index -> !swage.segment<bf16>
  %sbf = swage.make_segment %vbf, %offsets, %sid : memref<?xbf16>, memref<?xi32>, index -> !swage.segment<bf16>
  %e16 = swage.extent %s16 : !swage.segment<f16>
  %ebf = swage.extent %sbf : !swage.segment<bf16>
  return %e16, %ebf : index, index
}

// CHECK-LABEL: func.func @bf16_map
func.func @bf16_map(%s: !swage.segment<bf16>) -> !swage.segment<bf16> {
  // CHECK: swage.map %{{.*}} : !swage.segment<bf16> -> !swage.segment<bf16>
  %r = swage.map %s : !swage.segment<bf16> -> !swage.segment<bf16> {
  ^bb0(%x: bf16):
    %y = arith.negf %x : bf16
    swage.yield %y : bf16
  }
  return %r : !swage.segment<bf16>
}
