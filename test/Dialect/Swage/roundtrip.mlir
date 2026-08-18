// RUN: swage-opt %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of the core segment operations.

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
