// RUN: swage-opt %s --split-input-file --verify-diagnostics

func.func @element_type_mismatch(%values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index) {
  // expected-error @below {{values element type 'f32' does not match segment element type 'i32'}}
  %seg = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<i32>
  return
}

// -----

func.func @negative_axis() {
  // expected-error @below {{attribute 'axis' failed to satisfy constraint}}
  %sid = swage.segment_id -1
  return
}

// -----

func.func @offsets_not_integer(%values: memref<?xf32>, %offsets: memref<?xf32>, %sid: index) {
  // expected-error @below {{operand #1 must be}}
  %seg = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xf32>, index -> !swage.segment<f32>
  return
}

// -----

// Segment element types are restricted to integers and floats.
// expected-error @below {{segment element type must be an integer or float type}}
func.func private @bad_element_type(%s: !swage.segment<memref<2xf32>>)
