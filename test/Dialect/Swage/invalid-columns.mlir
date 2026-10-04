// test/Dialect/Swage/invalid-columns.mlir
// RUN: swage-opt %s --split-input-file --verify-diagnostics

// A rank-two buffer holds rows, and a segment is a run of scalars: a
// make_segment of rows names the column, and one of scalars names none.

func.func @rows_without_a_column(
    %values: memref<?x?xf32>, %offsets: memref<?xi32>, %sid: index) {
  // expected-error @below {{'swage.make_segment' op takes a column exactly when its values have rank two, got 'memref<?x?xf32>' without a column}}
  %segment = swage.make_segment %values, %offsets, %sid
      : memref<?x?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  return
}

// -----

func.func @column_of_scalars(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index,
    %col: index) {
  // expected-error @below {{'swage.make_segment' op takes a column exactly when its values have rank two, got 'memref<?xf32>' with a column}}
  %segment = swage.make_segment %values, %offsets, %sid column(%col)
      : memref<?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
  return
}

// -----

func.func @rank_three_values(
    %values: memref<?x?x?xf32>, %offsets: memref<?xi32>, %sid: index) {
  // expected-error @below {{operand #0 must be 1D/2D memref of any type values}}
  %segment = swage.make_segment %values, %offsets, %sid
      : memref<?x?x?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  return
}

// -----

func.func @rank_three_output(
    %segment: !swage.segment<f32>, %output: memref<?x?x?xf32>) {
  // expected-error @below {{operand #1 must be 1D/2D memref of integer or floating-point values}}
  swage.map_store %segment, %output : !swage.segment<f32>, memref<?x?x?xf32> {
  ^bb0(%value: f32):
    swage.yield %value : f32
  }
  return
}
