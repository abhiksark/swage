// test/Dialect/Swage/roles.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s

// A segment function declares what each argument is. The lowerings find the
// buffers and the counts through these roles and not through positions, so
// the arguments may come in any order.

// CHECK-LABEL: func.func @canonical_order(
// CHECK-SAME: %{{[^:]+}}: memref<?xf32> {swage.role = #swage.role<values>}
// CHECK-SAME: %{{[^:]+}}: memref<?xi32> {swage.role = #swage.role<offsets>}
// CHECK-SAME: %{{[^:]+}}: memref<?xf32> {swage.role = #swage.role<output>}
// CHECK-SAME: %{{[^:]+}}: i32 {swage.role = #swage.role<value_count>}
// CHECK-SAME: %{{[^:]+}}: i32 {swage.role = #swage.role<segment_count>}
func.func @canonical_order(
    %values: memref<?xf32> {swage.role = #swage.role<values>},
    %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
    %output: memref<?xf32> {swage.role = #swage.role<output>},
    %value_count: i32 {swage.role = #swage.role<value_count>},
    %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
  return
}

// CHECK-LABEL: func.func @counts_first(
// CHECK-SAME: %{{[^:]+}}: i32 {swage.role = #swage.role<segment_count>}
// CHECK-SAME: %{{[^:]+}}: i32 {swage.role = #swage.role<value_count>}
// CHECK-SAME: %{{[^:]+}}: memref<?xf32> {swage.role = #swage.role<output>}
func.func @counts_first(
    %segment_count: i32 {swage.role = #swage.role<segment_count>},
    %value_count: i32 {swage.role = #swage.role<value_count>},
    %output: memref<?xf32> {swage.role = #swage.role<output>},
    %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
    %values: memref<?xf32> {swage.role = #swage.role<values>}) {
  return
}

// The dialect checks each role against the type that carries it and nothing
// more. Other element and offset types than the lowerings admit are legal
// here, as they are for swage.make_segment.
// CHECK-LABEL: func.func @other_widths(
// CHECK-SAME: memref<?xi16> {swage.role = #swage.role<values>}
// CHECK-SAME: memref<?xi64> {swage.role = #swage.role<offsets>}
// CHECK-SAME: i64 {swage.role = #swage.role<value_count>}
func.func @other_widths(
    %values: memref<?xi16> {swage.role = #swage.role<values>},
    %offsets: memref<?xi64> {swage.role = #swage.role<offsets>},
    %value_count: i64 {swage.role = #swage.role<value_count>}) {
  return
}

// A declaration carries roles too.
// CHECK-LABEL: func.func private @declaration(
// CHECK-SAME: memref<?xf32> {swage.role = #swage.role<values>}
func.func private @declaration(
    memref<?xf32> {swage.role = #swage.role<values>},
    memref<?xi32> {swage.role = #swage.role<offsets>})
