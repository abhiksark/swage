// test/Dialect/Swage/effects.mlir
// RUN: swage-opt --cse %s | FileCheck %s
// RUN: swage-opt --canonicalize %s | FileCheck %s --check-prefix=CANON
// RUN: swage-opt --loop-invariant-code-motion %s \
// RUN:   | FileCheck %s --check-prefix=LICM

// Segment consumers read the values and offsets buffers behind the segment
// handle. They declare that read, so common subexpression elimination must not
// merge two of them across a write, and may merge them when nothing writes in
// between. Loop-invariant code motion must not hoist one out of a loop that
// writes the buffers.

// CHECK-LABEL: func.func @extent_across_offsets_write(
// CHECK: %[[BEFORE:.*]] = swage.extent
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = swage.extent
// CHECK: return %[[BEFORE]], %[[AFTER]]
func.func @extent_across_offsets_write(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index, %v: i32)
    -> (index, index) {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %before = swage.extent %seg : !swage.segment<f32>
  %c1 = arith.constant 1 : index
  memref.store %v, %offsets[%c1] : memref<?xi32>
  %after = swage.extent %seg : !swage.segment<f32>
  return %before, %after : index, index
}

// CHECK-LABEL: func.func @reduce_across_values_write(
// CHECK: %[[BEFORE:.*]] = swage.reduce
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = swage.reduce
// CHECK: return %[[BEFORE]], %[[AFTER]]
func.func @reduce_across_values_write(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index, %x: f32)
    -> (f32, f32) {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %before = swage.reduce %seg kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%e: f32):
    swage.yield %e : f32
  }
  %c0 = arith.constant 0 : index
  memref.store %x, %values[%c0] : memref<?xf32>
  %after = swage.reduce %seg kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%e: f32):
    swage.yield %e : f32
  }
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @map_across_values_write(
// CHECK: %[[BEFORE:.*]] = swage.map
// CHECK: memref.store
// CHECK: %[[AFTER:.*]] = swage.map
// CHECK: return %[[BEFORE]], %[[AFTER]]
func.func @map_across_values_write(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index, %x: f32)
    -> (!swage.segment<f32>, !swage.segment<f32>) {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %before = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%e: f32):
    swage.yield %e : f32
  }
  %c0 = arith.constant 0 : index
  memref.store %x, %values[%c0] : memref<?xf32>
  %after = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%e: f32):
    swage.yield %e : f32
  }
  return %before, %after : !swage.segment<f32>, !swage.segment<f32>
}

// A map_store between two readers is itself the write.

// CHECK-LABEL: func.func @extent_across_map_store(
// CHECK: %[[BEFORE:.*]] = swage.extent
// CHECK: swage.map_store
// CHECK: %[[AFTER:.*]] = swage.extent
// CHECK: return %[[BEFORE]], %[[AFTER]]
func.func @extent_across_map_store(
    %seg: !swage.segment<f32>, %out: memref<?xf32>) -> (index, index) {
  %before = swage.extent %seg : !swage.segment<f32>
  swage.map_store %seg, %out : !swage.segment<f32>, memref<?xf32> {
  ^bb0(%e: f32):
    swage.yield %e : f32
  }
  %after = swage.extent %seg : !swage.segment<f32>
  return %before, %after : index, index
}

// CHECK-LABEL: func.func @extent_without_write(
// CHECK: %[[ONLY:.*]] = swage.extent
// CHECK-NOT: swage.extent
// CHECK: return %[[ONLY]], %[[ONLY]]
func.func @extent_without_write(%seg: !swage.segment<f32>) -> (index, index) {
  %first = swage.extent %seg : !swage.segment<f32>
  %second = swage.extent %seg : !swage.segment<f32>
  return %first, %second : index, index
}

// CHECK-LABEL: func.func @reduce_without_write(
// CHECK: %[[ONLY:.*]] = swage.reduce
// CHECK-NOT: swage.reduce
// CHECK: return %[[ONLY]], %[[ONLY]]
func.func @reduce_without_write(%seg: !swage.segment<f32>) -> (f32, f32) {
  %first = swage.reduce %seg kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%e: f32):
    swage.yield %e : f32
  }
  %second = swage.reduce %seg kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%e: f32):
    swage.yield %e : f32
  }
  return %first, %second : f32, f32
}

// A read is not a reason to keep an unused result, under either pass.

// CHECK-LABEL: func.func @unused_extent(
// CHECK-NOT: swage.extent
// CHECK: return
// CANON-LABEL: func.func @unused_extent(
// CANON-NOT: swage.extent
// CANON: return
func.func @unused_extent(%seg: !swage.segment<f32>) {
  %unused = swage.extent %seg : !swage.segment<f32>
  return
}

// LICM-LABEL: func.func @extent_in_loop_with_offsets_write(
// LICM: scf.for
// LICM-NEXT: swage.extent
func.func @extent_in_loop_with_offsets_write(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index, %v: i32,
    %sink: memref<?xindex>, %count: index) {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %count step %c1 {
    %length = swage.extent %seg : !swage.segment<f32>
    memref.store %length, %sink[%i] : memref<?xindex>
    memref.store %v, %offsets[%c1] : memref<?xi32>
  }
  return
}

// The declared read does not hide a region's own effects. Two reductions
// whose regions write memory stay two, and an unused one stays alive.

memref.global "private" @scratch : memref<1xf32> = dense<[0.0]>

// CHECK-LABEL: func.func @reduce_with_writing_region(
// CHECK: %[[FIRST:.*]] = swage.reduce
// CHECK: %[[SECOND:.*]] = swage.reduce
// CHECK: return %[[FIRST]], %[[SECOND]]
func.func @reduce_with_writing_region(%seg: !swage.segment<f32>) -> (f32, f32) {
  %first = swage.reduce %seg kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%e: f32):
    %scratch = memref.get_global @scratch : memref<1xf32>
    %c0 = arith.constant 0 : index
    memref.store %e, %scratch[%c0] : memref<1xf32>
    swage.yield %e : f32
  }
  %second = swage.reduce %seg kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%e: f32):
    %scratch = memref.get_global @scratch : memref<1xf32>
    %c0 = arith.constant 0 : index
    memref.store %e, %scratch[%c0] : memref<1xf32>
    swage.yield %e : f32
  }
  return %first, %second : f32, f32
}

// CHECK-LABEL: func.func @unused_reduce_with_writing_region(
// CHECK: swage.reduce
// CHECK: memref.store
// CHECK: return
func.func @unused_reduce_with_writing_region(%seg: !swage.segment<f32>) {
  %unused = swage.reduce %seg kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%e: f32):
    %scratch = memref.get_global @scratch : memref<1xf32>
    %c0 = arith.constant 0 : index
    memref.store %e, %scratch[%c0] : memref<1xf32>
    swage.yield %e : f32
  }
  return
}
