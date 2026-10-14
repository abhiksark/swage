// test/Dialect/Swage/fuse-maps.mlir
// RUN: swage-opt --swage-fuse-maps %s | FileCheck %s

// A mapped segment is a lazy view: the consumer applies the region of the
// map to each element it reads. Fusion writes that out. It moves the region
// of a map that has one consumer in front of the region of the consumer,
// and the map disappears.

// The region of the map runs ahead of the region of the reduction, and the
// captures of the map come ahead of the captures of the reduction.
// CHECK-LABEL: func.func @map_into_reduce(
// CHECK-SAME: %[[SCALE:[^ ]*]]: f32, %[[BIAS:[^ ]*]]: f32) -> f32 {
// CHECK-NEXT: %[[SEG:.*]] = swage.make_segment
// CHECK-NEXT: %[[SUM:.*]] = swage.reduce %[[SEG]] captures(%[[SCALE]], %[[BIAS]] : f32, f32) kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[X:.*]]: f32, %[[S:.*]]: f32, %[[B:.*]]: f32):
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.mulf %[[X]], %[[S]] : f32
// CHECK-NEXT: %[[SHIFTED:.*]] = arith.addf %[[PRODUCT]], %[[B]] : f32
// CHECK-NEXT: swage.yield %[[SHIFTED]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[SUM]] : f32
func.func @map_into_reduce(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index,
    %scale: f32, %bias: f32) -> f32 {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %scaled = swage.map %seg captures(%scale : f32)
      : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32, %s: f32):
    %product = arith.mulf %x, %s : f32
    swage.yield %product : f32
  }
  %sum = swage.reduce %scaled captures(%bias : f32) kind<sum>
      : !swage.segment<f32> -> f32 {
  ^bb0(%y: f32, %b: f32):
    %shifted = arith.addf %y, %b : f32
    swage.yield %shifted : f32
  }
  return %sum : f32
}

// A chain fuses in application order: the first map, the second map, then
// the store.
// CHECK-LABEL: func.func @chain_into_map_store(
// CHECK-SAME: %[[OUTPUT:[^ ]*]]: memref<?xf32>, %[[A:[^ ]*]]: f32, %[[B:[^ ]*]]: f32, %[[C:[^ ]*]]: f32) {
// CHECK-NEXT: %[[SEG:.*]] = swage.make_segment
// CHECK-NEXT: swage.map_store %[[SEG]], %[[OUTPUT]] captures(%[[A]], %[[B]], %[[C]] : f32, f32, f32) : !swage.segment<f32>, memref<?xf32> {
// CHECK-NEXT: ^bb0(%[[X:.*]]: f32, %[[P:.*]]: f32, %[[Q:.*]]: f32, %[[R:.*]]: f32):
// CHECK-NEXT: %[[SUM:.*]] = arith.addf %[[X]], %[[P]] : f32
// CHECK-NEXT: %[[PRODUCT:.*]] = arith.mulf %[[SUM]], %[[Q]] : f32
// CHECK-NEXT: %[[DIFFERENCE:.*]] = arith.subf %[[PRODUCT]], %[[R]] : f32
// CHECK-NEXT: swage.yield %[[DIFFERENCE]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return
func.func @chain_into_map_store(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index,
    %output: memref<?xf32>, %a: f32, %b: f32, %c: f32) {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %first = swage.map %seg captures(%a : f32)
      : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32, %p: f32):
    %sum = arith.addf %x, %p : f32
    swage.yield %sum : f32
  }
  %second = swage.map %first captures(%b : f32)
      : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32, %q: f32):
    %product = arith.mulf %x, %q : f32
    swage.yield %product : f32
  }
  swage.map_store %second, %output captures(%c : f32)
      : !swage.segment<f32>, memref<?xf32> {
  ^bb0(%x: f32, %r: f32):
    %difference = arith.subf %x, %r : f32
    swage.yield %difference : f32
  }
  return
}

// A map that yields its element leaves the consumer as it was, reading the
// operand segment of the map.
// CHECK-LABEL: func.func @identity_map(
// CHECK-NEXT: %[[SEG:.*]] = swage.make_segment
// CHECK-NEXT: %[[MAX:.*]] = swage.reduce %[[SEG]] kind<max> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[X:.*]]: f32):
// CHECK-NEXT: swage.yield %[[X]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return %[[MAX]] : f32
func.func @identity_map(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index) -> f32 {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %same = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  %max = swage.reduce %same kind<max> : !swage.segment<f32> -> f32 {
  ^bb0(%y: f32):
    swage.yield %y : f32
  }
  return %max : f32
}

// A map may yield a capture. The consumer then reads the capture where it
// read its element.
// CHECK-LABEL: func.func @map_yields_its_capture(
// CHECK-SAME: %[[FILL:[^ ]*]]: f32) -> f32 {
// CHECK-NEXT: %[[SEG:.*]] = swage.make_segment
// CHECK-NEXT: %{{.*}} = swage.reduce %[[SEG]] captures(%[[FILL]] : f32) kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %[[F:.*]]: f32):
// CHECK-NEXT: %[[TWICE:.*]] = arith.addf %[[F]], %[[F]] : f32
// CHECK-NEXT: swage.yield %[[TWICE]] : f32
func.func @map_yields_its_capture(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index,
    %fill: f32) -> f32 {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %filled = swage.map %seg captures(%fill : f32)
      : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32, %f: f32):
    swage.yield %f : f32
  }
  %sum = swage.reduce %filled kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%y: f32):
    %twice = arith.addf %y, %y : f32
    swage.yield %twice : f32
  }
  return %sum : f32
}

// The fused consumer takes the element type of the operand segment of the
// map.
// CHECK-LABEL: func.func @map_changes_the_element_type(
// CHECK-NEXT: %[[SEG:.*]] = swage.make_segment
// CHECK-NEXT: %{{.*}} = swage.reduce %[[SEG]] kind<sum> : !swage.segment<f32> -> i32 {
// CHECK-NEXT: ^bb0(%[[X:.*]]: f32):
// CHECK-NEXT: %[[I:.*]] = arith.fptosi %[[X]] : f32 to i32
// CHECK-NEXT: swage.yield %[[I]] : i32
func.func @map_changes_the_element_type(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index) -> i32 {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %integers = swage.map %seg : !swage.segment<f32> -> !swage.segment<i32> {
  ^bb0(%x: f32):
    %i = arith.fptosi %x : f32 to i32
    swage.yield %i : i32
  }
  %sum = swage.reduce %integers kind<sum> : !swage.segment<i32> -> i32 {
  ^bb0(%y: i32):
    swage.yield %y : i32
  }
  return %sum : i32
}

// A map with two consumers is not fused.
// CHECK-LABEL: func.func @two_consumers(
// CHECK: %[[SQUARES:.*]] = swage.map
// CHECK: swage.reduce %[[SQUARES]] kind<sum>
// CHECK: swage.reduce %[[SQUARES]] kind<max>
func.func @two_consumers(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index)
    -> (f32, f32) {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %squares = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    %square = arith.mulf %x, %x : f32
    swage.yield %square : f32
  }
  %sum = swage.reduce %squares kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%y: f32):
    swage.yield %y : f32
  }
  %max = swage.reduce %squares kind<max> : !swage.segment<f32> -> f32 {
  ^bb0(%y: f32):
    swage.yield %y : f32
  }
  return %sum, %max : f32, f32
}

// Two maps fuse into one map when the consumer of the second is not a
// region operation.
// CHECK-LABEL: func.func @maps_before_an_extent(
// CHECK-NEXT: %[[SEG:.*]] = swage.make_segment
// CHECK-NEXT: %[[MAPPED:.*]] = swage.map %[[SEG]] : !swage.segment<f32> -> !swage.segment<f32> {
// CHECK-NEXT: ^bb0(%[[X:.*]]: f32):
// CHECK-NEXT: %[[SQUARE:.*]] = arith.mulf %[[X]], %[[X]] : f32
// CHECK-NEXT: %[[NEGATED:.*]] = arith.negf %[[SQUARE]] : f32
// CHECK-NEXT: swage.yield %[[NEGATED]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: %{{.*}} = swage.extent %[[MAPPED]]
func.func @maps_before_an_extent(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index) -> index {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %first = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    %square = arith.mulf %x, %x : f32
    swage.yield %square : f32
  }
  %second = swage.map %first : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    %negated = arith.negf %x : f32
    swage.yield %negated : f32
  }
  %length = swage.extent %second : !swage.segment<f32>
  return %length : index
}

// The consumer reads the buffers, not the map, so a write between the two
// stays ahead of the fused consumer and the consumer sees it.
// CHECK-LABEL: func.func @write_between_map_and_consumer(
// CHECK-NEXT: %[[SEG:.*]] = swage.make_segment
// CHECK-NEXT: memref.store
// CHECK-NEXT: %{{.*}} = swage.reduce %[[SEG]] kind<sum>
func.func @write_between_map_and_consumer(
    %values: memref<?xf32>, %offsets: memref<?xi32>, %sid: index,
    %value: f32) -> f32 {
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %squares = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    %square = arith.mulf %x, %x : f32
    swage.yield %square : f32
  }
  memref.store %value, %values[%sid] : memref<?xf32>
  %sum = swage.reduce %squares kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%y: f32):
    swage.yield %y : f32
  }
  return %sum : f32
}
