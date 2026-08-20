// test/Conversion/SwageToGPU/ragged-softmax.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s --check-prefix=NOSYNC

// Three phases in one kernel, one CTA per segment. The emitter synchronizes
// nothing itself: gpu.all_reduce with uniform=true both broadcasts its result
// to every thread and fences one phase from the next.
module {
  func.func @ragged_softmax(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
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

// CHECK: gpu.module @ragged_softmax_module
// CHECK: gpu.func @ragged_softmax(%[[VALUES:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %[[SEGMENT_COUNT:[^)]+]]: i32) kernel
// CHECK: %[[SID:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK: %[[BLOCK:.*]] = arith.constant 128 : index
// One CTA-uniform range guard holds every phase, which is what makes the
// barriers inside the all-reduces legal.
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi slt, %[[SID]], %{{.*}} : index
// CHECK: scf.if %[[IN_RANGE]] {
// CHECK:   %[[START:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:   %[[END:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:   %[[FIRST:.*]] = arith.addi %[[START]], %[[THREAD]] : index
// Phase one: block-stride maximum.
// CHECK:   %[[MLOCAL:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[END]] step %[[BLOCK]] iter_args(%{{.*}}) -> (f32) {
// CHECK:     arith.maximumf
// CHECK:   }
// CHECK:   %[[MAX:.*]] = gpu.all_reduce maximumf %[[MLOCAL]] uniform
// Phase two: block-stride sum of the shifted exponentials, from the same first.
// CHECK:   %[[SLOCAL:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[END]] step %[[BLOCK]] iter_args(%{{.*}}) -> (f32) {
// CHECK:     %[[SLOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK:     %[[SCENTERED:.*]] = arith.subf %{{.*}}, %[[MAX]] : f32
// CHECK:     %[[SSCALED:.*]] = arith.mulf %[[SCENTERED]], %[[SLOG2E]] : f32
// CHECK:     math.exp2 %[[SSCALED]] : f32
// CHECK:   }
// CHECK:   %[[TOTAL:.*]] = gpu.all_reduce add %[[SLOCAL]] uniform
// Phase three stores unguarded. No thread-zero predicate may appear between
// the second all-reduce and the store loop.
// CHECK-NOT: arith.cmpi eq, %[[THREAD]]
// CHECK:   scf.for %[[I:.*]] = %[[FIRST]] to %[[END]] step %[[BLOCK]] {
// CHECK:     %[[INDEX:.*]] = arith.index_cast %[[I]] : index to i64
// CHECK:     %[[ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[INDEX]]]
// CHECK:     %[[TLOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK:     %[[TCENTERED:.*]] = arith.subf %{{.*}}, %[[MAX]] : f32
// CHECK:     %[[TSCALED:.*]] = arith.mulf %[[TCENTERED]], %[[TLOG2E]] : f32
// CHECK:     %[[TEXP:.*]] = math.exp2 %[[TSCALED]] : f32
// CHECK:     %[[NORMALIZED:.*]] = arith.divf %[[TEXP]], %[[TOTAL]] : f32
// The output address uses the per-element index, not the segment id, and
// reuses the same index_cast as the load. That is the whole ABI delta.
// CHECK:     %[[OUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[INDEX]]]
// CHECK:     llvm.store %[[NORMALIZED]], %[[OUT_ADDRESS]]
// CHECK:   }

// The emitter contributes no synchronization, no scratch buffer, and no
// shuffle of its own; all of that belongs to the all-reduce lowering.
// NOSYNC-NOT: gpu.barrier
// NOSYNC-NOT: memref.alloc
// NOSYNC-NOT: gpu.shuffle
