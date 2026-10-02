// test/Conversion/SwagePlanToGPU/tasks.mlir
// Plan functions written by hand, one per form of the task operation. The
// conversion gives each a gpu.module named after it, turns every buffer into
// a pointer, and applies every bound the operation carries.
//
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s \
// RUN:   --implicit-check-not=swage --implicit-check-not=func.func

module {
  // One block per segment: the block index is the segment id, compared with
  // the segment count. The loaded range is clamped to the value count, the
  // block reduces as a whole, and thread 0 stores.
  // CHECK: gpu.module @direct_module {
  // CHECK-NEXT: gpu.func @direct(%[[VALUES:.*]]: !llvm.ptr, %[[OFFSETS:.*]]: !llvm.ptr, %[[OUTPUT:.*]]: !llvm.ptr, %[[VALUE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) kernel attributes {nvvm.reqntid = array<i32: 128, 1, 1>} {
  // CHECK-NEXT: %[[BLOCK:.*]] = gpu.block_id x
  // CHECK-NEXT: %[[THREAD:.*]] = gpu.thread_id x
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[C128:.*]] = arith.constant 128 : index
  // CHECK-NEXT: %[[SEGMENTS:.*]] = arith.index_cast %[[SEGMENT_COUNT]] : i32 to index
  // CHECK-NEXT: %[[HAS_TASK:.*]] = arith.cmpi slt, %[[BLOCK]], %[[SEGMENTS]] : index
  // CHECK-NEXT: scf.if %[[HAS_TASK]] {
  // CHECK: %[[START_WORD:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
  // CHECK: %[[END_WORD:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
  // CHECK: %[[START_FLOOR:.*]] = arith.maxsi %[[START_WORD]], %{{.*}} : i32
  // CHECK-NEXT: %[[START:.*]] = arith.minsi %[[START_FLOOR]], %[[VALUE_COUNT]] : i32
  // CHECK-NEXT: %[[END_FLOOR:.*]] = arith.maxsi %[[END_WORD]], %[[START]] : i32
  // CHECK-NEXT: %[[END:.*]] = arith.minsi %[[END_FLOOR]], %[[VALUE_COUNT]] : i32
  // CHECK: %[[LOCAL:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C128]] iter_args(
  // CHECK: llvm.getelementptr %[[VALUES]][
  // CHECK: %[[TOTAL:.*]] = gpu.all_reduce add %[[LOCAL]] uniform {
  // CHECK: %[[LEADER:.*]] = arith.cmpi eq, %[[THREAD]], %[[C0]] : index
  // CHECK-NEXT: scf.if %[[LEADER]] {
  // CHECK-NEXT: %[[SLOT:.*]] = llvm.getelementptr %[[OUTPUT]][
  // CHECK-NEXT: llvm.store %[[TOTAL]], %[[SLOT]] : f32, !llvm.ptr
  // CHECK: gpu.return
  func.func @direct(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  // One block per task: the block index is compared with the task count,
  // the segment id is loaded from the task buffer and compared with the
  // segment count, and an id that fails reads offsets[0] twice and stores
  // nothing. A warp task combines through a shuffle tree.
  // CHECK: gpu.module @warp_tasks_module {
  // CHECK-NEXT: gpu.func @warp_tasks(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %[[IDS:.*]]: !llvm.ptr, %{{.*}}: i32, %[[TASK_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) kernel attributes {nvvm.reqntid = array<i32: 32, 1, 1>} {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[TASKS:.*]] = arith.index_cast %[[TASK_COUNT]] : i32 to index
  // CHECK-NEXT: %[[HAS_TASK:.*]] = arith.cmpi slt, %{{.*}}, %[[TASKS]] : index
  // CHECK-NEXT: scf.if %[[HAS_TASK]] {
  // CHECK: %[[ID_ADDRESS:.*]] = llvm.getelementptr %[[IDS]][
  // CHECK-NEXT: %[[ID_WORD:.*]] = llvm.load %[[ID_ADDRESS]] : !llvm.ptr -> i32
  // CHECK-NEXT: %[[IN_RANGE:.*]] = arith.cmpi ult, %[[ID_WORD]], %[[SEGMENT_COUNT]] : i32
  // CHECK-NEXT: %[[ID:.*]] = arith.index_cast %[[ID_WORD]] : i32 to index
  // CHECK: arith.select %[[IN_RANGE]], %[[ID]], %[[C0]] : index
  // CHECK: arith.select %[[IN_RANGE]], %{{.*}}, %[[C0]] : index
  // CHECK-COUNT-5: gpu.shuffle xor
  // CHECK: %[[LEADER:.*]] = arith.cmpi eq,
  // CHECK-NEXT: %[[MAY_STORE:.*]] = arith.andi %[[LEADER]], %[[IN_RANGE]] : i1
  // CHECK-NEXT: scf.if %[[MAY_STORE]] {
  func.func @warp_tasks(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 32 : i32} {
    swage_plan.tasks policy<warp>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  // A block task combines across the block.
  // CHECK: gpu.module @cta_tasks_module {
  // CHECK-NEXT: gpu.func @cta_tasks({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>} {
  // CHECK: arith.cmpi ult,
  // CHECK: gpu.all_reduce add %{{.*}} uniform {
  func.func @cta_tasks(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}
