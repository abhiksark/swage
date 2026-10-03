// test/Conversion/SwagePlanToGPU/column-groups.mlir
// The row-stripe tile of rank-two values: a task operation of policy<cta>
// over rows of features, and the partial and merge kernels of the split,
// at the end of the file. A block runs items, each a task and a group of W
// adjacent columns, in a loop from the block index to task_count * groups
// by the grid size. W is the smallest power of two that is at least the
// feature count, capped at the subgroup width. Thread t owns column
// (t mod 32) mod W of the group and is one row stripe of it.
//
// - Each reduction stage folds the stripe of a thread into one scalar, the
//   one iteration argument of its loop, whatever the number of columns.
// - The stripes of one column combine in two steps. A butterfly of five
//   shuffles runs in every lane, and a select keeps the combination only
//   for an offset of at least W, which pairs lanes of one column. The
//   threads then exchange their results through the one workgroup buffer of
//   the kernel, block_threads elements, between two barriers, and every
//   thread combines the results of its column from each subgroup as a
//   pairwise tree.
// - Lane and segment validity act on addresses and on the final store only,
//   so no shuffle or barrier sits under thread-dependent control flow.
// - The thread of stripe zero of a column stores its result at
//   output[segment * D + column], when the column is below D and the
//   segment id loaded from the task buffer is below the segment count.
//
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s --check-prefix=MEAN
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s --check-prefix=SOFTMAX
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s --check-prefix=DIRECT
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s --check-prefix=PARTIAL
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s --check-prefix=MERGE

// CHECK-LABEL: gpu.func @rows_sum(
// CHECK-SAME: %[[VALUES:[^:]+]]: !llvm.ptr, %[[OFFSETS:[^:]+]]: !llvm.ptr, %[[OUTPUT:[^:]+]]: !llvm.ptr, %[[IDS:[^:]+]]: !llvm.ptr, %[[VALUE_COUNT:[^:]+]]: i32, %[[TASK_COUNT:[^:]+]]: i32, %[[SEGMENT_COUNT:[^:]+]]: i32, %[[FEATURE_COUNT:[^:]+]]: i32) workgroup(%[[EXCHANGE:[^ ]+]] : memref<128xf32, #gpu.address_space<workgroup>>) kernel attributes {nvvm.reqntid = array<i32: 128, 1, 1>} {
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// The column-group width: 1, then 2, 4, 8, 16, and 32 for a feature count
// above half of each.
// CHECK: %[[FEATURES:.*]] = arith.index_cast %[[FEATURE_COUNT]] : i32 to index
// CHECK: arith.cmpi sgt, %[[FEATURES]], %{{.*}} : index
// CHECK: arith.cmpi sgt, %[[FEATURES]], %{{.*}} : index
// CHECK: arith.cmpi sgt, %[[FEATURES]], %{{.*}} : index
// CHECK: arith.cmpi sgt, %[[FEATURES]], %{{.*}} : index
// CHECK: %[[WIDE:.*]] = arith.cmpi sgt, %[[FEATURES]], %{{.*}} : index
// CHECK: %[[WIDTH:.*]] = arith.select %[[WIDE]], %{{.*}}, %{{.*}} : index
// A feature count of zero or below has no column group.
// CHECK: %[[HAS_COLUMNS:.*]] = arith.cmpi sgt, %[[FEATURES]], %{{.*}} : index
// CHECK: %[[SOME_GROUPS:.*]] = arith.divui %{{.*}}, %[[WIDTH]] : index
// CHECK: %[[GROUPS:.*]] = arith.select %[[HAS_COLUMNS]], %[[SOME_GROUPS]], %{{.*}} : index
// The stripe of this thread and the elements between two of its rows.
// CHECK: %[[LANE:.*]] = arith.remui %[[THREAD]], %{{.*}} : index
// CHECK: %[[SUBGROUP:.*]] = arith.divui %[[THREAD]], %{{.*}} : index
// CHECK: %[[COLUMN:.*]] = arith.remui %[[LANE]], %[[WIDTH]] : index
// CHECK: %[[STRIPE:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: %[[STRIPES:.*]] = arith.divui %{{.*}}, %[[WIDTH]] : index
// CHECK: %[[ROW_STRIDE:.*]] = arith.muli %[[STRIPES]], %[[FEATURES]] : index
// The item loop, from the block index to task_count * groups by the grid.
// CHECK: %[[TASKS:.*]] = arith.index_cast %[[TASK_COUNT]] : i32 to index
// CHECK: %[[BLOCK:.*]] = gpu.block_id x
// CHECK: %[[GRID:.*]] = gpu.grid_dim x
// CHECK: %[[ITEMS:.*]] = arith.muli %[[TASKS]], %[[GROUPS]] : index
// CHECK: scf.for %[[ITEM:.*]] = %[[BLOCK]] to %[[ITEMS]] step %[[GRID]] {
// CHECK-NEXT: %[[TASK:.*]] = arith.divui %[[ITEM]], %[[GROUPS]] : index
// CHECK-NEXT: %[[GROUP:.*]] = arith.remui %[[ITEM]], %[[GROUPS]] : index
// The segment is loaded from the task buffer and its rows are clamped.
// CHECK: %[[SEGMENT_WORD:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK-NEXT: %[[SEGMENT_IN_RANGE:.*]] = arith.cmpi ult, %[[SEGMENT_WORD]], %[[SEGMENT_COUNT]] : i32
// CHECK-NEXT: %[[SEGMENT:.*]] = arith.index_cast %[[SEGMENT_WORD]] : i32 to index
// CHECK: arith.minsi %{{.*}}, %[[VALUE_COUNT]] : i32
// CHECK: arith.minsi %{{.*}}, %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[START:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT: %[[END:.*]] = arith.index_cast %{{.*}} : i32 to index
// The column of this thread in the group, and the first element of its
// stripe, which is the end for a column beyond the features.
// CHECK-NEXT: %[[GROUP_START:.*]] = arith.muli %[[GROUP]], %[[WIDTH]] : index
// CHECK-NEXT: %[[OWN_COLUMN:.*]] = arith.addi %[[GROUP_START]], %[[COLUMN]] : index
// CHECK-NEXT: %[[COLUMN_IN_RANGE:.*]] = arith.cmpi ult, %[[OWN_COLUMN]], %[[FEATURES]] : index
// CHECK-NEXT: %[[FIRST_ROW:.*]] = arith.addi %[[START]], %[[STRIPE]] : index
// CHECK-NEXT: %[[FIRST_ROW_START:.*]] = arith.muli %[[FIRST_ROW]], %[[FEATURES]] : index
// CHECK-NEXT: %[[FIRST_OWN:.*]] = arith.addi %[[FIRST_ROW_START]], %[[OWN_COLUMN]] : index
// CHECK-NEXT: %[[LAST:.*]] = arith.muli %[[END]], %[[FEATURES]] : index
// CHECK-NEXT: %[[FIRST:.*]] = arith.select %[[COLUMN_IN_RANGE]], %[[FIRST_OWN]], %[[LAST]] : index
// One scalar per thread and stage.
// CHECK: %[[STRIPE_SUM:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[LAST]] step %[[ROW_STRIDE]] iter_args(%{{.*}} = %{{.*}}) -> (f32) {
// CHECK: arith.addf
// CHECK: }
// Five shuffles in every lane; a select keeps those that pair one column.
// CHECK: %[[S1:.*]], %{{.*}} = gpu.shuffle xor %[[STRIPE_SUM]], %{{.*}}, %{{.*}} : f32
// CHECK: arith.cmpi uge, %{{.*}}, %[[WIDTH]] : index
// CHECK: %[[C1:.*]] = arith.select
// CHECK: gpu.shuffle xor %[[C1]]
// CHECK: %[[C2:.*]] = arith.select
// CHECK: gpu.shuffle xor %[[C2]]
// CHECK: %[[C3:.*]] = arith.select
// CHECK: gpu.shuffle xor %[[C3]]
// CHECK: %[[C4:.*]] = arith.select
// CHECK: gpu.shuffle xor %[[C4]]
// CHECK: %[[C5:.*]] = arith.select
// The exchange between two barriers, and a pairwise tree over the four
// subgroups of the block.
// CHECK: memref.store %[[C5]], %[[EXCHANGE]][%{{.*}}] : memref<128xf32, #gpu.address_space<workgroup>>
// CHECK-NEXT: gpu.barrier
// CHECK-COUNT-4: memref.load %[[EXCHANGE]]
// CHECK: %[[PAIR_A:.*]] = arith.addf
// CHECK-NEXT: %[[PAIR_B:.*]] = arith.addf
// CHECK-NEXT: %[[TOTAL:.*]] = arith.addf %[[PAIR_A]], %[[PAIR_B]] : f32
// CHECK-NEXT: gpu.barrier
// The store from stripe zero of a column below D of a segment in range.
// CHECK: %[[SLOT_ROW:.*]] = arith.muli %[[SEGMENT]], %[[FEATURES]] : index
// CHECK-NEXT: %[[SLOT:.*]] = arith.addi %[[SLOT_ROW]], %[[OWN_COLUMN]] : index
// CHECK-NEXT: %[[IN_RANGE:.*]] = arith.andi %[[COLUMN_IN_RANGE]], %[[SEGMENT_IN_RANGE]] : i1
// CHECK-NEXT: %[[LEADER:.*]] = arith.cmpi eq, %[[STRIPE]], %{{.*}} : index
// CHECK-NEXT: %[[MAY_STORE:.*]] = arith.andi %[[LEADER]], %[[IN_RANGE]] : i1
// CHECK-NEXT: scf.if %[[MAY_STORE]] {
// CHECK-NEXT: %[[SLOT64:.*]] = arith.index_cast %[[SLOT]] : index to i64
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SLOT64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT: llvm.store %[[TOTAL]], %[[ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// An f64 mean at 512 threads: an exchange of 512 f64 elements, a tree over
// 16 subgroups, and the extent in rows, the clamped end minus the clamped
// start, divided once after the combination.
// MEAN-LABEL: gpu.func @rows_mean(
// MEAN-SAME: workgroup(%[[MEAN_EXCHANGE:[^ ]+]] : memref<512xf64, #gpu.address_space<workgroup>>) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>} {
// MEAN: arith.minsi
// MEAN: arith.minsi
// MEAN-NEXT: %[[MEAN_START:.*]] = arith.index_cast %{{.*}} : i32 to index
// MEAN-NEXT: %[[MEAN_END:.*]] = arith.index_cast %{{.*}} : i32 to index
// MEAN: %[[ROWS:.*]] = arith.subi %[[MEAN_END]], %[[MEAN_START]] : index
// MEAN: scf.for {{.*}} -> (f64) {
// MEAN-COUNT-5: gpu.shuffle xor
// MEAN: gpu.barrier
// MEAN-COUNT-16: memref.load %[[MEAN_EXCHANGE]]
// MEAN: gpu.barrier
// MEAN: %[[COUNT:.*]] = arith.index_cast %[[ROWS]] : index to i32
// MEAN-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f64
// MEAN-NEXT: %[[MEAN:.*]] = arith.divf %{{.*}}, %[[DIVISOR]] : f64
// MEAN: llvm.store %[[MEAN]], %{{.*}} : f64, !llvm.ptr

// The softmax: two reduction stages, each with its own exchange between two
// barriers through the one buffer, so the trailing barrier of the first
// stage protects the buffer from the store of the second. The map store
// writes every element of the stripe at the index it read, under no
// condition.
// SOFTMAX-LABEL: gpu.func @rows_softmax(
// SOFTMAX-SAME: workgroup(%[[SOFTMAX_EXCHANGE:[^ ]+]] : memref<128xf32, #gpu.address_space<workgroup>>)
// SOFTMAX: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {
// SOFTMAX: scf.for {{.*}} -> (f32) {
// SOFTMAX: arith.maximumf
// SOFTMAX-COUNT-5: gpu.shuffle xor
// SOFTMAX: memref.store %{{.*}}, %[[SOFTMAX_EXCHANGE]]
// SOFTMAX-NEXT: gpu.barrier
// SOFTMAX: gpu.barrier
// SOFTMAX: scf.for {{.*}} -> (f32) {
// SOFTMAX: math.exp2
// SOFTMAX-COUNT-5: gpu.shuffle xor
// SOFTMAX: memref.store %{{.*}}, %[[SOFTMAX_EXCHANGE]]
// SOFTMAX-NEXT: gpu.barrier
// SOFTMAX: gpu.barrier
// SOFTMAX-NEXT: scf.for %[[INDEX:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// SOFTMAX-NEXT: %[[INDEX64:.*]] = arith.index_cast %[[INDEX]] : index to i64
// SOFTMAX: arith.divf
// SOFTMAX-NEXT: %[[SLOT:.*]] = llvm.getelementptr %{{.*}}[%[[INDEX64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// SOFTMAX-NEXT: llvm.store %{{.*}}, %[[SLOT]] : f32, !llvm.ptr
// SOFTMAX-NEXT: }
// SOFTMAX-NEXT: }
// SOFTMAX-NEXT: gpu.return

// Without a task buffer, task t is segment t, and the loop runs to
// segment_count * groups.
// DIRECT-LABEL: gpu.func @rows_direct(
// DIRECT-SAME: %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i32, %[[DIRECT_SEGMENT_COUNT:[^:]+]]: i32, %{{[^:]+}}: i32) workgroup(
// DIRECT: %[[DIRECT_TASKS:.*]] = arith.index_cast %[[DIRECT_SEGMENT_COUNT]] : i32 to index
// DIRECT: %[[DIRECT_ITEMS:.*]] = arith.muli %[[DIRECT_TASKS]], %{{.*}} : index
// DIRECT: scf.for %{{.*}} = %{{.*}} to %[[DIRECT_ITEMS]] step %{{.*}} {
// DIRECT-NOT: arith.cmpi ult, %{{.*}}, %[[DIRECT_SEGMENT_COUNT]] : i32

// The partial kernel of the split runs the same tile over the rows of the
// range record of a task, clamped to the row count, and stores each column
// at scratch[task * D + column], a row of scratch per task.
// PARTIAL-LABEL: gpu.func @rows_partial(
// PARTIAL-SAME: %[[P_VALUES:[^:]+]]: !llvm.ptr, %[[P_RANGES:[^:]+]]: !llvm.ptr, %[[P_SCRATCH:[^:]+]]: !llvm.ptr, %[[P_VALUE_COUNT:[^:]+]]: i32, %[[P_PARTIAL_COUNT:[^:]+]]: i32, %[[P_FEATURE_COUNT:[^:]+]]: i32) workgroup(%[[P_EXCHANGE:[^ ]+]] : memref<512xf32, #gpu.address_space<workgroup>>) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>} {
// PARTIAL: %[[P_FEATURES:.*]] = arith.index_cast %[[P_FEATURE_COUNT]] : i32 to index
// PARTIAL: %[[P_TASKS:.*]] = arith.index_cast %[[P_PARTIAL_COUNT]] : i32 to index
// PARTIAL: %[[P_ITEMS:.*]] = arith.muli %[[P_TASKS]], %[[P_GROUPS:.*]] : index
// PARTIAL: scf.for %[[P_ITEM:.*]] = %{{.*}} to %[[P_ITEMS]] step %{{.*}} {
// PARTIAL-NEXT: %[[P_TASK:.*]] = arith.divui %[[P_ITEM]], %[[P_GROUPS]] : index
// PARTIAL: llvm.getelementptr %[[P_RANGES]][
// PARTIAL: llvm.getelementptr %[[P_RANGES]][
// PARTIAL: arith.minsi %{{.*}}, %[[P_VALUE_COUNT]] : i32
// PARTIAL: arith.minsi %{{.*}}, %[[P_VALUE_COUNT]] : i32
// PARTIAL: %[[P_COLUMN:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// PARTIAL-NEXT: %[[P_COLUMN_IN_RANGE:.*]] = arith.cmpi ult, %[[P_COLUMN]], %[[P_FEATURES]] : index
// PARTIAL: scf.for {{.*}} -> (f32) {
// PARTIAL: llvm.getelementptr %[[P_VALUES]][
// PARTIAL-COUNT-5: gpu.shuffle xor
// PARTIAL: memref.store %{{.*}}, %[[P_EXCHANGE]]
// PARTIAL-NEXT: gpu.barrier
// PARTIAL-COUNT-16: memref.load %[[P_EXCHANGE]]
// PARTIAL: gpu.barrier
// PARTIAL-NEXT: %[[P_ROW:.*]] = arith.muli %[[P_TASK]], %[[P_FEATURES]] : index
// PARTIAL-NEXT: %[[P_SLOT:.*]] = arith.addi %[[P_ROW]], %[[P_COLUMN]] : index
// PARTIAL-NEXT: %[[P_LEADER:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}} : index
// PARTIAL-NEXT: %[[P_MAY_STORE:.*]] = arith.andi %[[P_LEADER]], %[[P_COLUMN_IN_RANGE]] : i1
// PARTIAL-NEXT: scf.if %[[P_MAY_STORE]] {
// PARTIAL-NEXT: %[[P_SLOT64:.*]] = arith.index_cast %[[P_SLOT]] : index to i64
// PARTIAL-NEXT: llvm.getelementptr %[[P_SCRATCH]][%[[P_SLOT64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32

// The merge kernel runs the tile over the rows of scratch its merge record
// names, clamped to the partial count. The extent of a mean is read from
// the range records, a number of rows, and the result of each column is
// stored at output[segment * D + column] when the segment of the record is
// below the segment count.
// MERGE-LABEL: gpu.func @rows_merge_mean(
// MERGE-SAME: %[[M_SCRATCH:[^:]+]]: !llvm.ptr, %[[M_OUTPUT:[^:]+]]: !llvm.ptr, %[[M_MERGES:[^:]+]]: !llvm.ptr, %[[M_RANGES:[^:]+]]: !llvm.ptr, %[[M_PARTIAL_COUNT:[^:]+]]: i32, %[[M_MERGE_COUNT:[^:]+]]: i32, %[[M_SEGMENT_COUNT:[^:]+]]: i32, %[[M_FEATURE_COUNT:[^:]+]]: i32) workgroup(%[[M_EXCHANGE:[^ ]+]] : memref<512xf32, #gpu.address_space<workgroup>>)
// MERGE: %[[M_FEATURES:.*]] = arith.index_cast %[[M_FEATURE_COUNT]] : i32 to index
// MERGE: %[[M_TASKS:.*]] = arith.index_cast %[[M_MERGE_COUNT]] : i32 to index
// MERGE: %[[M_ITEMS:.*]] = arith.muli %[[M_TASKS]], %{{.*}} : index
// MERGE: scf.for %{{.*}} = %{{.*}} to %[[M_ITEMS]] step %{{.*}} {
// MERGE: llvm.getelementptr %[[M_MERGES]][
// MERGE-NEXT: %[[M_SEGMENT_WORD:.*]] = llvm.load
// MERGE-NEXT: %[[M_SEGMENT_IN_RANGE:.*]] = arith.cmpi ult, %[[M_SEGMENT_WORD]], %[[M_SEGMENT_COUNT]] : i32
// MERGE-NEXT: %[[M_SEGMENT:.*]] = arith.index_cast %[[M_SEGMENT_WORD]] : i32 to index
// MERGE: arith.minsi %{{.*}}, %[[M_PARTIAL_COUNT]] : i32
// MERGE: arith.minsi %{{.*}}, %[[M_PARTIAL_COUNT]] : i32
// MERGE: %[[M_ROWS:.*]] = scf.if %{{.*}} -> (index) {
// MERGE: llvm.getelementptr %[[M_RANGES]][
// MERGE: llvm.getelementptr %[[M_RANGES]][
// MERGE: %[[M_COLUMN:.*]] = arith.addi %{{.*}}, %{{.*}} : index
// MERGE-NEXT: %[[M_COLUMN_IN_RANGE:.*]] = arith.cmpi ult, %[[M_COLUMN]], %[[M_FEATURES]] : index
// MERGE: scf.for {{.*}} -> (f32) {
// MERGE: llvm.getelementptr %[[M_SCRATCH]][
// MERGE-COUNT-5: gpu.shuffle xor
// MERGE: memref.store %{{.*}}, %[[M_EXCHANGE]]
// MERGE-NEXT: gpu.barrier
// MERGE: gpu.barrier
// MERGE-NEXT: %[[M_COUNT:.*]] = arith.index_cast %[[M_ROWS]] : index to i32
// MERGE-NEXT: %[[M_DIVISOR:.*]] = arith.sitofp %[[M_COUNT]] : i32 to f32
// MERGE-NEXT: %[[M_MEAN:.*]] = arith.divf %{{.*}}, %[[M_DIVISOR]] : f32
// MERGE-NEXT: %[[M_ROW:.*]] = arith.muli %[[M_SEGMENT]], %[[M_FEATURES]] : index
// MERGE-NEXT: %[[M_SLOT:.*]] = arith.addi %[[M_ROW]], %[[M_COLUMN]] : index
// MERGE-NEXT: %[[M_IN_RANGE:.*]] = arith.andi %[[M_COLUMN_IN_RANGE]], %[[M_SEGMENT_IN_RANGE]] : i1
// MERGE: scf.if
// MERGE: llvm.getelementptr %[[M_OUTPUT]][
// MERGE-NEXT: llvm.store %[[M_MEAN]]

module {
  func.func @rows_sum(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  func.func @rows_mean(
      %values: memref<?x?xf64>, %offsets: memref<?xi32>,
      %output: memref<?x?xf64>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?x?xf64>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?x?xf64>) {
    ^bb0(%column: !swage.segment<f64>, %rows: index):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f64> -> f64 {
      ^bb0(%value: f64):
        swage.yield %value : f64
      }
      %count = arith.index_cast %rows : index to i32
      %divisor = arith.sitofp %count : i32 to f64
      %mean = arith.divf %sum, %divisor : f64
      swage_plan.yield %mean : f64
    }
    return
  }

  func.func @rows_softmax(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32) {
    ^bb0(%column: !swage.segment<f32>):
      %max = swage.reduce %column kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      %total = swage.reduce %column captures(%max : f32) kind<sum>
          : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32, %m: f32):
        %log2e = arith.constant 1.44269502 : f32
        %centered = arith.subf %value, %m : f32
        %scaled = arith.mulf %centered, %log2e : f32
        %exponential = math.exp2 %scaled : f32
        swage.yield %exponential : f32
      }
      swage.map_store %column, %output captures(%max, %total : f32, f32)
          : !swage.segment<f32>, memref<?x?xf32> {
      ^bb0(%value: f32, %m: f32, %t: f32):
        %log2e = arith.constant 1.44269502 : f32
        %centered = arith.subf %value, %m : f32
        %scaled = arith.mulf %centered, %log2e : f32
        %exponential = math.exp2 %scaled : f32
        %normalized = arith.divf %exponential, %t : f32
        swage.yield %normalized : f32
      }
      swage_plan.yield
    }
    return
  }

  func.func @rows_direct(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32)
      attributes {swage_plan.block_threads = 64 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<min> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  func.func @rows_partial(
      %values: memref<?x?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?x?xf32>, %value_count: i32, %partial_count: i32,
      %feature_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.partial_tasks values(%values : memref<?x?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32)
        feature_count(%feature_count : i32)
        into(%scratch : memref<?x?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
  func.func @rows_merge_mean(
      %scratch: memref<?x?xf32>, %output: memref<?x?xf32>,
      %merges: memref<?xi32>, %ranges: memref<?xi32>, %partial_count: i32,
      %merge_count: i32, %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.merge_tasks scratch(%scratch : memref<?x?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32) ranges(%ranges : memref<?xi32>)
        into(%output : memref<?x?xf32>) {
    ^bb0(%partials: !swage.segment<f32>, %rows: index):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      %count = arith.index_cast %rows : index to i32
      %divisor = arith.sitofp %count : i32 to f32
      %mean = arith.divf %total, %divisor : f32
      swage_plan.yield %mean : f32
    }
    return
  }
}
