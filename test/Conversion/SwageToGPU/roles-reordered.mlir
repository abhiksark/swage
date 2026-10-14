// test/Conversion/SwageToGPU/roles-reordered.mlir
// The kernel takes its arguments in the order of its layout, whatever order
// the segment function declares them in: the roles name the arguments. This
// function is the one in segmented-sum.mlir with its arguments reversed, and
// it lowers to the same kernel. Only the launch contract differs: it binds
// each kernel argument to the position the segment function declares it at,
// so the diff below leaves out the line of the kernel, which carries the
// contract, and the checks pin the contract of each order.
//
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu \
// RUN:   %S/segmented-sum.mlir > %t.ordered
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | diff -I '.*swage.kernel_contract' %t.ordered -
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s
// RUN: FileCheck %s --check-prefix=ORDERED < %t.ordered
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' %s \
// RUN:   | FileCheck %s --check-prefix=PLAN

// CHECK: gpu.func @segmented_sum(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: i32)
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 4 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 3 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 1 : i64}, {kind = "i32", origin = "user", source_index = 2 : i64}, {kind = "i32", origin = "user", source_index = 0 : i64}], backend = "cuda", entry = "segmented_sum", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// CHECK-NOT: swage.role

// The plan function takes the layout order and records where the segment
// function declared each argument.
// PLAN: func.func @segmented_sum(%{{.*}}: memref<?xf32> {swage.role = #swage.role<values>, swage_plan.source_index = 4 : i32}, %{{.*}}: memref<?xi32> {swage.role = #swage.role<offsets>, swage_plan.source_index = 3 : i32}, %{{.*}}: memref<?xf32> {swage.role = #swage.role<output>, swage_plan.source_index = 1 : i32}, %{{.*}}: i32 {swage.role = #swage.role<value_count>, swage_plan.source_index = 2 : i32}, %{{.*}}: i32 {swage.role = #swage.role<segment_count>, swage_plan.source_index = 0 : i32})

// ORDERED: gpu.func @segmented_sum(
// ORDERED-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {kind = "i32", origin = "user", source_index = 3 : i64}, {kind = "i32", origin = "user", source_index = 4 : i64}], backend = "cuda", entry = "segmented_sum", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}

module {
  func.func @segmented_sum(
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %values: memref<?xf32> {swage.role = #swage.role<values>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
