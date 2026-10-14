// test/Conversion/SwagePlanToGPU/schedule-list.mlir
// A schedule list plans three kernels of one function, and the conversion
// gives one gpu.module per kernel, in the order of the list.
//
// RUN: swage-opt \
// RUN:   --swage-to-plan='schedule=task-ids,split-partial,split-merge block-threads=32' \
// RUN:   --swage-plan-to-gpu %S/../SwageToGPU/segmented-sum.mlir \
// RUN:   | FileCheck %s --implicit-check-not=func.func

// CHECK: gpu.module @segmented_sum_module {
// CHECK-NEXT: gpu.func @segmented_sum({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 32, 1, 1>,
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "task_ids", kind = "ptr", origin = "plan"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "task_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}], backend = "cuda", entry = "segmented_sum", launch = {block = array<i32: 32, 1, 1>, model = "spmd-grid"}, version = 2 : i64}} {
// CHECK: gpu.shuffle xor
// CHECK: gpu.module @segmented_sum__partial_module {
// CHECK-NEXT: gpu.func @segmented_sum__partial({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>,
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", key = "partial_ranges", kind = "ptr", origin = "plan"}, {access = "write", key = "scratch", kind = "ptr", origin = "scratch"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "partial_count", kind = "i32", origin = "derived"}], backend = "cuda", entry = "segmented_sum__partial", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}} {
// CHECK: gpu.all_reduce add
// CHECK: gpu.module @segmented_sum__merge_module {
// CHECK-NEXT: gpu.func @segmented_sum__merge({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>,
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", key = "scratch", kind = "ptr", origin = "scratch"}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "merge_records", kind = "ptr", origin = "plan"}, {key = "partial_count", kind = "i32", origin = "derived"}, {key = "merge_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}], backend = "cuda", entry = "segmented_sum__merge", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}} {
// CHECK: gpu.all_reduce add
