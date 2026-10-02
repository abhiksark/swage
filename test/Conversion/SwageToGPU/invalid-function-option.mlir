// test/Conversion/SwageToGPU/invalid-function-option.mlir
// The function option names a segment function of the module. Every
// segmented pass reports a name that does not.
//
// RUN: not swage-opt --swage-to-plan='schedule=sequential function=absent' \
// RUN:   --swage-plan-to-scf \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: not swage-opt \
// RUN:   --swage-to-plan='schedule=direct block-threads=128 function=absent' \
// RUN:   --swage-plan-to-gpu \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: not swage-opt \
// RUN:   --swage-to-plan='schedule=split-partial function=absent' \
// RUN:   --swage-plan-to-gpu \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: not swage-opt --swage-to-plan='function=absent' \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=ABSENT
// RUN: not swage-opt --swage-to-plan='schedule=sequential function=bystander' \
// RUN:   --swage-plan-to-scf \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=BYSTANDER
// RUN: not swage-opt \
// RUN:   --swage-to-plan='schedule=direct block-threads=128 function=bystander' \
// RUN:   --swage-plan-to-gpu \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=BYSTANDER
// RUN: not swage-opt \
// RUN:   --swage-to-plan='schedule=split-partial function=bystander' \
// RUN:   --swage-plan-to-gpu \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=BYSTANDER
// RUN: not swage-opt --swage-to-plan='function=bystander' \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=BYSTANDER

// ABSENT: error: function names @absent, which is not a function of the module
// BYSTANDER: error: function names @bystander, which holds no Swage segment operation

module {
  func.func @segmented_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }
  func.func @bystander(%x: i32) -> i32 {
    return %x : i32
  }
}
