// test/Conversion/SwageToGPU/no-segment-function.mlir
// A module without a segment function has nothing to lower, and every
// segmented pass leaves it as it is.
//
// RUN: swage-opt --swage-to-plan='schedule=sequential' \
// RUN:   --swage-plan-to-scf %s | FileCheck %s
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s
// RUN: swage-opt --swage-to-plan='schedule=split-partial' \
// RUN:   --swage-plan-to-gpu %s | FileCheck %s
// RUN: swage-opt --swage-to-plan %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT: func.func @bystander(%[[X:.*]]: i32) -> i32 {
// CHECK-NEXT: return %[[X]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: }

module {
  func.func @bystander(%x: i32) -> i32 {
    return %x : i32
  }
}
