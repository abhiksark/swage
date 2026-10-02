// test/Conversion/SwageToGPU/no-segment-function.mlir
// A module without a segment function has nothing to lower, and every
// segmented pass leaves it as it is.
//
// RUN: swage-opt --swage-segmented-reduction-to-scf %s | FileCheck %s
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu %s | FileCheck %s
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
