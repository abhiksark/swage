// RUN: swage-opt %s --split-input-file --verify-diagnostics

func.func @bad_kind(%s: !swage.segment<f32>) -> f32 {
  // expected-error @below {{expected ::mlir::swage::ReductionKind to be one of: sum, max, min}}
  // expected-error @below {{failed to parse Swage_ReductionKindAttr parameter 'value'}}
  %r = swage.reduce %s kind<mul> : !swage.segment<f32> -> f32 {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  return %r : f32
}

// -----

func.func @result_type_mismatch(%s: !swage.segment<f32>) -> i32 {
  // expected-error @below {{yield type 'f32' does not match the expected element type 'i32'}}
  %r = swage.reduce %s kind<sum> : !swage.segment<f32> -> i32 {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  return %r : i32
}
