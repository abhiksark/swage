// RUN: swage-opt %s --split-input-file --verify-diagnostics

func.func @yield_type_mismatch(%s: !swage.segment<f32>) -> !swage.segment<f32> {
  // expected-error @below {{yield type 'f16' does not match the expected element type 'f32'}}
  %r = swage.map %s : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    %t = arith.truncf %x : f32 to f16
    swage.yield %t : f16
  }
  return %r : !swage.segment<f32>
}

// -----

func.func @capture_arity_mismatch(%s: !swage.segment<f32>, %c: f32) -> !swage.segment<f32> {
  // expected-error @below {{region expects 2 arguments (element plus captures), got 1}}
  %r = swage.map %s captures(%c : f32) : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  return %r : !swage.segment<f32>
}

// -----

func.func @capture_type_mismatch(%s: !swage.segment<f32>, %c: i32) -> !swage.segment<f32> {
  // expected-error @below {{region argument #1 type 'f32' does not match capture type 'i32'}}
  %r = swage.map %s captures(%c : i32) : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32, %m: f32):
    swage.yield %x : f32
  }
  return %r : !swage.segment<f32>
}

// -----

func.func @element_type_mismatch(%s: !swage.segment<i32>) -> !swage.segment<i32> {
  // expected-error @below {{region argument #0 type 'f32' does not match segment element type 'i32'}}
  %r = swage.map %s : !swage.segment<i32> -> !swage.segment<i32> {
  ^bb0(%x: f32):
    %y = arith.fptosi %x : f32 to i32
    swage.yield %y : i32
  }
  return %r : !swage.segment<i32>
}

// -----

func.func @isolation_violation(%s: !swage.segment<f32>, %outer: f32) -> !swage.segment<f32> {
  // expected-note @below {{required by region isolation constraints}}
  %r = swage.map %s : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    // expected-error @below {{using value defined outside the region}}
    %y = arith.addf %x, %outer : f32
    swage.yield %y : f32
  }
  return %r : !swage.segment<f32>
}
