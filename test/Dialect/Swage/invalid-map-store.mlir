// RUN: swage-opt %s --split-input-file --verify-diagnostics

func.func @output_element_mismatch(%s: !swage.segment<f32>, %out: memref<?xi32>) {
  // expected-error @below {{yield type 'f32' does not match the expected element type 'i32'}}
  swage.map_store %s, %out : !swage.segment<f32>, memref<?xi32> {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  return
}
