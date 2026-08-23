// test/Dialect/SwagePlan/invalid.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)
  func.func @bad_policy(%offsets: memref<?xi32>, %value_count: i32,
                        %segment_count: i32) -> !swage_plan.task_range {
    // expected-error@+2 {{expected ::mlir::swage_plan::TaskPolicy to be one of: warp, cta}}
    // expected-error@+1 {{failed to parse SwagePlan_TaskPolicyAttr parameter}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @semantic_sum, policies = [#swage_plan.policy<packed_warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)
  func.func @negative_threshold(%offsets: memref<?xi32>, %value_count: i32,
                                %segment_count: i32) -> !swage_plan.task_range {
    // expected-error@+1 {{attribute 'warp_max_elements' failed to satisfy constraint}}
    %tasks = "swage_plan.classify"(%offsets, %value_count, %segment_count) {kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = -1 : i32} : (memref<?xi32>, i32, i32) -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)
  func.func @wrong_policy_order(%offsets: memref<?xi32>, %value_count: i32,
                                %segment_count: i32) -> !swage_plan.task_range {
    // expected-error@+1 {{policies must be ordered warp then CTA}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @semantic_sum, policies = [#swage_plan.policy<cta>, #swage_plan.policy<warp>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)
  func.func @wrong_offsets(%offsets: memref<?xf32>, %value_count: i32,
                           %segment_count: i32) -> !swage_plan.task_range {
    // expected-error@+1 {{operand #0 must be 1D memref of 32-bit signless integer values}}
    %tasks = "swage_plan.classify"(%offsets, %value_count, %segment_count) {kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : (memref<?xf32>, i32, i32) -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)
  func.func @wrong_count(%offsets: memref<?xi32>, %value_count: i64,
                         %segment_count: i32) -> !swage_plan.task_range {
    // expected-error@+1 {{operand #1 must be 32-bit signless integer}}
    %tasks = "swage_plan.classify"(%offsets, %value_count, %segment_count) {kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : (memref<?xi32>, i64, i32) -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)
  func.func @wrong_result(%offsets: memref<?xi32>, %value_count: i32,
                          %segment_count: i32) -> i32 {
    // expected-error@+1 {{result #0 must be SwagePlan task range}}
    %tasks = "swage_plan.classify"(%offsets, %value_count, %segment_count) {kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : (memref<?xi32>, i32, i32) -> i32
    return %tasks : i32
  }
}

// -----

module {
  func.func @missing_kernel(%offsets: memref<?xi32>, %value_count: i32,
                            %segment_count: i32) -> !swage_plan.task_range {
    // expected-error@+1 {{kernel must reference a func.func semantic kernel}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @absent, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @wrong_signature(memref<?xi32>)
  func.func @wrong_kernel_signature(
      %offsets: memref<?xi32>, %value_count: i32, %segment_count: i32)
      -> !swage_plan.task_range {
    // expected-error@+1 {{kernel must use the canonical five-argument semantic ABI}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @wrong_signature, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  memref.global "private" @not_a_function : memref<1xi32>
  func.func @non_function_kernel(
      %offsets: memref<?xi32>, %value_count: i32, %segment_count: i32)
      -> !swage_plan.task_range {
    // expected-error@+1 {{kernel must reference a func.func semantic kernel}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @not_a_function, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func @self_referencing_kernel(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{kernel must not reference its containing function}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @self_referencing_kernel, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return
  }
}

// -----

module {
  func.func private @signed_value_count_kernel(
      memref<?xf32>, memref<?xi32>, memref<?xf32>, si32, i32)
  func.func @signed_value_count(
      %offsets: memref<?xi32>, %value_count: i32, %segment_count: i32)
      -> !swage_plan.task_range {
    // expected-error@+1 {{kernel must use the canonical five-argument semantic ABI}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @signed_value_count_kernel, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @unsigned_segment_count_kernel(
      memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, ui32)
  func.func @unsigned_segment_count(
      %offsets: memref<?xi32>, %value_count: i32, %segment_count: i32)
      -> !swage_plan.task_range {
    // expected-error@+1 {{kernel must use the canonical five-argument semantic ABI}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @unsigned_segment_count_kernel, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}

// -----

module {
  func.func private @non_default_memory_kernel(
      memref<?xf32, #gpu.address_space<workgroup>>, memref<?xi32>,
      memref<?xf32>, i32, i32)
  func.func @non_default_kernel_memory_space(
      %offsets: memref<?xi32>, %value_count: i32, %segment_count: i32)
      -> !swage_plan.task_range {
    // expected-error@+1 {{kernel must use the canonical five-argument semantic ABI}}
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {kernel = @non_default_memory_kernel, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}
