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
