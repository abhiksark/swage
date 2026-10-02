// test/Dialect/SwagePlan/invalid.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

// The launch width attribute makes a function a plan function, and the
// dialect verifies the shape of a plan function with it.

// expected-error@+1 {{swage_plan.block_threads belongs on a func.func, got 'builtin.module'}}
module attributes {swage_plan.block_threads = 128 : i32} {
}

// -----

module {
  // expected-error@+1 {{'swage_plan.launch_width' is not an operation attribute of the swage_plan dialect; the dialect defines swage_plan.block_threads}}
  func.func private @unknown_attribute() attributes {swage_plan.launch_width = 128 : i32}
}

// -----

module {
  // expected-error@+1 {{swage_plan.block_threads must be a positive i32, got 0 : i32}}
  func.func @zero_threads(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 0 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  // expected-error@+1 {{swage_plan.block_threads must be a positive i32, got 128 : i64}}
  func.func @wide_threads(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i64} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  // expected-error@+1 {{a plan function has no result, got '(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> i32'}}
  func.func @returns_a_value(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) -> i32
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return %value_count : i32
  }
}

// -----

// A kernel takes a buffer as a pointer, so the conversion needs a buffer it
// can address that way.
module {
  // expected-error@+1 {{plan function argument #2 must be a signless integer or a memref of rank one or two of signless integers or floats with dynamic sizes, the identity layout, and the default memory space, got 'memref<8xf32>'}}
  func.func @fixed_size_output(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<8xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<8xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  // expected-error@+1 {{a plan function has a body of one block}}
  func.func private @declaration(memref<?xf32>, i32)
      attributes {swage_plan.block_threads = 128 : i32}
}

// -----

module {
  // expected-error@+1 {{a plan function holds one task operation followed by a return, found 1 operations}}
  func.func @no_task_operation(%value_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    return
  }
}

// -----

// The task operation ties the counts and the task buffer to the offsets, so
// a pattern takes one word type from any of them.
module {
  func.func @wide_value_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i64, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op value_count must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i64) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @narrow_segment_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i16) {
    // expected-error@+1 {{'swage_plan.tasks' op segment_count must have the element type of the offsets, 'i32', got 'i16'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i16)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @ids_without_task_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op ids and task_count are given together: the ids name the segment of each task, and task_count bounds the task index}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @task_count_without_ids(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op ids and task_count are given together: the ids name the segment of each task, and task_count bounds the task index}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        task_count(%task_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @wide_ids(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi64>, %value_count: i32,
      %task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op an element of ids must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi64>) task_count(%task_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @wide_task_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i64, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op task_count must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i64)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The region runs on the bound segment, which is its one argument.
module {
  func.func @region_without_a_segment(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op region takes the bound segment as its one argument, of type !swage.segment<T>}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%value: f32):
      swage_plan.yield
    }
    return
  }
}

// -----

module {
  func.func @segment_of_another_type(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op region binds a segment of 'f32', the element type of the values, got '!swage.segment<i32>'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%segment: !swage.segment<i32>):
      swage_plan.yield
    }
    return
  }
}

// -----

module {
  func.func @other_terminator(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op region must end in swage_plan.yield}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%segment: !swage.segment<f32>):
      llvm.unreachable
    }
    return
  }
}

// -----

// A task region holds the consumers of the bound segment and nothing else.
module {
  func.func @other_operation(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%segment: !swage.segment<f32>):
      // expected-error@+1 {{'swage.extent' op is not allowed in the region of 'swage_plan.tasks'; the region holds swage.reduce and swage.map_store operations and ends in swage_plan.yield}}
      %length = swage.extent %segment : !swage.segment<f32>
      swage_plan.yield
    }
    return
  }
}

// -----

module {
  func.func @consumer_of_another_segment(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32,
      %sid: index) {
    %other = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      // expected-error@+1 {{'swage.reduce' op must read the bound segment, the argument of the task region}}
      %sum = swage.reduce %other kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The into buffer receives the yielded scalar, so the two come together.
module {
  func.func @into_without_a_scalar(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op into and a yielded scalar are given together: the scalar of each segment is stored in the into buffer}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      swage_plan.yield
    }
    return
  }
}

// -----

module {
  func.func @scalar_without_into(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op into and a yielded scalar are given together: the scalar of each segment is stored in the into buffer}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @scalar_of_another_type(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf64>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op region must yield 'f64', the element type of the into buffer, got 'f32'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf64>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The oracle visits every segment in order on one thread: it has no task
// buffer and no launch width.
module {
  func.func @sequential_with_ids(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op policy<sequential> visits every segment in order and takes no ids}}
    swage_plan.tasks policy<sequential>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  // expected-error@+1 {{swage_plan.block_threads gives the launch width of a kernel, and policy<sequential> runs on one thread without a kernel; a function has one or the other}}
  func.func @sequential_with_a_launch_width(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<sequential>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @yield_outside_a_task_region() {
    // expected-error@+1 {{'swage_plan.yield' op expects parent op to be one of 'swage_plan.tasks, swage_plan.partial_tasks, swage_plan.merge_tasks, swage_plan.fused_tasks, swage_plan.persistent_tasks'}}
    swage_plan.yield
  }
}
