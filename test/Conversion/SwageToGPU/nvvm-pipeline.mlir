// test/Conversion/SwageToGPU/nvvm-pipeline.mlir
// RUN: swage-opt --split-input-file %s \
// RUN:   --pass-pipeline='builtin.module(swage-to-plan{schedule=direct block-threads=128},swage-plan-to-gpu,gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CTA \
// RUN:       --implicit-check-not=swage. --implicit-check-not=scf. \
// RUN:       --implicit-check-not=math. --implicit-check-not=gpu.func \
// RUN:       --implicit-check-not=gpu.all_reduce \
// RUN:       --implicit-check-not=gpu.shuffle --implicit-check-not=gpu.barrier
// RUN: swage-opt --split-input-file %s \
// RUN:   --pass-pipeline='builtin.module(swage-to-plan{schedule=task-ids block-threads=32},swage-plan-to-gpu,gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WARP \
// RUN:       --implicit-check-not=swage. --implicit-check-not=scf. \
// RUN:       --implicit-check-not=math. --implicit-check-not=gpu.func \
// RUN:       --implicit-check-not=gpu.all_reduce \
// RUN:       --implicit-check-not=gpu.shuffle --implicit-check-not=gpu.barrier
// RUN: swage-opt --split-input-file \
// RUN:     --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | swage-opt --split-input-file \
// RUN:     --pass-pipeline='builtin.module(gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))' \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CTA

// The code generation C API lowers a kernel with this pipeline: the planner
// and the plan conversion, then the two upstream conversions nested on the
// GPU module. The driver runs the same pipeline from text, in one invocation
// or with the lowered module printed and parsed in between. The upstream
// conversions find their patterns through dialect extensions, so this test
// also holds the driver to registering them.

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
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// The kernel keeps its name, its launch width, and its parameter order: the
// buffers as pointers, then the i32 counts. The upstream conversion copies
// the discardable launch contract onto the LLVM function; code generation
// validates and removes it before this conversion, so it never reaches PTX.
// CHECK-LABEL: gpu.module @segmented_sum_module
// CHECK: llvm.func @segmented_sum(
// CTA-SAME: %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
// CTA-SAME: attributes {gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 128, 1, 1>, swage.kernel_contract = {{.+}}}
// WARP-SAME: %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
// WARP-SAME: attributes {gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 32, 1, 1>, swage.kernel_contract = {{.+}}}
// CHECK: nvvm.read.ptx.sreg.ctaid.x
// CHECK: nvvm.read.ptx.sreg.tid.x

// A block of four warps reduces through the upstream all-reduce lowering,
// which shuffles inside each subgroup and synchronizes the block.
// CTA: nvvm.shfl.sync {{ *}}bfly
// CTA: nvvm.barrier0

// One warp reduces with the butterfly the plan conversion emits: five steps
// for 32 lanes, and no block barrier.
// WARP-COUNT-5: nvvm.shfl.sync {{ *}}bfly
// WARP-NOT: nvvm.shfl.sync
// WARP-NOT: nvvm.barrier0
// CHECK: llvm.return

// -----

module {
  func.func @exponential_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %exponential = math.exp2 %value : f32
      swage.yield %exponential : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// The NVVM conversion turns math.exp2 into a call of a libdevice function.
// The C API replaces that call with an LLVM intrinsic in a step that is not
// a registered pass, so the text below is where a pipeline in this driver
// ends, one step short of what the C API compiles to PTX.
// CHECK-LABEL: gpu.module @exponential_sum_module
// CHECK: llvm.func @__nv_exp2f(f32) -> f32
// CHECK: llvm.func @exponential_sum(
// CHECK-SAME: swage.kernel_contract = {{.+}}
// CHECK: llvm.call @__nv_exp2f(
// CHECK: llvm.return

// -----

module {
  func.func @segmented_max_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %max = swage.reduce %segment kind<max>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %max, %output[%sid] : memref<?xf64>
    return
  }
}

// A warp shuffle moves 32 bits, so the NVVM conversion sends an f64 value as
// its two i32 halves and joins them again. One warp therefore reduces with
// ten shuffles for its five steps, and the block reduction doubles in the
// same way. The combine stays an f64 maximum.
// CHECK-LABEL: gpu.module @segmented_max_f64_module
// CHECK: llvm.func @segmented_max_f64(
// CTA-SAME: attributes {gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 128, 1, 1>, swage.kernel_contract = {{.+}}}
// WARP-SAME: attributes {gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 32, 1, 1>, swage.kernel_contract = {{.+}}}
// CHECK: llvm.intr.maximum(%{{.*}}) : (f64, f64) -> f64
// CHECK: llvm.bitcast %{{.*}} : f64 to i64
// CHECK: nvvm.shfl.sync {{ *}}bfly %{{.*}} : i32 -> i32
// CHECK: nvvm.shfl.sync {{ *}}bfly %{{.*}} : i32 -> i32
// CHECK: llvm.bitcast %{{.*}} : i64 to f64
// CHECK-NEXT: llvm.intr.maximum(%{{.*}}) : (f64, f64) -> f64
// CTA: nvvm.barrier0
// WARP-COUNT-8: nvvm.shfl.sync {{ *}}bfly %{{.*}} : i32 -> i32
// WARP-NOT: nvvm.shfl.sync
// WARP-NOT: nvvm.barrier0
// CHECK: llvm.return
