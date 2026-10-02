<!-- docs/adr/ADR-0020-planned-per-function-lowering.md -->
# ADR-0020: Segmented GPU lowering as a planned per-function conversion

- Status: accepted; all ten steps of the migration sequence are implemented
- Date: 2026-10-02
- Accepted: 2026-10-02, with the recommended answer to every question at the
  end

The design was built one migration step at a time, and every step is in
the tree: step 0 (the digest gate, dialect extensions in `swage-opt`, and a
lit test of the nested NVVM pipeline), step 1 (the target description),
step 2 (argument roles, the kernel layouts, admission per function, any
number of segment functions in a module, and the symbol checks before
mutation), step 3 (map fusion), step 4 (the plan stage and the conversion
for the direct and task-id schedules, softmax included), step 5 (the CPU
oracle on the shared patterns), step 6 (the split partial stage, schedule
lists, and the record layouts), step 7 (the fused mixed kernel), step 8
(the split merge stage), step 9 (the persistent queue kernel), and step 10
(the removal of the three legacy passes).

What was built:

- A segment function declares its arguments with `swage.role`, and a
  module may hold any number of segment functions.
- `--swage-to-plan` admits each selected function, fuses its maps, and
  replaces it by one plan function per schedule of its list: `direct`,
  `task-ids`, `fused-mixed`, `split-partial`, `split-merge`, `persistent`,
  or `sequential` alone.
- A plan function has the parameter list of its kernel, its launch width
  as an attribute, and one task operation that takes every buffer and
  every device bound as an operand. The dialect has five task operations
  and `swage_plan.yield`.
- `--swage-plan-to-gpu` is a dialect conversion with one pattern per
  operation, and `--swage-plan-to-scf` converts the sequential oracle with
  the same consumer patterns.
- One target description is read by the planner, the conversion, the C
  API, and the host.
- The code generation C API keeps its six compile entry points and runs
  the planner and the conversion as two passes.

What differs from the proposal is listed in "Migration sequence", once for
steps 1 to 4 and once for steps 5 to 10. No difference changed a kernel:
the digests of the lowered MLIR and of the PTX did not move in any step.

Byte-identical emission is the acceptance criterion of every step through
step 9: the committed digests of the lowered MLIR and of the PTX must not
move. The conversion mechanics were checked against the headers and sources
of the pinned LLVM release when this record was written. Step 4 proved them
on the direct and task-id kernels, with the two adjustments that "The
conversion" names.

## Context

This section describes the tree before the migration started. Line numbers
refer to revision `406b725`. A bare line number refers to
`lib/Conversion/SegmentedReduction/SegmentedReduction.cpp`; every other file
is named by its path. Python code is cited by function name.

### How functions are found

- `findSegmentedReduction` (182-201) collects every `func.func` that
  contains an operation of the Swage dialect and requires exactly one. The
  SCF pass (1470), the GPU pass (1568), and the split pass (1634) all start
  there.
- `SwageToPlanPass` (1691-1697) is stricter: the module must hold exactly one
  `func.func`, bystanders included.
- `lib/CAPI/Codegen.cpp` checks that `kernelName` names some function (184),
  then requires exactly one `gpu.module` after lowering (251-257).
  `verifyCompiledKernel` (270-282) compares the last non-external `llvm.func`
  in that module with `kernelName` plus `__partial` or `__merge`.
- Every emitter ends in `source.erase()` (1233, 1292, 1314, 1449) with no
  symbol-use check. Each creates `<name>_module` with no symbol lookup
  (672-673, 1329-1330). A caller of the kernel, or a symbol clash, is
  therefore reported by the verifier that runs after the pass, when the
  module has already been changed.

### How the ABI is fixed

Source side:

- `verifySegmentedFunctionShape` (216-235) requires exactly
  `(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()`.
- The same rule is repeated in `hasCanonicalSemanticABI`
  (`lib/Dialect/SwagePlan/IR/SwagePlanDialect.cpp:30-41`) for the verifier
  of `swage_plan.classify`.
- `swage.make_segment` must bind `getArgument(0)` and `getArgument(1)`
  (279-283). The terminal must bind `getArgument(2)` (376, 381-382).
- The SCF lowering reads arguments 0, 1, 2, and 4 (560-603).

Kernel side:

- `buildGPUProgram` assembles `inputs` from three booleans (682-697).
- It then reads arguments by index: `valueCountIndex` is 3, 4, or 10
  (723-728), and the segment count is `inputs.size() - 1` (733).
- The persistent branch reads `getArgument(3)` through `getArgument(14)`
  (890, 922, 938, 954, 986, 1022, 1038, 1043, 1070, 1162, 1199, 1225).
- `buildSplitGPUProgram` reads arguments 0 to 5, with `merge ? 2 : 1` and
  `merge ? 1 : 2` selecting the record and sink pointers (1358-1442).

Host side:

- `python/swage/_segmented_qualification.py` holds eight launch tuples that
  repeat the six layouts below.
- Three helpers of `_CudaDriver` in `python/swage/_runtime.py` slice those
  tuples into pointers and counts: `launch_segmented` (`[:3]`),
  `launch_segmented_tasks` (`[:4]`, also used by `launch_segmented_mixed`),
  and `launch_persistent` (`[:10]`).
- `launchKernel` in `python/SwageExtensionNanobind.cpp` (80-115) takes
  pointers and then `int32_t` scalars, at most 16 in total.

| Kernel | Pointers, then i32 counts | Launch site in `_segmented_qualification.py` |
|---|---|---|
| direct, softmax | values, offsets, output; value_count, segment_count | `launch_gpu`; `launch_softmax_gpu` |
| task-id (32 and 128 threads) | values, offsets, output, task_ids; value_count, task_count, segment_count | `_launch_segmented_sum_tasks`; `submit` in `_prepare_planned_reduction` |
| fused mixed | values, offsets, output, task_ids; value_count, warp_task_count, cta_task_count, segment_count | `mixed` in `_prepare_planned_reduction` |
| split partial | values, partial_ranges, scratch; value_count, partial_count | `mixed` in `_prepare_planned_reduction` |
| split merge | scratch, output, merge_records; partial_count, merge_count, segment_count | `mixed` in `_prepare_planned_reduction` |
| persistent | values, offsets, output, warp_ids, cta_ids, partial_ranges, partial_merge_ids, merge_records, scratch, counters; value_count, warp_count, cta_count, partial_count, merge_count, segment_count | `launch` in `_prepare_persistent_sum` |

### Where each schedule is emitted

All in `SegmentedReduction.cpp`:

| Piece | Lines |
|---|---|
| Admission (`analyzeSegmentProgram` and its helpers) | 216-408 |
| Region detachment and fusion (`fusionChain`, `detachSegmentProgram`) | 206-214, 411-448 |
| Sequential oracle (`buildSequentialProgram`) | 548-610 |
| Bounds helpers (`clampRange`, `isLoadedIndexInRange`) | 634-664 |
| Shared segment body (`emitSegment`, `emitTaskSegment`) | 756-875 |
| Persistent queue (claim, CTA queue, partial queue with publish and merge, warp queue) | 877-1235 |
| Fused mixed | 1237-1294 |
| Direct and task-id | 1296-1314 |
| Split partial and merge (`buildSplitGPUProgram`) | 1317-1450 |
| Pass classes, option checks, factories, registration | 1452-1760 |

The pipeline in `lib/CAPI/Codegen.cpp` (232-241) is the Swage pass, then
`convert-scf-to-cf` and `convert-gpu-to-nvvm{index-bitwidth=64}` nested on
`gpu.module`. `replaceLibdeviceCalls` (121-148) and `emitPTX` (300-328)
follow as plain functions.

### What `swage_plan` contains and who reads it

- Contents: `#swage_plan.policy<warp|cta>`, `!swage_plan.task_range`, and
  `swage_plan.classify`
  (`include/swage/Dialect/SwagePlan/IR/SwagePlanOps.td`). The verifier
  resolves a sibling symbol inside `verify()`
  (`lib/Dialect/SwagePlan/IR/SwagePlanDialect.cpp:73-95`). The host
  classifier (`lib/Dialect/SwagePlan/IR/TaskClassifier.cpp`) is compiled
  into the dialect library.
- Writer: `buildPlanningCompanion` (487-515) adds a private
  `@<f>__swage_plan`.
- Reader of plan IR: only `swageMaterializeSegmentedPlan`
  (`lib/CAPI/Codegen.cpp:500-569`). It clones the module, runs the pass,
  finds the one `classify`, and reads back the two limits its caller passed
  in. It then calls `classifyTasks` and buckets the descriptors into four
  flat arrays (541-567).
- The runtime calls that function once per program and pair of limits, on a
  layout without segments, to admit the program (`_admit_program`). It
  classifies every real layout with `swageClassifySegments`
  (`lib/CAPI/Codegen.cpp:571-601`), which takes no module: it calls
  `classifyTaskRecords` in `TaskClassifier.cpp` and returns all records in
  one buffer.
- Not a reader: no lowering reads plan IR. Running a GPU pass after
  `--swage-to-plan` fails, because `source.erase()` breaks the symbol the
  `classify` operation refers to.
- Schedule logic outside the IR, all in `_segmented_qualification.py`:
  - block sizes (`_WARP_BLOCK`, `_CTA_BLOCK`, `_SPLIT_BLOCK`,
    `_PERSISTENT_BLOCK`);
  - the selection rule (`use_direct_cta` in `_prepare_planned_reduction`)
    and the element-work walk with its weights
    (`_has_small_element_program`);
  - grid formulas (the fused grid in `mixed`; the work groups and resident
    blocks in `_prepare_persistent_sum`);
  - record strides (the pointer arithmetic over the record buffer in
    `_prepare_planned_reduction` and `_prepare_persistent_sum`);
  - the counter layout (`3 + merge_count` in `_prepare_persistent_sum`);
  - launch ordering.

### Every literal that encodes the target

| Literal | Where | Meaning |
|---|---|---|
| 32 | 621, 1565 | warp width in the power-of-two rule and its diagnostic |
| 32 | 810-812 | butterfly bound and `gpu.shuffle` width |
| 32 | 904 | claim broadcast shuffle width |
| 32 | 1194, 1240 | lane and physical-warp arithmetic |
| 32 | 1306 | `blockSize == 32` selects the shuffle reduction |
| 3, 4 | 1238-1248 | four warp slots per fused block |
| 4, 8 | 880-881 | persistent partial and warp claim batches |
| 128 | 1532; `lib/CAPI/Codegen.cpp:440` | fused block |
| 512 | 1322, 1538, 1739; `lib/CAPI/Codegen.cpp:457, 474, 491` | split and persistent block |
| 1024 | 1527; `lib/Conversion/FixedBlockToGPU/FixedBlockToGPU.cpp:324`; `lib/CAPI/Codegen.cpp:176`; `python/SwageExtensionNanobind.cpp:89` | largest block |
| `nvvm.reqntid` | 703-705, 1347-1349; `FixedBlockToGPU.cpp:257-259` | launch-width contract |
| `NVVM::MembarOp` (GPU scope) | 1080, 1118 | fence |
| `memref<2xi32>` workgroup | 708-711 | two claim broadcast slots |
| counters 0, 1, 2, then 3 + merge id | 921, 953, 1053, 1198; `_prepare_persistent_sum` | queue counter layout |
| record strides 2 and 3 | 981, 1062, 1121, 1365; `lib/CAPI/Codegen.cpp:553, 557`; `classifyTaskRecords`; the record pointer arithmetic in `_prepare_planned_reduction` and `_prepare_persistent_sum` | partial and merge records |
| `sm_` parser, 12 processors | `lib/CAPI/Codegen.cpp:79-110` | admitted processors |
| `nvptx64-nvidia-cuda` | `lib/CAPI/Codegen.cpp:302, 352` | triple |
| 32, 4096 | 1718, 1722; `python/SwageExtensionNanobind.cpp:458-459, 465-466`; `_CTA_CHUNK_ELEMENTS` and the defaults of `_prepare_planned_sum`, `_prepare_planned_reduction`, and `_prepare_persistent_sum` | planning defaults |
| 32, 128, 512, `(n + 3) // 4`, `512 // 32` | the four block constants, `_validate_warp_count`, `_launch_segmented_sum_tasks`, `mixed`, and `_prepare_persistent_sum` | host mirrors |
| "divided by 32" | `include/swage-c/Codegen.h:92-96` | documentation |
| `_CTA_CHUNK_ELEMENTS`, `_PERSISTENT_BLOCK` | `benchmarks/benchmark_persistent_sum.py:32-33` | benchmark copy |

Left as they are, and named so the list is complete: `__nv_exp2f`
(`lib/CAPI/Codegen.cpp:124`), `libcuda.so.1`
(`python/SwageExtensionNanobind.cpp:66`), the `f"sm_{major}{minor}"` string
in `_target`, and two upstream constants this project cannot change:
`kSubgroupSize` (32) in
`mlir/lib/Dialect/GPU/Transforms/AllReduceLowering.cpp` and `kWarpSize` in
`mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp:219`.

### What the pinned LLVM provides

Spellings confirmed in LLVM 22.1.8. Paths are relative to the LLVM source
tree.

- Conversion patterns (`mlir/include/mlir/Transforms/DialectConversion.h`):
  `OpConversionPattern<Op>` with both the 1:1 and the `OneToNOpAdaptor`
  overload, `ConversionTarget::{addLegalDialect, addIllegalDialect,
  addDynamicallyLegalOp, markOpRecursivelyLegal}`, and
  `applyFullConversion`.
- Rewriter hooks (same header): `replaceAllUsesWith(Value, ValueRange)`
  ("supports both 1:1 and 1:N replacements"), `legalize(Region *)` (line
  1025), `inlineBlockBefore`, `getRemappedValue`, `eraseOp`.
- Rollback: `ConversionConfig::allowPatternRollback` defaults to `true`;
  `false` is labelled experimental.
- Operand remapping (`mlir/lib/Transforms/Utils/DialectConversion.cpp`,
  `remapValues`, line 1485): a pattern built without a type converter
  receives "the most recently mapped values" for each operand.
- Replaced operations (same file, 1884-1975): in rollback mode a replaced
  operation stays in place until commit; without rollback it is erased at
  once. An upstream TODO near line 1242 points toward the non-rollback
  mode, so the proposed design never reads a replaced operation.
- Greedy driver: `applyPatternsGreedily`
  (`mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h:177`).
- Argument attributes: `Dialect::verifyRegionArgAttribute`
  (`mlir/include/mlir/IR/Dialect.h:132`), called from
  `mlir/include/mlir/Interfaces/FunctionInterfaces.h:186`. The syntax
  `%arg0: i1 {dialect.attr = 10 : i64}` is in
  `mlir/test/IR/parser.mlir:910`.
- Precedent for replacing a function operation inside a conversion: the
  `GPUFuncOpLowering` pattern in
  `mlir/lib/Conversion/GPUCommon/GPUOpsLowering.cpp`.
- Driver extensions: `registerAllExtensions`
  (`mlir/include/mlir/InitAllExtensions.h:25`).

## Decision

### Overview

```text
swage (roles on arguments)
  --swage-to-plan='schedule=...'   admit, fuse maps, build one plan function per kernel
swage_plan (plan functions: kernel signature + task ops with explicit bounds)
  --swage-plan-to-gpu              per-function dialect conversion, no options
gpu + scf + arith + llvm           unchanged from today, byte for byte
```

How the design answers each objection the two external reviews raised:

| Objection | Answer |
|---|---|
| The lowerings recognize one function shape and emit kernels over a positional ABI | a per-function conversion with operation patterns; any number of functions; the ABI declared by `swage.role` and by the plan function signature |
| `swage_plan` is not a pipeline stage and nothing consumes it | the conversion consumes plan operations; `classify`, `task_range`, and the companion are removed |
| Memory safety is enforced by the host and by argument position, not in the IR | each device bound is a required operand of a plan operation |
| The emitters check module-level symbol constraints only through the verifier, after they have mutated | symbol checks are part of admission; the conversion rolls back on failure |
| The dialect does not say when a mapped segment is evaluated | a mapped segment is a lazy view; fusion is a rewrite rooted at the consumer |
| `swage-opt` aborts on the nested NVVM pipeline | `registerAllExtensions` in `swage-opt` (step 0) |
| Everything below the semantic IR is specific to one vendor, with a literal warp width of 32 | one target description read by planner, conversion, C API, and host; the use of NVVM is reduced to two hooks |

The absence of an optimization pipeline and of libdevice, and the fixed
vector-add recognizer, are out of scope.

### ABI declaration: `swage.role`

Implemented in step 2. The role is an enum attribute in
`include/swage/Dialect/Swage/IR/SwageOps.td` (`Swage_ArgumentRole`,
mnemonic `role`), attached as an argument attribute:

```mlir
func.func @segmented_sum(
    %values: memref<?xf32> {swage.role = #swage.role<values>},
    %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
    %output: memref<?xf32> {swage.role = #swage.role<output>},
    %value_count: i32 {swage.role = #swage.role<value_count>},
    %segment_count: i32 {swage.role = #swage.role<segment_count>})
```

The contract of each role, stated in the dialect description:

- `values` and `offsets` are the buffers `swage.make_segment` binds.
- `output` is the buffer of the terminal.
- `value_count` is the element count that bounds every range into `values`.
  For a `map_store` program it also bounds the store, as ADR-0012 already
  says.
- `segment_count` is the extent of `swage.segment_id 0` and the bound of
  every loaded segment index.

Verification:

- `SwageDialect::verifyRegionArgAttribute` rejects a role on the wrong
  type and a duplicated role. The negative cases are in
  `test/Dialect/Swage/invalid-roles.mlir`.
- Admission (`readSegmentABI` in `SegmentedReduction.cpp`) requires every
  argument to carry a role and each role exactly once. There is no
  positional default, and the argument order is free:
  `roles-reordered.mlir` lowers a function with its arguments reversed to
  the same kernel.

Types: the element type comes from the `values` argument. The index word
type comes from the `offsets` element type, and the count types have to
equal it. The admitted set stays f32 and i32, expressed as two small
predicates instead of a signature comparison.

Why the counts are declared and the buffers are declared too: no operation
refers to the counts, so they cannot be derived. The buffers could be
derived from `make_segment` and the terminal. Declaring them too gives
closed-world admission and lets the existing binding diagnostics keep their
text.

The SCF lowering removes the role attributes it consumed. The runner
tests pipe its output into upstream `mlir-opt`, which does not know the
Swage dialect.

### Functions, kernel layouts, and symbols

Implemented in step 2, in the passes that existed before the plan stage.
The planner and the conversion apply the same rules since step 4, and after
step 10 "a pass" below is the planner.

- A pass lowers every function that holds a Swage operation and leaves the
  other functions as they are. A module without a segment function is left
  unchanged. `function=<symbol>` restricts a pass to one function.
- A pass admits every function it will lower before it changes any of them.
- A GPU lowering requires that the function has no symbol use and that the
  names it creates are free: `<kernel>_module`, and `<kernel>` when it
  differs from the function name, which is the case for the split stages.
  `invalid-kernel-symbols.mlir` and `invalid-split-kernel-symbols.mlir`
  hold the negative cases, and `two-kernels.mlir` the positive ones.
- The emitters find each kernel parameter through
  `include/swage/Dialect/SwagePlan/IR/KernelLayout.h`, which holds the six
  parameter layouts of the context section, so no emitter reads a parameter
  by position.
- The C API passes `kernelName` as `function` and selects the `gpu.module`
  by the symbol `<kernel>_module`. `swageMaterializeSegmentedPlan` takes a
  `kernelName` as well, so no entry point keeps a one-function rule.

### The plan stage

Implemented in step 4 for the direct and task-id schedules and in steps 5
to 9 for the others.

Principle: plan IR holds what the lowering consumes. A planning parameter
that no kernel depends on stays a host parameter.

`warp_max_elements` and `cta_chunk_elements` are in the second group. No
kernel reads them; they only steer the host classifier. They have left the
IR: the planner takes no limit, and `swageMaterializeSegmentedPlan` checks
them.

Plan function: a `func.func` whose signature is the kernel ABI and whose
body is one task operation plus `return`. It carries
`swage_plan.block_threads = N : i32`, verified by
`SwagePlanDialect::verifyOperationAttribute`, which also verifies the shape
of the function: no result, every argument a signless integer or a rank-one
memref with a dynamic size, the identity layout, and the default memory
space, and a body of one task operation and a return.

Planner sequence (`--swage-to-plan`, `planSegmentFunctions`):

1. Run the read-only admission, per function, with the symbol checks of
   step 2. Every function is admitted before any is changed.
2. Fuse maps (see "Map fusion").
3. Absorb `swage.segment_id`, `swage.make_segment`, and the scalar
   `memref.store` into the task operation.
4. Build the plan function signature from the layout table,
   `kernelLayout(kind)` in
   `include/swage/Dialect/SwagePlan/IR/KernelLayout.h`. The five arguments
   the segment function declared keep their `swage.role`; the arguments a
   schedule adds carry none.
5. Move the reductions into the task region in program order and a map
   store after them, which is the order the kernel runs them in and the
   order the emitters used.

Why the planner absorbs those three operations: no per-operation pattern
can lower them.

- A loaded segment id comes with an in-range predicate that must gate the
  store, so it is two values.
- The scalar store is schedule logic that only the leader thread performs.
- No operation refers to the counts.

Turning them into operands is what makes the bounds structural.

Assembly, for the direct schedule at 128 threads:

```mlir
func.func @segmented_sum(%values: memref<?xf32> {...}, %offsets: memref<?xi32> {...},
    %output: memref<?xf32> {...}, %value_count: i32 {...}, %segment_count: i32 {...})
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
  return
}
```

Each operand group carries its types, so the optional groups (`ids`,
`task_count`, `into`) parse without ambiguity.

Operations. All six exist:

| Operation | Operands | Regions | Bounds the operation requires |
|---|---|---|---|
| `swage_plan.tasks` | values, offsets, value_count, segment_count; optional ids and task_count; optional output; attribute `policy` | 1 | range clamped to value_count; task index below task_count (or segment_count without ids); loaded id below segment_count |
| `swage_plan.partial_tasks` | values, value_count, ranges, partial_count, scratch | 1 | range clamped to value_count; task below partial_count |
| `swage_plan.merge_tasks` | scratch, partial_count, merges, merge_count, output, segment_count | 1 | range clamped to partial_count; output index below segment_count |
| `swage_plan.fused_tasks` | values, offsets, output, ids, value_count, warp_task_count, cta_task_count, segment_count | 2 (warp, CTA) | as `tasks` with ids |
| `swage_plan.persistent_tasks` | the ten buffers and six counts of the persistent layout | 4 (CTA, partial, merge, warp) | all of the above, plus merge id below merge_count |
| `swage_plan.yield` | optional scalar | terminator | none |

Notes on the operations:

- The region of `tasks` has one block argument, the bound segment, of type
  `!swage.segment<T>`. It contains only `swage.reduce`, `swage.map_store`,
  and the terminator, and every consumer reads the block argument. The
  region is not isolated: a map store names the output argument of the
  function, and a reduction captures the results before it.
- The count operands are mandatory in ODS, so a plan function cannot omit a
  bound. The verifier ties the counts and the task buffer to the element
  type of the offsets, and the bound segment to the element type of the
  values.
- `into` and a yielded scalar come together. A region that ends in a map
  store yields nothing and has no `into`.
- The two regions of `fused_tasks` hold the same program: the planner
  fills the warp region and copies it into the block region. They yield a
  scalar and hold no store. `policyOfRegion` gives `warp` for the first
  region and `cta` for the second.
- The merge region holds an identity `swage.reduce` of the same kind over
  scratch, which the planner writes instead of moving the reduction of the
  function. "The merge never runs the element program" is a property of
  plan IR: `SwageToPlan/split-merge.mlir` plans a program with a map and a
  transformed reduction and finds no arithmetic in the merge plan.
- The four regions of `persistent_tasks` are task regions like the
  others: the block and warp regions bind a segment of the values and
  yield into the output, the partial region binds a chunk of the values
  and yields into scratch, and the merge region binds a range of scratch
  and yields into the output. The planner fills the block region with the
  program, copies it into the partial and warp regions, and writes the
  identity reduction of the merge. `policyOfRegion` gives `warp` for the
  last region and `cta` for the three before it. The operation says
  nothing about claims, barriers, or fences; those belong to the
  conversion.
- `policy` reuses the `#swage_plan.policy` attribute. The task-id schedule
  uses `warp` exactly when `block_threads` equals the subgroup width. The
  direct schedule always uses `cta`.
- `policy` has a third case, `sequential`, for the oracle (step 5). The
  host classifier never produces it. A sequential task operation takes no
  task buffer, and a function with a launch width cannot hold one; the
  dialect verifies both.

One function to one or more kernels: the planner replaces the semantic
function by the plan function of the requested kernel. With a list of
schedules it plans one function per kernel, in the order of the list: every
kernel but the last is planned from a copy of the semantic function.

| `schedule=` | Plan function | Operation | Block threads | Exists |
|---|---|---|---|---|
| `direct` | `@f` | `tasks` | `block-threads` option, checked by the target | yes |
| `task-ids` | `@f` | `tasks` with ids | same | yes |
| `fused-mixed` | `@f` | `fused_tasks` | target value (128) | yes |
| `split-partial` | `@f__partial` | `partial_tasks` | target value (512) | yes |
| `split-merge` | `@f__merge` | `merge_tasks` | target value (512) | yes |
| `persistent` | `@f` | `persistent_tasks` | target value (512) | yes |
| `sequential` | `@f` (kept, callers allowed) | `tasks policy<sequential>` | none | yes |

- `schedule` defaults to `direct` and `block-threads` to the block-task
  width of the target, 128.
- The option takes a list, for example `schedule=task-ids,split-partial`,
  which gives two plan functions and later two `gpu.module` operations.
  `block-threads` applies to the direct and task-id kernels; the target
  fixes the width of every other kernel.
- Two schedules that name the same kernel, such as `direct,task-ids`, are a
  diagnostic, and the sequential schedule, which keeps its function, stands
  alone.
- A `function=<symbol>` option restricts planning to one function. Without
  it every function with segment operations is planned, and bystanders are
  untouched. The passes of step 2 already work this way.
- Module and kernel names are unchanged (`<kernel>_module`).

Admission additions, implemented in step 2: for GPU schedules, the function
has no symbol uses, and `<kernel>` and `<kernel>_module` are free. Both are
diagnosed before any mutation.

Removed in step 4: `swage_plan.classify`, `!swage_plan.task_range`, the
companion, and `hasCanonicalSemanticABI`. This record supersedes the dialect
boundary of ADR-0014, and ADR-0014 says so.

What stays on the host, and where it lives:

| Item | Where | Why |
|---|---|---|
| `classifyTasks`, `classifyTaskRecords`, and their limits | `TaskClassifier.cpp`, unchanged | they need runtime offsets |
| Record layouts | `include/swage/Dialect/SwagePlan/IR/TaskRecords.h`, read by `classifyTaskRecords`, by the plan call, and by the lowerings that load a record (step 6) | the strides exist once on the compiler side; the runtime library of an artifact, which is plain C without LLVM, keeps its own and is compared with the classifier by `RuntimeTest` |
| Element-work estimate | `swage::estimateElementWork` in the planner library, through `swageEstimateElementWork` and the binding; `_has_small_element_program` compares it with the 32-unit budget | the weights live once, in C++, and the budget stays with the selection rule on the host |
| The two-chunk selection rule | `_prepare_planned_reduction` | it needs the device SM count and the runtime layout |
| Launch order, stream dependencies, scratch and counter storage, version and context checks | `_segmented_qualification.py` | runtime state |
| Grid sizes | `_segmented_qualification.py`, computed from target description fields | launch-time arithmetic |

`swageMaterializeSegmentedPlan` no longer runs a pass. It checks the
limits, runs the planner's admission on the named function
(`admitTaskProgram`), and classifies. Since step 6 it classifies through
`classifyTaskRecords` and hands each callback its slice of the one record
buffer. It takes i64 offsets and the record classifier takes i32, so it
first refuses an offset outside i32, with the message and the precedence of
the descriptor classifier. `swageClassifySegments` is unchanged.

### The conversion: `--swage-plan-to-gpu`

Implemented in step 4 for `tasks` and in steps 6 to 9 for the other task
operations.

Location and shape:

- Library `MLIRSwagePlanToGPU` under `lib/Conversion/SwagePlanToGPU/`.
- One pass, `SwagePlanToGPUPass`, with no options, and one function,
  `convertPlanToGPU(module, target)`, which takes a
  `const TargetDescription &`.
- It calls `applyFullConversion` on the module with the default
  `ConversionConfig`.

Legality:

- The `swage` and `swage_plan` dialects are illegal, a `func.func` with
  `swage_plan.block_threads` is illegal, and so is the `func.return` of one.
- A `func.func` without `swage_plan.block_threads` is recursively legal, so
  unplanned functions and bystanders are untouched.
- Every other operation is legal. The conversion emits `arith`, `scf`,
  `gpu`, and `llvm` operations and the launch-width attribute of the
  target, and it leaves what it does not own as it is: globals, kernel
  modules that already exist, and operations of dialects it does not know.
  `bystanders.mlir` pins that.

Signature mapping, without a `TypeConverter`: `PlanKernelFuncPattern`
creates `gpu.module @<name>_module` and a `gpu.func` kernel. Each memref
argument becomes `!llvm.ptr`, and each integer stays as it is, in order.

Because the operation patterns have no converter, the driver hands them the
most recently mapped value of each operand: the pointer for a buffer, the
unchanged `i32` for a count. No materialization and no
`unrealized_conversion_cast` is involved.

Patterns never read an element type from a buffer operand. The element type
comes from the segment type, and the word type from the count operands,
which the operation verifier ties to the buffers.

Patterns:

| Pattern | Root | Emits | Exists |
|---|---|---|---|
| `PlanKernelFuncPattern` | `func.func` with `swage_plan.block_threads` | module, kernel shell, launch width through the target hook; the two workgroup claim slots when the body holds `persistent_tasks` | yes |
| `PlanKernelReturnPattern` | `func.return` of a plan function | `gpu.return` | yes |
| `TasksPattern` | `tasks` | prelude, block-uniform guard, binding (offsets loads, `clampRange`, `isLoadedIndexInRange`), then the sink | yes |
| `PartialTasksPattern` | `partial_tasks` | prelude, guard, the range record of the chunk (`loadRecordField`, `clampRange`), then the store of the result in the scratch slot of the task | yes |
| `MergeTasksPattern` | `merge_tasks` | the same shape over merge records: the loaded segment is compared with the segment count and gates the store | yes |
| `FusedTasksPattern` | `fused_tasks` | the fused skeleton, with slots `block_threads / subgroupWidth`; each of its two sites binds and converts a region through `convertSegmentTask`, which `TasksPattern` uses too | yes |
| `PersistentTasksPattern` | `persistent_tasks` | the persistent skeleton: claims, barriers, fences. The block and warp regions go through `convertSegmentTask`, the partial and merge regions through `convertRangeTask`, which `PartialTasksPattern` and `MergeTasksPattern` use too | yes |
| `ReducePattern` | `swage.reduce` | identity, strided `scf.for`, element program, combine; then the shuffle tree (warp) or `gpu.all_reduce uniform` (CTA) | yes |
| `MapStorePattern` | `swage.map_store` | the guard-free strided store loop | yes |

How a region reaches its consumers, per site:

1. The task-operation pattern emits the binding.
2. It replaces the segment argument of the region with four values (base
   pointer, first, end, stride) using
   `replaceAllUsesWith(Value, ValueRange)`.
3. It legalizes each consumer with `rewriter.legalize(Operation *)` while
   the task operation is still the parent.
4. It moves what the consumer patterns created into the guarded block,
   reads the yielded value with `getRemappedValue`, and emits the sink.

The consumer patterns take the four values from their `OneToNOpAdaptor`.
They take the site policy from `swage_plan::policyOfRegion(Region *)`,
which dispatches on the parent operation. No state lives outside the IR,
and nothing depends on reading a replaced operation.

Two points where the pinned driver made the mechanics differ from the first
sketch of this record, both found in step 4 and neither needing the
fallback:

- Moving, not inlining. `inlineBlockBefore` replaces every argument of the
  block it inlines, and rollback mode asserts when a value is replaced
  twice. After the 1:N replacement of the segment argument the task pattern
  therefore moves the converted operations with `moveOpBefore`, which is
  what `inlineBlockBefore` does itself when a listener is attached. It
  legalizes the consumers one by one instead of calling
  `legalize(Region *)`, which would also visit the `swage_plan.yield`.
- Legalize, then move the body. The task pattern needs the launch width,
  which is an attribute of the plan function. The function pattern
  therefore legalizes the body while the `func.func` is still its parent
  and then moves the converted operations into the kernel, instead of
  inlining the body first. The return pattern tests the same parent.

Why emission is byte-identical: the patterns call the same functions in the
same order as the emitters (prelude, guard, binding, one block of operations
per reduction stage in program order, sink). The functions are in
`include/swage/Conversion/SwagePlanToGPU/Emission.h`, which the schedules
that are still emitted by the segmented lowering share. A fused region
clones the same operation sequence that `evaluateElement` inlined.

Reduction kinds are in one place: `identityFor`, `combine`, and
`allReduceOperationFor` in `Emission.h`, three functions with one case per
kind, instead of a table type. They replace the inline ternaries.

Failure behavior:

- Admission stays read-only in the planner, so diagnostics name the
  admission rule.
- With rollback on, `applyFullConversion` restores the module when a pattern
  fails. A failed pattern can only report "failed to legalize", so the
  conversion checks every plan function before it starts and reports each
  rule with the function it applies to: the launch width against the
  target, the element and word types against the admitted ones,
  `policy<warp>` against the subgroup width, the element programs against
  the admitted operations and kinds, and the kernel symbols. These checks
  matter for plan IR written by hand; the planner produces only plan
  functions that pass. `SwagePlanToGPU/invalid.mlir` holds the cases.

The oracle (`--swage-plan-to-scf`, `lib/Conversion/SwagePlanToSCF/`),
implemented in step 5:

- The sequential schedule plans a function in place: it keeps its
  signature, its roles, and its callers and gets no launch width.
- `SequentialTasksPattern` emits one loop over the segments, reads the
  range of each segment from the offsets without a clamp, binds the segment
  to the values memref, and legalizes and moves the consumers as the kernel
  pattern does.
- `ReducePattern` and `MapStorePattern` are the ones the kernel conversion
  uses, in `ConsumerPatterns.cpp`. The emission functions load and store
  through a memref or a pointer by the type of the buffer they are given,
  and `policy<sequential>` combines nothing across threads.
- The conversion removes the `swage.role` attributes of the function, so
  its output parses without the Swage dialects. `buildSequentialProgram` is
  deleted, and the oracle is `--swage-to-plan='schedule=sequential'`
  followed by this conversion.
- The conversion checks the element and word types and the element
  programs of every sequential task operation before it changes anything,
  as the kernel conversion does.

C API (`lib/CAPI/Codegen.cpp`):

- The compile functions add two passes, the planner with
  `function=kernelName` and the schedule of the entry point and then the
  conversion, ahead of the unchanged nested NVVM steps. The pass manager
  verifies the plan between the two.
- The entry point of the direct and task-id kernels takes the launch width
  from its caller and checks it before it plans, so its diagnostic still
  names the power-of-two warp rule.
- The `gpu.module` is selected by symbol (`<expected kernel>_module`)
  instead of by count.
- The six compile entry points and their signatures stay.

### Map fusion

Implemented in step 3. `swage.map` is lowered by fusion, not by emitting
code.

- Rewrite: `fuseMapIntoConsumer` moves the region of a single-consumer map
  ahead of the consumer's own and puts the captures of the map ahead of the
  captures of the consumer. `FuseMapIntoReduce`, `FuseMapIntoMapStore`, and
  `FuseMapIntoMap` are `RewritePattern` classes rooted at the consumer that
  call it.
- Location: `lib/Dialect/Swage/Transforms/FuseMaps.cpp`, library
  `MLIRSwageTransforms`. The pass `--swage-fuse-maps` runs the patterns
  with `applyPatternsGreedily`, with folding, constant merging, and region
  simplification off.
- In the lowerings the same function is applied to the admitted consumers
  directly, not through the greedy driver. The driver also deletes dead
  operations, and an admitted program may hold a reduction that nothing
  reads, which the emitters lower as a stage. `unused-reduction.mlir` pins
  that.
- Order: fusion runs after admission, so "planning requires capture-free
  maps" and "swage.map result must have exactly one segment consumer" stay
  attached to the map. `fusionChain` is deleted.
- Semantics: the `swage.map` description says the result is a lazy view
  read at each consumer. `swage.map_store` has `RecursiveMemoryEffects`.

### Target description

Implemented in step 1. One struct, one instance, in the library
`MLIRSwageTarget`: `include/swage/Target/TargetDescription.h` and
`lib/Target/NVIDIATarget.cpp`. The struct also has a `subgroupCount`
helper, which the block-size diagnostic prints.

```text
struct TargetDescription {
  llvm::StringLiteral name, triple, processorPrefix;  // "nvidia", "nvptx64-nvidia-cuda", "sm_"
  llvm::ArrayRef<uint16_t> processors;                // 80 86 87 88 89 90 100 101 103 110 120 121
  int32_t subgroupWidth;            // 32
  int32_t maxBlockThreads;          // 1024
  int32_t ctaBlockThreads;          // 128: CTA task kernel and fused kernel
  int32_t splitBlockThreads;        // 512
  int32_t persistentBlockThreads;   // 512
  int32_t persistentPartialClaim;   // 4
  int32_t persistentWarpClaim;      // 8
  int32_t defaultWarpMaxElements;   // 32
  int32_t defaultCtaChunkElements;  // 4096
  void (*emitDeviceFence)(OpBuilder &, Location);           // nvvm.membar, GPU scope
  void (*pinLaunchWidth)(gpu::GPUFuncOp, int32_t threads);  // nvvm.reqntid
  constexpr bool admitsBlockThreads(int64_t threads) const; // 1..max, power-of-two subgroup count
  constexpr int32_t slotsPerBlock(int32_t threads) const;   // threads / subgroupWidth
};
const TargetDescription &nvidiaTarget();
```

The fence and the launch-width contract are code, so they are two function
members with one implementation each. There is no enumeration with one case
and no second record.

Readers:

- the segmented lowering, and from step 4 the planner: admitted and fixed
  block sizes, and the existing block-size diagnostics;
- the segmented lowering, and from step 4 the conversion: subgroup width,
  slots, claim batches, and the two hooks;
- `lib/Conversion/FixedBlockToGPU/FixedBlockToGPU.cpp`: the largest block
  and the launch-width hook;
- `lib/CAPI/Codegen.cpp`: processors, prefix, triple, and the fixed block
  widths;
- the host: through `include/swage-c/Target.h`
  (`swageGetTargetDescription`, a plain C struct of the integers and the
  processor list) and a `_target_description()` binding in
  `python/SwageExtensionNanobind.cpp`.

Host reading: `_segmented_qualification.py` reads the description lazily at
first use, inside functions that need the native package anyway, and an
omitted block size or planning limit resolves to its value. The pure tier
uses only `_compile_once` and `_load_once` from that module, so
`tests/python` stays free of the native package.

Copies deliberately kept: `_ADMITTED` in
`python/tests/mlir/test_target_compile.py` and `_ADMITTED_TARGETS` in
`python/swage/env.py`, as independent pins of the processor list (the
environment report works without the native package), and the frozen
benchmark scripts.

### Private ABI changes and launch-site consequences

Kernel parameter layouts did not change. All six rows of the layout table
in the context section are reproduced by `kernelLayout`, so every launch
tuple in `_segmented_qualification.py` and every slice in the `_CudaDriver`
helpers stayed as it was.

The pinned evidence is the signature counts in
`python/tests/mlir/test_segmented_codegen.py` (lines 241-242, 273-274,
386-387, 471-472) and the `.param` counts beside them.

Private interface changes that are not kernel ABIs:

| Change | Step | Consequence |
|---|---|---|
| `_semantic_module` and `_SOFTMAX_MODULE` gain role attributes | 2 | text only; the call in `_runner_module` is unchanged because a call carries no argument attributes |
| `_materialize_segmented_plan` gains `kernel_name` | 2 | one call site, in `_admit_program` |
| Module constants replaced by the description | 1 | same values; the four block constants, `_CTA_CHUNK_ELEMENTS`, `_validate_warp_count`, `_launch_segmented_sum_tasks`, `mixed`, and `_prepare_persistent_sum` |
| Element-work walk replaced by a native call | 4 | `_has_small_element_program` and its caller `_parsed_module` |
| The oracle command of the private runner names the planner and the conversion | 10 | `_execute`, which runs `swage-opt` |

### Tests: what would stay, what would change, how equivalence is shown

Unchanged in RUN and CHECK lines through step 9, with only the input
signature gaining role attributes in step 2:

- All of `test/Conversion/SwageToGPU/`. This includes `persistent.mlir`
  (308 `CHECK-NEXT` lines), `fused-mixed.mlir` (116), `split-partial.mlir`,
  `split-merge.mlir`, `segment-bounds.mlir`, `segment-id-bounds.mlir`, and
  `invalid-block-size.mlir`.
- All positive and runner tests in `test/Conversion/SwageToCPU/`, plus
  `invalid-unverified.mlir`.
- The fixed vector-add tests were not touched at all.

The legacy pass names stayed through step 9 as entry points that called the
planner and a conversion, which is why the RUN lines held. Step 10 rewrote
the RUN lines to the planner and a conversion and left every CHECK line of
a kernel as it was.

Changes, with the reason:

| Step | Test | Reason |
|---|---|---|
| 2 | `test/Conversion/SwageToCPU/invalid-segmented-reduction.mlir`, case at line 48 | the five-argument signature message becomes role and type diagnostics |
| 2 | same file, cases at lines 283 and 293 | the function-count rule is gone; the two-function case becomes a positive test |
| 2 | same file, case at line 415 | text kept; the check is then against the role-declared arguments |
| 2 | `unittests/CodegenCAPITest.cpp:554` (`RejectsAModuleThatAlreadyHoldsAGPUModule`) | a `gpu.module` that already exists is no longer a count error; it becomes a symbol-clash case |
| 2 | `unittests/CodegenCAPITest.cpp:478` (`RejectsAKernelNameThatIsNotTheCompiledKernel`) and `test_kernel_name_must_match_the_compiled_kernel` | selection by symbol changes the mechanism and the message |
| 4 | all four files in `test/Conversion/SwageToPlan/`; `test/Dialect/SwagePlan/{roundtrip,invalid,effects}.mlir`; `APlanCallLoadsThePlanningDialect` | the planner output is plan functions, and `classify` is removed |
| 10 | RUN lines of the GPU tests; the option-combination prefixes in `persistent.mlir`, `fused-mixed.mlir`, `invalid-persistent.mlir` | legacy flags are deleted; `schedule=` makes those combinations inexpressible |

New tests: role round trip and negatives, `fuse-maps.mlir`,
`two-kernels.mlir`, `invalid-kernel-symbols.mlir`, one hand-written plan-IR
file per operation under `test/Conversion/SwagePlanToGPU/`, plan operation
round trip and negatives, and, through step 9, one pipeline test per
schedule that compared the legacy flag with the planner and the conversion.
Step 10 removed those comparisons with the flags; `nvvm-pipeline.mlir`
runs the planned pipeline through the nested NVVM conversion.

Equivalence:

1. No planned text change in any kernel. The gate is the digest matrix that
   step 0 commits: `python/tests/mlir/test_ptx_digests.py` and its data file
   `python/tests/mlir/ptx_digests.json`. Each pair is the SHA-256 of the
   lowered MLIR and of the PTX of one kernel. The matrix holds 552 pairs:
   the fixed add on 12 processors (12), six reduction programs times seven
   entry points times 12 processors (504), the persistent kernel on 12
   (12), and the ragged softmax at 128 and 32 threads on 12 (24). The
   softmax rows are there because the other programs hold no `map_store`.
2. Plan-stage output is new text by definition. Its equivalence is
   transitive: the plan IR it produces lowers to the unchanged goldens and
   digests.
3. Contingency, if a step cannot keep the MLIR text: the PTX digest must
   still match. If it does not, the step needs a differential run on the
   qualification GPU: baseline PTX and new PTX loaded in one process, run
   on the inputs of `test_segmented_runtime.py`,
   `test_segmented_numerics.py`, and `test_persistent_runtime.py`, with
   outputs compared as bit patterns, plus `SWAGE_RACECHECK=1`. This matters
   because the existing runtime tests use a tolerance on non-integer sums
   and would not see a reassociation.

## Rejected alternatives

1. A schedule attribute on the semantic function, plus a transient site
   operation that exists only inside one conversion. This has the smallest
   IR surface and is the fallback named in open question 1. It is rejected
   as the primary design because the printed plan would carry only an
   enumerator and an integer. The kernel ABI and the bounds would stay
   implicit in C++, which is the substance of the objections about the plan
   stage and about memory safety.
2. Per-operation patterns for every operation, with site context in a C++
   side table. State outside the IR is not rolled back, not printable, and
   not testable from text.
3. Fine-grained plan operations for claims, barriers, and fences. This
   would encode the GPU dialect a second time and move the
   ordering-sensitive emission of ADR-0018 into new code with no golden to
   compare against.
4. Memref descriptors as the ABI (counts from `memref.dim`). This changes
   every kernel ABI and the softmax bound at once, and gives up the
   byte-identical PTX gate.
5. A positional default for functions without roles, or a function-level
   index map. Both keep positions as the contract.
6. Keeping `classify` and `task_range` with `SymbolUserOpInterface`.
   Nothing would consume the result, and the limits do not affect any
   kernel.
7. Fusing inside `ReducePattern` by walking to a `swage.map` the driver
   already replaced. This works in LLVM 22.1.8 only because rollback mode
   delays erasure, and upstream is moving away from that mode.
8. A `TypeConverter` for the signature or the segment. With a converter
   attached, every operand of every pattern needs a rule or the pattern
   fails to match. The mapping here is five lines without one.
9. TableGen records for the target with a generator. One record does not
   justify a backend, and the two hooks are code regardless.
10. The target as an IR attribute. The IR could then name a subgroup width
    that the upstream all-reduce lowering does not implement.
11. One `gpu.module` holding all kernels of a schedule set. This changes
    PTX text and load sites; it can follow once the gate is no longer
    needed.
12. Normalizing preludes and constant placement during the migration. Every
    text change gives up a golden. The whole-kernel goldens are better
    relaxed after the rewrite.

## Migration sequence

Every step is gated by the full set unless noted: lit, C++ unit,
`tests/python`, and `python/tests/mlir` on the qualification GPU, with the
digest test at 552 of 552. Rollback for every step is a revert of that
step's commits. No step changes a launch tuple. The legacy pass names held
until step 10, and the six compile entry points keep their signatures.

What steps 1 to 4 built differently from the first sketch of this record.
Each point is also stated where its subject is described:

- C API. `swageMaterializeSegmentedPlan` gained `kernelName` in step 2, as
  question 8 decided, so one C signature changed before step 10. Step 4
  added one exported function, `swageEstimateElementWork`. The estimate is
  `swage::estimateElementWork` in the planner library, not a function of
  the `swage_plan` namespace, and the 32-unit budget it is compared with
  stays in the private runner.
- Kernel module selection. The C API finds the `gpu.module` by its symbol,
  so a module that already holds an unrelated `gpu.module` compiles where
  it used to be refused by count. `CodegenCAPITest` pins both the selection
  and the symbol clash.
- Fixed-block lowering. The symbol checks of step 2 cover the segmented
  lowerings, the planner, and the conversion. `--swage-fixed-block-to-gpu`
  got the same checks in a commit of its own after step 5, with
  `invalid-fixed-kernel-symbols.mlir`.
- Semantic ABI predicate. `hasCanonicalSemanticABI` accepted the five
  arguments in any order from step 2, because roles free the order, until
  step 4 removed it with `swage_plan.classify`.
- Role tests. The negative cases of `swage.role` are in
  `test/Dialect/Swage/invalid-roles.mlir`.
- Fusion driver. The lowerings and the planner apply the fusion rewrite to
  the admitted consumers directly instead of running the greedy driver,
  which would also delete a reduction that nothing reads.
- Planner options. `--swage-to-plan` takes no planning limit. Without
  options it plans the direct kernel at 128 threads. The schedule list
  arrived in step 6, with the first second kernel of one function. The limit rule, "planning limits must satisfy", is checked by
  `swageMaterializeSegmentedPlan` and pinned by `CodegenCAPITest` and
  `test_segmented_classification.py`.
- Conversion mechanics. Consumers are legalized one by one and the results
  moved with `moveOpBefore`, and the function pattern legalizes its body
  before it moves it. "The conversion" gives the reasons. The fallback was
  not needed.
- Classification in the plan call. `swageMaterializeSegmentedPlan`
  classified through `classifyTasks` until step 6, which moved it to
  `classifyTaskRecords` behind a range check and added `TaskRecords.h`.
- Reduction kinds. Three functions in `Emission.h` instead of a table type.
- Commits. Step 4 is four commits (the shared emission functions, the
  admission move, the plan stage and conversion, the element-work
  estimate), each with the digests unchanged.

What steps 5 to 10 built differently from the first sketch:

- Oracle. The sequential schedule is a third case of the policy attribute
  of `swage_plan.tasks`, as decided before step 5. It plans a function in
  place, so it stands alone in a schedule list.
- Schedule lists. `schedule=` takes a list from step 6, and the planner
  refuses a list in which two schedules name the same kernel. After step
  10 that rule is what replaces the option-combination diagnostics of the
  legacy pass.
- Checks on plan IR written by hand. The conversion checks, before it
  changes anything, what the dialect verifier does not promise: the launch
  width the target admits, whole subgroups for a fused block, the
  persistent width, and the identity sum of the persistent regions.
- Persistent claim slots. The function pattern adds them and the
  persistent pattern finds them through the kernel of its operands, as
  step 9 describes.
- Pass factories. `createSwageToPlanPass` and `createSwagePlanToGPUPass`
  have an overload that takes the options and the target description, for
  the C API and for the unit test that lowers with another subgroup width.
- Width diagnostics. The planner reports one rule for `block-threads`,
  "a launch width the target admits". The C API keeps the wording of the
  legacy pass for its direct and task-id entry point, which
  `test_segmented_bounds.py` pins.
- Pipeline comparisons. The tests that compared a legacy flag with the
  two-pass pipeline were removed with the flags in step 10.
  `SwagePlanToGPU/schedule-list.mlir` keeps the schedule list case.

Step 0. Gates. This step does not depend on the decision.

- Files: `python/tests/mlir/test_ptx_digests.py` and its data file;
  `tools/swage-opt/swage-opt.cpp` (`registerAllExtensions`) and its CMake
  file; `test/Conversion/SwageToGPU/nvvm-pipeline.mlir`, which runs the
  nested NVVM pipeline through `swage-opt`.
- Emitted IR: none.
- Gate: the digests pass at the commit that generated them, by
  construction.

Step 1. Target description.

- Files: the new target library and `include/swage-c/Target.h`;
  `SegmentedReduction.cpp`, `FixedBlockToGPU.cpp`, `lib/CAPI/Codegen.cpp`,
  `python/SwageExtensionNanobind.cpp`, `_segmented_qualification.py`;
  `unittests/TargetDescriptionTest.cpp`.
- Emitted IR: none.
- Gate, in addition:
  - `admitsBlockThreads` admits exactly 192 of the sizes 1 to 1024 and
    agrees with `hasPowerOfTwoWarpCount` on every size;
  - a unit test lowers with a copy of the description at subgroup width 16
    and checks four shuffle steps of width 16 at the GPU-dialect level,
    which proves the emitter reads the field;
  - `invalid-block-size.mlir` and `test_target_compile.py`.

Step 2. Roles, per-function admission, any number of functions.

- Files: `SwageOps.td`, `lib/Dialect/Swage/IR/SwageDialect.cpp`,
  `KernelLayout.h`, `SegmentedReduction.cpp` (admission per function, role
  lookups replacing every positional read, the symbol checks, a loop over
  functions), `lib/CAPI/Codegen.cpp`, `python/SwageExtensionNanobind.cpp`,
  `_segmented_qualification.py`, the 37 lit inputs with segment programs,
  and the tests listed above.
- Emitted IR: a module with several functions would give one `gpu.module`
  per function. Single-function output is unchanged.
- Gate: `two-kernels.mlir`; `invalid-kernel-symbols.mlir`; CHECK lines
  unchanged.

Step 3. Map fusion.

- Files: the fusion library, `SegmentedReduction.cpp` (admit, fuse, emit;
  `fusionChain` deleted), the two `SwageOps.td` edits, `fuse-maps.mlir`.
- Emitted IR: none.
- Gate: `segmented-sum-map.mlir`, `ragged-softmax.mlir`,
  `split-partial.mlir`, `SwageToCPU/segmented-sum-map-chain.mlir`, and the
  runner tests.

Step 4. Plan stage and conversion for direct and task-id (softmax
included).

- Files: `SwagePlanOps.td` (`tasks`, `yield`, the attribute check;
  `classify` and `task_range` removed), `SwagePlanDialect.cpp`, new
  `lib/Conversion/SwageToPlan/` (admission, the planner, the element-work
  estimate), new `lib/Conversion/SwagePlanToGPU/` (the emission functions
  and the conversion), `SegmentedReduction.cpp` (the direct and task-id
  branch deleted; the legacy pass routes these schedules through the new
  path), `lib/CAPI/Codegen.cpp`, `_segmented_qualification.py`, the status
  of this record and of ADR-0014, and the internals pages. `TaskRecords.h`
  waits for step 6, the first step whose lowering reads a record of more
  than one word.
- Emitted IR: `--swage-to-plan` output becomes plan functions. GPU output is
  unchanged.
- Gate: the seven direct and task-id lit files, the `DIRECT` and `TASKS`
  prefixes of the two bounds files, `invalid-block-size.mlir`,
  `invalid-segmented-task-ids.mlir`, the one-pipeline test,
  `test_segmented_bounds.py`.
- This step proved the conversion mechanics: the goldens and the digests
  hold, with the two adjustments "The conversion" names and without the
  fallback.

Step 5. Oracle onto the shared patterns.

- Files: new `lib/Conversion/SwagePlanToSCF/` (a `tasks policy<sequential>`
  pattern; `ReducePattern` and `MapStorePattern` move to
  `ConsumerPatterns.cpp` and dispatch on a memref or pointer base); the
  sequential schedule in the planner; `buildSequentialProgram` deleted.
- Emitted IR: none. The oracle output of every lit input is byte-identical
  to the output before the step.
- Gate: all of `SwageToCPU/`, the four runner tests,
  `test_segmented_numerics.py`.

Step 6. Split partial.

- Files: `partial_tasks` and `PartialTasksPattern`; the split-partial
  schedule and the schedule list in the planner; `TaskRecords.h`; the plan
  call on `classifyTaskRecords`; the partial half of `buildSplitGPUProgram`
  deleted, and the merge half moved onto the shared emission functions.
- Emitted IR: none.
- Gate: `split-partial.mlir`; the `PARTIAL` and `SPLIT` prefixes;
  `invalid-split.mlir`.

Step 7. Fused mixed.

- Files: `fused_tasks` and `FusedTasksPattern`; the fused-mixed schedule
  in the planner; the fused branch deleted, which leaves the persistent
  kernel as the one schedule of `--swage-segmented-reduction-to-gpu` that
  is emitted without a plan.
- Emitted IR: none.
- Gate: `fused-mixed.mlir` including the `SYNC` run; the `FUSED` prefixes;
  the mixed cases of `test_segmented_runtime.py`; racecheck run locally.

Step 8. Split merge.

- Files: `merge_tasks` and `MergeTasksPattern`; the split-merge schedule
  in the planner; the rest of the split emitter deleted, so
  `--swage-split-segmented-reduction-to-gpu` only plans and converts.
- Emitted IR: none.
- Gate: `split-merge.mlir` with its `--implicit-check-not=arith.mulf` and
  `RANGE` runs; the `MERGE` prefixes; a plan-level check that the merge
  region is an identity reduce.

Step 9. Persistent queue.

- Files: `persistent_tasks` and `PersistentTasksPattern`; the persistent
  schedule in the planner; the counter layout in `TaskRecords.h`, which the
  conversion reads and the private runner still mirrors as `3 +
  merge_count`; the
  persistent emitter deleted, so `--swage-segmented-reduction-to-gpu` only
  validates its options, plans, and converts.
- Emitted IR: none.
- Gate: `persistent.mlir` (all four prefixes and the round trip);
  `invalid-persistent.mlir`; the `PERSISTENT` prefixes;
  `test_persistent_runtime.py`, run alone thirty times; racecheck; ten
  consecutive native-tier runs.
- What was built differently from the sketch:
  - The persistent pattern runs while the plan function is still its
    parent, so it finds the claim slots through the kernel that owns the
    parameters its operands were replaced by. The function pattern adds
    the slots, as sketched.
  - The conversion checks two more rules before it changes anything, for
    plan functions written by hand: the launch width is the persistent
    width of the target, and every region holds one capture-free identity
    `kind<sum>` reduction. The planner admits no other program for this
    schedule, and the queue kernel has been run with no other.
  - The record index arithmetic of the queue kernel stays as the emitter
    wrote it, because the kernel text is pinned: the kernel reuses its
    index constant one for the fields of a record, and the leader
    predicate of the prelude for its stores, where the split patterns
    create a constant and a comparison per site.

Step 10. Remove the legacy surface.

- Files: `lib/Conversion/SegmentedReduction/` and its header deleted, with
  the three pass names `--swage-segmented-reduction-to-scf`,
  `--swage-segmented-reduction-to-gpu`, and
  `--swage-split-segmented-reduction-to-gpu`; `lib/CAPI/Codegen.cpp` on
  the planner and conversion pass factories; RUN lines rewritten to
  `--swage-to-plan=...` followed by `--swage-plan-to-gpu` or
  `--swage-plan-to-scf`; the option-combination prefixes of
  `persistent.mlir`, `fused-mixed.mlir`, and `invalid-persistent.mlir`
  replaced by schedule list tests, and the messages of
  `invalid-block-size.mlir` by the planner's width rule; the oracle command
  of the private runner; documentation.
- Emitted IR: none. For every lit input, the planner and a conversion
  print what the legacy flag printed at step 9, and fail where it failed.
- Gate: CHECK lines of the kernels unchanged; digests.

## Risks and the test that would detect each

| Risk | Detected by |
|---|---|
| Emission order drifts in a pattern | the `CHECK-NEXT` goldens (492 lines across `persistent.mlir`, `fused-mixed.mlir`, `split-partial.mlir`, and `split-merge.mlir`) and the digest test |
| A barrier ends up under non-uniform control flow | `persistent.mlir`, the `SYNC` run of `fused-mixed.mlir`, racecheck, `test_persistent_runtime.py` |
| A device bound is dropped | `segment-bounds.mlir` and `segment-id-bounds.mlir` on every mode; `test_segmented_bounds.py`; the mandatory count operands and their negative tests |
| The power-of-two rule is lost when 32 becomes data | `invalid-block-size.mlir`; the 192-of-1024 unit test; the block-size cases in `test_segmented_bounds.py` |
| The description disagrees with the 32 that upstream hard-codes | the lit test from step 0 that runs the NVVM pipeline; a unit test pinning `subgroupWidth == 32` with a note to check it again in a pin-bump change |
| Driver behavior the design leans on (`legalize(Region *)`, 1:N block-argument replacement, `scf` builder callbacks inside a conversion pattern) differs from this reading of the sources | step 4 on the smallest schedule; the sanitizer job. If 1:N replacement misbehaves, the fallback is a pure `swage_plan` binding operation that consumers read through their adaptor |
| Wrong module selected, or a symbol clash, with several functions | `two-kernels.mlir`, `invalid-kernel-symbols.mlir`, `CodegenCAPITest`, the bystander tests in `test_segmented_codegen.py` |
| Fusion changes a program | `fuse-maps.mlir`; the unchanged goldens; the CPU runner tests; PyTorch as the third reference at runtime |
| Diagnostics drift | every `invalid-*.mlir` run with `--verify-diagnostics` |
| Host and compiler disagree on a constant | one source through the C API; `test_segmented_classification.py` and the grid-dependent runtime tests |
| Classification output changes when the record layouts move | `TaskClassifierTest`, the plan and classification cases of `CodegenCAPITest`, `test_segmented_classification.py` |

## Scope

Not part of this proposal:

- a second backend, any interface with two implementations, or a driver
  seam;
- new element types, i64 offsets, new reduction kinds;
- lowering `swage.extent`; `map_store` programs without a reduction;
- more math operations, libdevice, or an LLVM optimization pipeline;
- moving the LLVM pin; the public frontend; `FixedBlockToGPU` beyond
  reading the description;
- C API consolidation; relaxing the goldens;
- widening persistent beyond identity sum; changing any heuristic.

How the design would make the later work easier:

- Element and offset types. Patterns would take the element type from
  `!swage.segment<T>` and the word type from the count operands.
  `clampRange` and `isLoadedIndexInRange` already take their width from
  their operands. A new type would be a row in the admission tables plus
  host scalar widths (`launchKernel` takes `int32_t` today), not an emitter
  edit.
- Reduction kinds. One case in each of the three kind functions of
  `Emission.h`. The planner would decide whether a kind may be split.
- `map_store` only, and `swage.extent`. The task region already admits a
  body with no reduction, and the four-value segment makes an extent a
  subtraction.

## Questions decided at acceptance

Each question was answered as recommended.

1. Plan granularity: six printed operations (this record), or a schedule
   attribute plus a transient site operation? Recommended: the six
   operations. They put the kernel ABI and the bounds in printed IR, and
   they let the planner and the lowering be tested separately. The fallback
   has the same conversion mechanics and about one third of the ODS work.
   Under it, the conversion and fusion sections hold, and the
   task-operation patterns become per-function patterns keyed on the
   attribute.
2. Delete `classify`, `task_range`, and the companion, and supersede part
   of ADR-0014? Recommended: yes. The limits steer only host
   classification.
3. Require `swage.role` on every argument? Recommended: yes, with the 37 lit
   inputs and two Python templates edited mechanically in step 2.
4. Keep the committed digest matrix of step 0 as a permanent gate, with
   regeneration allowed only in a pin-bump change or a change that states
   which kernels move and why? Recommended: yes.
5. Migrate the persistent kernel, or drop it, given that ADR-0018 is still
   proposed after a failed gate? Recommended: migrate it last. It is tested
   and clean under racecheck. If it is dropped instead, step 9 becomes a
   deletion.
6. Move the oracle onto the shared patterns (step 5)? Recommended: yes, as
   its own step. It gives one admission and one kind table, and the
   oracle's own lit and runner tests gate it without the GPU.
7. Move the selection rule into C++ as well? Recommended: only the
   element-work estimate now. Moving the rule would change which kernel
   runs for eligible batches and would touch the frozen benchmark records.
8. Add `kernel_name` to `swageMaterializeSegmentedPlan`? Recommended: yes,
   in step 2, so no entry point keeps a one-function rule.
9. Give plan-added arguments a `swage_plan.role` and bind launches by role?
   Recommended: not now, since nothing would read it. Do it with the i64
   offsets work, when scalar widths start to vary.
10. Keep the legacy pass names after step 9? Recommended: delete them in
    step 10. The C API entry points stay.
11. Normalize the split merge layout to the common order? Recommended: no.
    It would change 72 digest pairs and one launch tuple for no functional
    gain.
12. Name of the C getter: recommended is the neutral
    `swageGetTargetDescription` returning the one record. That keeps a
    vendor name out of the C API without adding a second target.
