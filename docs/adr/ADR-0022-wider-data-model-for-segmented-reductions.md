<!-- docs/adr/ADR-0022-wider-data-model-for-segmented-reductions.md -->
# ADR-0022: Wider data model for the segmented reductions

- Status: accepted; steps 1 to 6 of the migration sequence are
  implemented; superseded in part by
  [ADR-0023](ADR-0023-row-stripe-tile-for-rank-two-values.md), which
  gives the public calls on rank-two values a row-stripe tile and a split
  in place of the one column tile
- Date: 2026-10-03
- Accepted: 2026-10-03, with the recommended answer to every question at the
  end

This record widens what the segmented reductions compute and what they
read: two more kinds, one more element type, a trailing feature dimension,
and 64-bit offsets. It was built one migration step at a time, and all
six steps exist. "Migration sequence" records what each step built and
where it differs from what was proposed.

## Context

`swage.segment_reduce` returns a sum or a maximum of rank-one
`torch.float32` values under `torch.int32` offsets. `swage.segment_softmax`
has the same data model. Callers of ragged data ask for more: a minimum and
a mean, `torch.float64` values, values with a trailing feature dimension
`[N, D]`, and the `torch.int64` offsets that PyTorch and PyTorch Geometric
produce.

ADR-0020 states, under "How the design would make the later work easier",
what such a widening would cost. Each claim was checked against the code
before this decision:

| Claim of ADR-0020 | Finding |
|---|---|
| Patterns take the element type from `!swage.segment<T>` | True for `ReducePattern` and `MapStorePattern` in `lib/Conversion/SwagePlanToGPU/ConsumerPatterns.cpp`. `identityFor` in `lib/Conversion/SwagePlanToGPU/Emission.cpp` built an f32 constant whatever the type, so an f64 reduction would have fed an f32 identity into an f64 addition. |
| `clampRange` and `isLoadedIndexInRange` take their width from their operands | True for `isLoadedIndexInRange`. `clampRange` creates a 32-bit zero, and `loadTaskWord` loads an i32. `emitSegmentBinding` is generic in the word type. |
| A new type is a row in the admission tables, not an emitter edit | `isAdmittedElementType` and `isAdmittedIndexType` are the tables. `verifyRegion` and `verifyOperationCaptures` in `lib/Conversion/SwageToPlan/Admission.cpp` tested for f32 directly, and `identityFor` is an emitter edit. |
| A new kind is one case in each of three functions | The three functions were two-way choices in which every kind that is not `sum` took the branch of `max`. A kind admitted without rewriting them would have computed a maximum under its own name. |
| The planner would decide whether a kind may be split | No such decision point exists. `verifyPlanningProgram` does not look at the kind. |
| The task region already admits a body with no reduction | True of the plan verifier. Admission requires a reduction, and `fuseAdmittedMaps` reads the first one. |
| The four-value segment makes an extent a subtraction | False as built. The binding is base, first, end, and stride, with `first` the start plus the thread index, so `end - first` is the extent on thread zero only. In a merge region the bound range is scratch, whose extent is the partial count. |

What was known and what was not, when the decision was taken:

- `swage-opt` and the pinned `mlir-translate` and `llc` were run on f64
  programs, so the lowering of f64 to PTX text was seen. The f64 CPU oracle
  transport was run through the pinned `mlir-opt` and `mlir-runner`.
- The conventions of `torch.segment_reduce` for the new kinds were checked
  on PyTorch 2.12.0, on the CPU.
- No f64 kernel had run on a device. Step 3 treats the first run as an
  experiment and records what it found here.
- The cost of the rank-two schedule was estimated from instruction counts
  and not measured.

## Decision

### Kinds

`min` is a third case of the three kind functions `identityFor`, `combine`,
and `allReduceOperationFor`. Its identity is positive infinity, its combine
is `arith.minimumf`, and its block reduction is the `minimumf` case of
`gpu.all_reduce`. The three functions are switches over every kind without
a default, so a kind without a case is a compiler diagnostic and never the
lowering of another kind. A minimum mirrors a maximum: an empty segment
gives positive infinity, a NaN element gives NaN, and the result does not
depend on the order. A split works as for the other kinds, because the
merge region already holds an identity reduction of the kind of the
program. The persistent schedule keeps `kind<sum>`.

`mean` is a composition in the semantic IR and not a kind:

```mlir
%sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 { ... }
%n = swage.extent %segment : !swage.segment<f32>
%count = arith.index_cast %n : index to i32
%divisor = arith.sitofp %count : i32 to f32
%mean = arith.divf %sum, %divisor : f32
memref.store %mean, %output[%sid] : memref<?xf32>
```

The division runs once per segment, between the reduction and the store.
The reasons for a composition:

- `include/swage/Dialect/Swage/IR/SwageOps.td` and ADR-0008 define a kind
  as an identity and an order-free combine, which is what lets a lowering
  split and merge with the same kind. A mean has no identity, and a mean of
  partial means is wrong.
- A `kind<mean>` reduction in a merge region would read a range of scratch
  and divide by a number that is not the extent of that range.
- `lib/AGENTS.md` asks for standard dialects for ordinary arithmetic, and a
  division is ordinary arithmetic.
- `swage.extent` exists and has no consumer.

By layer:

- Admission admits `swage.extent` and a scalar epilogue of exactly
  `arith.index_cast`, `arith.sitofp`, and `arith.divf` after the
  reductions.
- The regions of `swage_plan.tasks`, `swage_plan.fused_tasks`, and
  `swage_plan.merge_tasks` take an optional second argument, the extent of
  the semantic segment of the task. `swage_plan.partial_tasks` never takes
  it: a partial task yields the raw sum, and the merge sums the partial
  results and divides once.
- A merge gets its extent from the partial range records, through one
  optional operand on `merge_tasks`. The chunks of a split segment are
  consecutive records, so the extent is the end of the last minus the begin
  of the first. No existing kernel, record, or launch argument list
  changed.
- An empty segment gives NaN, from zero divided by zero, as
  `torch.segment_reduce` returns.

### Element types

float64 is admitted for the four reductions and refused for
`segment_softmax`:

- `!swage.segment<f64>` already verifies, and the plan buffers already
  admit f64. The admission tables admit f32 and f64, the region checks
  compare with the element type of the function, and `identityFor` takes
  the element type. That is the only emitter edit.
- `gpu.shuffle` on f64 lowers to two 32-bit shuffles, so a warp tree is ten
  shuffles where f32 has five. `gpu.all_reduce` lowers for `add`,
  `minimumf`, and `maximumf` on f64.
- The pinned NVPTX backend has no f64 `exp2`: `llc` on `llvm.exp2.f64`
  aborts. Admission therefore refuses `math.exp2` on any type but f32, so
  that a program the GPU cannot compile is refused with a diagnostic on
  both backends. The replacement of libdevice calls in the code generation
  C API stays f32 only: extended to `__nv_exp2`, it would hand the backend
  the intrinsic it aborts on.
- The CPU oracle transport becomes typed: f64 values and results travel as
  64-bit patterns.
- A float64 reference no longer suffices for an f64 sum. The tests use an
  exactly rounded reference.

f16, bf16, and integer values stay refused. A 16-bit shuffle has no
lowering, a half-precision sum needs an accumulator type that no admitted
region operation expresses, and bf16 `ex2.approx` needs a newer processor
than the floor. ADR-0008 gives an integer sum no lowering and no defined
overflow, and `torch.segment_reduce` raises for integer values, so there is
no reference.

### Trailing dimension

`[N, D]` values are a strided column segment in the IR, lowered to a column
tile: one thread reduces one column of one segment.

```mlir
%sid = swage.segment_id 0
%col = swage.segment_id 1
%segment = swage.make_segment %values, %offsets, %sid column(%col)
    : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
%sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 { ... }
memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
```

- The program instance is one segment and one column, and it yields a
  scalar. The segment type stays `!swage.segment<f32>`, so no runtime
  identity enters a type and no thread or block index enters the IR.
- A sixth role, `feature_count`, gives the extent of the second axis.
- The plan has a `column` policy with one kernel per program and no
  split: thread `t` of a block runs the region for the columns `t`,
  `t + block_threads`, and so on, each alone. Nothing is combined across
  threads, so the kernel holds no shuffle, barrier, or shared memory.
- Each thread holds one scalar accumulator per reduction stage, never one
  per column.
- A column sum is added in row order, the order of the CPU oracle. Its
  bound is `(n - 1) * eps * sum(|x|)`, weaker than the rank-one bound for a
  long segment, and its bits do not depend on the batch.
- `[N, 1]` values take the rank-one schedules through a view.

Stated limits of this schedule: a long segment with few columns is reduced
by few threads, and a long segment occupies one block for its whole length.

### int64 offsets

int64 offsets are narrowed on the host after they were checked. The kernels
read a private int32 copy, and the cap of `2**31 - 1` rows and segments
stays:

- The width of an offset is a property of its storage. Under the cap every
  valid offset fits in int32, so narrowing loses nothing and adds no kernel.
- The offsets are validated as int64 first: they start at zero, never
  decrease, and end at or below the row count. Narrowing first would turn
  `[0, 2**32 + 3, 5]` into a valid array.
- The narrowed copy is uploaded as a private int32 tensor, in the same
  tensor as the task records where a call uploads records, and retained on
  the stream like them. No kernel reads the caller's int64 tensor.
- Lifting the cap could not lift the segment count: a launch uses one block
  per task on the x axis of the grid.

int64 therefore does not reach the dialect or the lowering.

### The matrix

- The kernel function name is `segmented_<kind>[_f64][_r2]` and
  `ragged_softmax[_r2]`. f32 rank one stays unsuffixed, so the three
  existing programs keep their text.
- One generator produces every program text and returns the existing text
  for the existing programs.
- No C API entry point is added.
- The 552 digest pairs that existed before this record do not move at any
  step. New cells cover every new program on each variant it admits, on
  `sm_80` and `sm_86`. The f64 `sum` and `max` programs are covered on every
  admitted processor, because f64 `maximum` takes another
  instruction-selection path.
- The artifact moves to format version 2 in step 1, as ADR-0021 records.

### Public contract when every step is in

`swage.segment_reduce(values, offsets, kind, *, out=None)`:

- `values`: contiguous, rank one or two, `torch.float32` or
  `torch.float64`, on the current CUDA device, not requiring grad. Rank two
  is `[N, D]`, reduced along axis 0 within each segment. Nothing is cast.
- `offsets`: contiguous, rank one, `torch.int32` or `torch.int64`, with one
  entry more than there are segments.
- `kind`: `"sum"`, `"max"`, `"min"`, or `"mean"`.
- `out`: the dtype of `values`, shape `[S]` or `[S, D]`.

| Kind | Empty segment | A NaN element |
|---|---|---|
| `sum` | `0.0` | NaN |
| `max` | negative infinity | NaN |
| `min` | positive infinity | NaN |
| `mean` | NaN | NaN |

`swage.segment_softmax` takes rank one or two `torch.float32` values and
refuses float64. On `[N, D]` the softmax is per column.

## Rejected alternatives

- `mean` as a kind with a finalize step. It has a smaller admission
  surface. It makes the meaning of a reduction depend on its parent
  operation, needs a planner rewrite of the kind in a partial task, and
  puts a division inside the reduction pattern.
- A fourth merge-record word for the extent. The persistent kernel reads
  merge records too, so existing digest pairs would move, both classifiers
  would change, and the runtime ABI version would change.
- Offsets and the value count on the merge ABI. Two more arguments, and the
  divisor would come from reloaded offsets while the sums come from
  recorded ranges.
- An i64 load of the offsets with a device clamp. It doubles every kernel
  that reads offsets and gains nothing under the cap.
- Lifting the cap. It needs 64-bit classification on every call and cannot
  lift the segment count.
- A segment of `[D]` rows. The reduction would yield a value of runtime
  size, which the rules forbid.
- `D` launches over strided columns. `D` host round trips.
- A column loop inside the existing row tiles. It keeps the split and the
  rank-one bound, at the cost of doubling four schedules and one
  cross-thread combine per column and segment. It could be added later as a
  planner choice, on measurement.
- A float64 softmax through an `exp2` expansion in the tree. That is a
  math-library decision of its own.

## Migration sequence

Every step keeps the 552 existing digest pairs and is gated by lit, the
unit tests, `tests/python`, and `python/tests/mlir` on the qualification
GPU.

Step 1. `min`, and artifact format version 2. Implemented.

- The kind functions are switches, and `unittests/EmissionTest.cpp` holds
  each kind to its own identity, combine, and block reduction.
- Admission admits `kind<min>`. The kind check is gone, because every kind
  of `swage.reduce` now has a lowering.
- `test/Conversion/SwageToPlan/segmented-min.mlir` and the files of the
  same name under `SwageToCPU` and `SwageToGPU` pin the plan, the oracle,
  and every kernel schedule. The GPU file refuses a maximum in any of them.
- `swage.segment_reduce` takes `kind="min"`.
- The digest matrix gains the identity and the map-chain minimum on `sm_80`
  and `sm_86`, 28 pairs.
- The artifact is format version 2 and holds `segmented_min`.

Step 2. int64 offsets. Python only. Implemented.

- The shared validation of the private runner admits int64 offsets for the
  two public launch paths, validates the int64 host copy, and narrows it.
  The private helpers keep int32.
- `segment_reduce` uploads the narrowed copy behind its task records, or as
  a tensor of its own for the selected CTA schedule. `segment_softmax`
  uploads it as a tensor of its own. A batch without segments uploads
  nothing.
- `python/tests/mlir/test_public_segments.py` pins the refusals by 64-bit
  value, the bits of the int32 call on every distribution and kind and for
  the softmax, and that every kernel reads a copy the call retained on the
  launch stream and never the caller's tensor.
- The fresh-offsets harness has a candidate, `swage_public_call_int64`,
  that times the call on int64 offsets. No record holds it.
- No kernel, no digest, and no artifact changes.

Step 3. float64 reductions, rank one. Implemented.

- The admission tables admit f32 and f64. A function has one element type,
  the one of its values. The output, every region value, and every capture
  must have it, so a function that mixes the two is refused, and so is
  f16. `math.exp2` is admitted in an f32 program only, and the persistent
  schedule is refused for f64 by the planner and by the conversion.
- `identityFor` takes the element type, and `unittests/EmissionTest.cpp`
  holds the identity of each kind to each element type.
- `test/Conversion/SwageToPlan/segmented-f64.mlir` and the files of the
  same name under `SwageToCPU`, with a runner file, and `SwageToGPU` pin
  the plan, the oracle, and the kernel schedules.
  `test/Conversion/SwageToGPU/nvvm-pipeline.mlir` holds an f64 maximum
  through the NVVM conversion.
- `swage.segment_reduce` takes float64 values and returns float64, and
  `out` has the dtype of the values. `swage.segment_softmax` refuses
  float64 values and states the reason. The private runner reads the
  element type of a program from the declaration of its values and requires
  values, output, and scratch of that type.
- The CPU oracle transport is typed: an f64 result travels as a 64-bit
  pattern through `printMemrefI64`.
- The PTX scan of `python/tests/mlir/test_segmented_numerics.py` takes the
  element type of the kernel and reports an instruction of the other width.
  It now also sees instructions with a mixed-case modifier, such as
  `max.NaN.f32`, which it had skipped.
- `python/tests/mlir/test_segmented_codegen.py` holds every compile
  function to the diagnostic for an f64 `math.exp2`.
- The digest matrix gains 210 pairs: the f64 identity sum and the f64
  identity maximum on every admitted processor, and the f64 identity
  minimum, square sum, and map-chain maximum on `sm_80` and `sm_86`.
- The artifact holds three more programs, `segmented_sum_f64`,
  `segmented_max_f64`, and `segmented_min_f64`, in format version 2.

What the first runs on a device showed, on the RTX A6000 (`sm_86`):

- An f64 sum stays within `k * eps64 * sum(|x|)` of the exactly rounded
  sum, with the `k` of the f32 trees, on every static schedule and on the
  one-CTA path. The largest error the committed test measures is
  `0.35 * eps64 * sum(|x|)`. The bits differ between the schedules and
  repeat between launches.
- A maximum and a minimum equal the reference bit for bit, on values that
  differ only below f32 precision, and on NaN, both infinities, signed
  zeros, and subnormals down to `2**-1074`.
- The CPU oracle returns the left-to-right float64 sum bit for bit.
- The PTX is as expected at the decision: ten shuffles for a warp tree,
  forty shuffles and two barriers for a block reduction, a 256-byte shared
  buffer, `add.rn.f64`, and no f32 instruction in an f64 kernel.
- One detail differs from the estimate. The backend expands an f64 maximum
  or minimum into `max.f64` or `min.f64`, `setp.nan.f64`, `setp.eq.f64`,
  and four `selp.f64`, where three selects were expected. The result is the
  IEEE-754 maximum or minimum, as the exact tests show.
- NVIDIA Compute Sanitizer reports no shared-memory hazard in the f64
  kernels.
- With the admission rule removed, an f64 `math.exp2` does not reach the
  backend through the code generation C API: its libdevice check refuses
  the unresolved `__nv_exp2`. `llc` on the intrinsic does abort. The rule
  gives the refusal earlier, by name, and on both backends.

Nothing the device showed contradicts the decision.

Step 4. `mean`, both dtypes, rank one. Implemented.

- Admission collects `swage.extent` and the three operations of the
  epilogue and admits one shape: the extent of the segment of the function,
  cast to the count type, converted to the element type, divides one
  reduction result, and the quotient is what the function stores.
- The plan regions of `tasks`, `fused_tasks`, and `merge_tasks` take the
  extent as a second argument and may then hold the epilogue after their
  consumers. `merge_tasks` has the optional operand `ranges`, tied by its
  verifier to the extent argument. The regions of `partial_tasks` and of
  the persistent queue kernel take neither, and the planner refuses a mean
  on the persistent schedule.
- The planner absorbs `swage.extent` into the region argument, moves the
  epilogue behind the reductions, leaves it out of a partial task, and
  writes a copy of it into the merge region. The merge kernel of such a
  program has the layout `SplitMergeExtent`: scratch, output, merge
  records, and range records, then the three counts of a merge.
- The conversions give a task region its extent as the clamped end minus
  the clamped start of the segment, and a merge region the end of its last
  range record minus the begin of its first, read through the clamped
  range of partials, or zero for an empty one.
- `swage.segment_reduce` takes `kind="mean"`. The runner passes the merge
  kernel of a mean the pointer of the range records, which it already
  uploads. The artifact holds `segmented_mean` and `segmented_mean_f64`.
- `test/Conversion/SwageToPlan/segmented-mean.mlir` requires the division
  in the task, fused, and merge regions and refuses any arithmetic in the
  partial region. The files of the same name under `SwageToCPU`, with a
  runner file, and `SwageToGPU` pin the oracle and the kernels.
- The digest matrix gains 28 pairs, the f32 and the f64 mean on `sm_80`
  and `sm_86`. The 790 pairs before it did not move.

What was built against what was proposed for step 4:

- The proposal let the epilogue be any sequence of the three operations
  over reduction results, the extent, and earlier epilogue results.
  Admission accepts the one sequence of a mean. A merge has to rebuild the
  epilogue from partial results, and a fixed shape makes that a copy with
  two substitutions. The plan verifier is wider than admission: it admits
  the three operations after the consumers of a region that takes an
  extent, in any number.
- The proposal stated the bound of a mean without a condition. A mean lies
  within `(k + 1) * eps * sum(|x|) / n` of the exact mean when its sum does
  not overflow and the mean is not subnormal: a sum that overflows gives an
  infinite mean, and a quotient in the subnormal range is rounded to a
  subnormal. The user guide states both conditions.
- On the device a mean equals the sum of the same schedule divided by the
  length, bit for bit, on every static schedule and on the one-CTA path in
  both element types. The PTX holds one `cvt.rn` from i32 and one `div.rn`
  per task region, in the element type, and none in the partial kernel.

Step 5. Rank-two reductions. Implemented.

- The dialect has the role `feature_count`, the optional operand `column`
  of `swage.make_segment`, tied by its verifier to rank-two values, and a
  rank-two output of `swage.map_store`. `swage.segment_id 1` is the column.
- Admission reads the sixth role, requires rows of an admitted element
  type with a rank-two output and three counts, the two segment ids of the
  two axes, and a store at `output[segment, column]`. Planning admission
  refuses a rank-two function, so every schedule that reads a task buffer
  refuses it with that diagnostic.
- `swage_plan.tasks` takes `feature_count` and has `policy<column>`. Its
  verifier ties the two to rank-two values and a rank-two `into`, and
  refuses `ids`. The plan function keeps the rank-two buffer types, and the
  kernel layout `DirectColumns` adds the feature count to the counts of
  the direct kernel.
- The kernel conversion emits the column tile: the rows of the segment
  clamped to the row count, a loop over the columns of a thread bounded by
  the feature count, the region on the strided run of one column with no
  combination across threads, and a store by every thread at
  `output[segment * columns + column]`.
- The sequential conversion loops over the columns of each segment and
  reads the values through a `memref.reinterpret_cast` to their row-order
  view.
- `swage.segment_reduce` takes `[N, D]` values and returns `[S, D]`. It
  runs the column kernel, validates the offsets, and classifies nothing.
  `[N, 1]` values take the rank-one schedules through a view, and `[N, 0]`
  values launch nothing. The artifact holds eight more programs,
  `segmented_<kind>[_f64]_r2`, each with the one role `column`.
- `test/Conversion/SwageToPlan/segmented-columns.mlir` and the files of the
  same name under `SwageToCPU`, with a runner file, and `SwageToGPU` pin
  the plan, the oracle, and the kernel, which may hold no shuffle, block
  reduction, or barrier. `python/tests/mlir/test_segment_columns.py` holds
  the public contract, and `python/tests/mlir/test_segmented_bounds.py`
  the device bounds.
- The digest matrix gains 16 pairs, the eight programs at the launch width
  on `sm_80` and `sm_86`. The 818 pairs before it did not move.

What was built against what was proposed for step 5:

- The proposal bound a column as the run from `start * D + column` to
  `end * D + column`. The kernel and the oracle use `end * D` as the end,
  which the columns of a thread share. Both name the same elements,
  because a column index is below `D`.
- The proposal did not name the role of the kernel in an artifact. It is
  `column`.
- The proposal listed a tail candidate for the fresh-offsets harness. None
  was added: no record would hold it, and the user guide states the cost
  of a long segment with few columns from the tile itself.
- An offsets refusal of a rank-two call names the number of rows as the
  value count, in the words of the rank-one refusal.
- On the device every kind and both element types agree with
  `torch.segment_reduce` along axis 0 at 1, 3, 64, 129, 200, and 1024
  columns, and equal the CPU oracle bit for bit on values that are not
  exactly summable, as the row order predicts. A column sum stays within
  `(n - 1) * eps * sum(|x|)` of the exactly rounded sum. The bits of a
  column do not change with the block width or with the other segments of
  the batch.
- NVIDIA Compute Sanitizer runs the column kernels with the other kernel
  families and reports nothing, as a kernel without shared memory must.

Step 6. Rank-two softmax, f32. Implemented.

- The compiler did not change. The softmax program over rank-two values,
  `ragged_softmax_r2`, is the rank-one program with the `feature_count`
  role, `swage.segment_id 1` as the column of `swage.make_segment`, and a
  rank-two output of `swage.map_store`. Step 5 admits, plans, and lowers
  it: the direct schedule gives one kernel of `policy<column>`, in which a
  thread runs the maximum, the sum of the exponentials, and the store of
  its column one after the other.
- The kernel holds one maximum, one addition, two `ex2.approx.f32`, and one
  division, and no shuffle, barrier, or shared memory. A thread holds one
  scalar per reduction stage. The store writes the flat index it loaded,
  so the row clamp and the feature-count bound of the column tile bound
  the stores as they bound the loads.
- `swage.segment_softmax` takes `[N, D]` float32 values and returns
  `[N, D]`. It validates offsets that cover every row and classifies
  nothing. `[N, 1]` values run the rank-one kernel through a view, and
  `[N, 0]` values launch nothing. float64 values stay refused. The
  artifact holds one more program, `ragged_softmax_r2`, with the one role
  `column`: eighteen programs and forty-two kernels.
- `test/Conversion/SwageToPlan/ragged-softmax-columns.mlir` and the files
  of the same name under `SwageToCPU`, with a runner file, and `SwageToGPU`
  pin the plan, the oracle, and the kernel.
  `python/tests/mlir/test_segment_columns.py` holds the public contract,
  and `python/tests/mlir/test_segmented_bounds.py` the device bounds.
- The digest matrix gains 2 pairs, the program at the launch width on
  `sm_80` and `sm_86`. The 834 pairs before it did not move.

What was built against what was proposed for step 6:

- The proposal named the softmax module generator, `segment_softmax`, and
  the artifact table as the files of the step. That held: no C++ file
  changed.
- The proposal gave the bound of the softmax page at `k = n - 1`. The
  tests assert it for every output of every column against float64
  `torch.softmax` along the rows of a segment, at 3, 64, 129, 200, and
  1024 columns and for a segment of 100,003 rows. At one column the call
  runs the rank-one kernel and keeps the rank-one `k`.
- The comparison with the CPU oracle is within a tolerance and not
  bitwise, unlike the reductions. Both add a column in row order, and they
  differ in the exponential: `ex2.approx.f32` on the device and `exp2f` on
  the host.
- The private launch admits an output of fewer rows than the values, as
  the rank-one softmax launch does, and passes the smaller row count to
  the kernel. The public call requires the shape of the values.
- A wrong `out` of a rank-two softmax is refused with
  `out must have shape (N, D), the shape of values; found ...`, which the
  proposal did not word.
- On the device the special values follow `torch.softmax` per column: a
  NaN, a positive infinity, or nothing but negative infinities in a column
  of a segment makes that column of that segment NaN and nothing else. The
  bits of a column do not change with the block width or with the other
  segments of the batch.
- NVIDIA Compute Sanitizer runs the kernel with the other kernel families
  and reports nothing.

## Risks and the test that detects each

| Risk | Detected by |
|---|---|
| A new kind falls into the branch of another kind | The switches without a default; `unittests/EmissionTest.cpp`; the `segmented-min.mlir` files; exact comparison with PyTorch |
| An f32 identity in an f64 kernel | The f64 lit files; the verifier after the conversion; `unittests/EmissionTest.cpp`; the typed PTX scan; exact f64 results on values that are not f32 values |
| f64 `exp2` reaches NVPTX and aborts the process | The admission rule; a negative lit case; a compile-only Python test |
| A mean of partial means, or a wrong divisor | The plan check on `arith.divf` in `test/Conversion/SwageToPlan/segmented-mean.mlir`; the bitwise `sum / length` tests on split lengths |
| The extent taken as the end minus the first element of a thread | `test/Conversion/SwageToGPU/segmented-mean.mlir`; the result alone cannot show it, because the thread that stores starts at the start of the segment |
| The merge reads a range record out of bounds | The stray-record case of `python/tests/mlir/test_segmented_bounds.py`, with range records between guards |
| int64 wraps on narrowing | The refusal of `[0, 2**32 + 3, 5]` |
| The private offsets copy is freed early | A lifetime test |
| A column reads a neighbor or writes past `[S, D]` | Values that depend on the row and on the column in `python/tests/mlir/test_segment_columns.py`; the driver-level launches of `python/tests/mlir/test_segmented_bounds.py` with guards around the values and canaries around the output; `test/Conversion/SwageToGPU/segmented-columns.mlir` |
| A thread holds one accumulator per column | The same lit file, which pins one iteration argument of the row loop; `python/tests/mlir/test_segmented_codegen.py` |
| A softmax column stores a row the host did not validate, or a column beyond the feature count | The driver-level launches of the softmax column kernel in `python/tests/mlir/test_segmented_bounds.py`, with canaries around the `[N, D]` output and an output of fewer rows than the values; `test/Conversion/SwageToGPU/ragged-softmax-columns.mlir`, which pins the store at the loaded index |
| The PTX scan passes arithmetic of the wrong type | The typed arithmetic scan |
| An existing kernel's text moves | The 552 digest pairs |
| f64 behaves differently on the device | The GPU tier, from step 3: the `eps64` bound against exactly rounded sums, the exact extremes, and the special values on every schedule |

## Scope

Not part of this record: autograd; f16, bf16, and integer values; `prod`,
weighted sums, and arg-reductions; a lengths argument, a base offset, and
an option for the value of an empty segment; lifting the `2**31 - 1` cap;
the persistent kernel beyond the f32 identity sum; rank three and above; a
split or a row-parallel tile for rank two; libdevice or a math library;
device-side classification; and a public segment syntax.

## Questions decided at acceptance

Each question was answered as recommended.

1. `mean` as a composition, or a kind with a finalize step? A composition.
2. Where does the merge get its extent? From the partial range records.
3. int64 offsets: host narrowing, or an i64 load in the kernels? Host
   narrowing, with the cap kept.
4. A float64 softmax? Refused.
5. Rank two as one column-tile schedule without a split? Yes, with its
   costs in the contract.
6. `swage.segment_id 1` for the column, or a new operation?
   `swage.segment_id 1`.
7. Artifact format version 2, without the unlaunched warp kernel? Yes, in
   step 1.
8. An empty `mean` is NaN? Yes, as `torch.segment_reduce`.
9. `[N, 1]` through the rank-one schedules? Yes.
10. The `(n - 1)` bound for rank two? Accepted and documented.
11. New digest cells on two processors, with f64 `sum` and `max` on every
    admitted processor? Yes.
