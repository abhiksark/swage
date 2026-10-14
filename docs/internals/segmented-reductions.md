<!-- docs/internals/segmented-reductions.md -->

# Segmented Reductions

Canonical segmented sum, max, and min execute through a sequential CPU
oracle and a one-CTA GPU path. This page records the exact internal
contracts; none of them is a public API. The public
`swage.segment_reduce` runs the identity sum, max, min, and mean through
the planned path, over f32 or f64 values, and [Sum rounding](#sum-rounding)
says which schedule it gets. A mean is a sum with a division, as
[Mean](#mean) describes.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) for the executable evidence.*

A function over rank-two values is described under
[Rank-two values](#rank-two-values). An admitted segment function over
rank-one values has one axis-zero segment ID, one segment over
rank-one values and rank-one i32 offsets, one capture-free reduction of kind
`sum`, `max`, or `min`, with an optional element expression and
single-consumer map chains, one rank-one output, and explicit i32 value and
segment counts. The values, every region, and the output have one element
type, f32 or f64; [Element types](#element-types) states what differs
between the two. A module may hold any number of segment functions
next to other functions. A lowering turns each segment function into a
kernel of its own, as
[Compiler Tools and Passes](compiler-tools.md#functions-and-symbols)
describes, and the runtime compiles the one it names.

The function declares what each argument is with a `swage.role` argument
attribute, as [Textual Swage IR](../language/swage-ir.md#argument-roles)
describes, and may declare the arguments in any order. A lowering finds an
argument by its role. The kernel it emits takes its parameters in one fixed
order, whatever order the function declared:

```text
values*, offsets*, output*, value_count:i32, segment_count:i32
```

The CPU path lowers to sequential SCF and memref operations and executes with
upstream `mlir-runner`. The GPU path uses one CTA per segment and block-stride
loads. Empty sums produce zero, empty maxima negative infinity, and empty
minima positive infinity. Max and min use NaN-propagating semantics: the
kernels combine with `arith.maximumf` and `arith.minimumf`. The same
single-stage programs also support private warp, fused mixed, and split
execution; see [Task Planning](planning.md) and
[Split Execution](split-execution.md).

## Element types

A segment function is admitted over f32 or over f64 values. The element
type is the one the function declares for its `values` argument. The output,
every region argument and result, and every constant of a region must have
it: a function that mixes the two is rejected, and so is any other element
type. The plan, the task records, and the schedules do not depend on the
element type. The offsets and the counts are i32 for both.

An f64 kernel differs from the f32 kernel of the same program in these
ways:

- **Loads, stores, and scratch.** Every value, partial result, and result
  is eight bytes. The scratch of a split segment has the element type.
- **Warp shuffles.** A shuffle moves 32 bits, so the upstream NVVM
  conversion sends an f64 value as two halves. A warp tree is ten
  `shfl.sync` instructions where the f32 tree has five, and a block
  reduction is forty where the f32 one has twenty. The number of barriers
  is the same.
- **Shared memory.** The block reduction holds one slot for each of up to
  32 warps: 256 bytes where the f32 kernel takes 128.
- **Sum.** Every addition is `add.rn.f64`, and a multiply of an element
  program is `mul.rn.f64`: round to nearest, not contracted.
- **Maximum and minimum.** The f32 kernels use the single instructions
  `max.NaN.f32` and `min.NaN.f32`. The pinned NVPTX backend writes no such
  instruction for f64 and expands each combine into `max.f64` or `min.f64`,
  a NaN test (`setp.nan.f64`), a zero test (`setp.eq.f64`), and four
  `selp.f64`. The result is the IEEE-754 maximum or minimum, as for f32:
  the tests compare it bit for bit on NaN, both infinities, signed zeros,
  and subnormals.

Two programs stay f32 only:

- **`math.exp2`.** The device has an approximate `exp2` instruction for f32
  and none for f64, and the pinned backend aborts on an f64 `exp2` that it
  cannot select. Planning admission therefore refuses `math.exp2` in an f64
  program with a diagnostic, for the GPU paths and for the CPU oracle, so
  the oracle never runs a program that no kernel can. This is why
  `swage.segment_softmax` refuses float64 values.
- **Persistent execution.** The experimental persistent kernel admits the
  f32 identity sum only.

A compile-only test scans the PTX of every f64 kernel with the element type
of the kernel: an instruction of the other width is a violation, as is a
fused multiply-add, another rounding mode, and flush-to-zero.

## Mean

A mean is not a reduction kind. Its program is a `kind<sum>` reduction,
`swage.extent` of the same segment, and one division, which
[Textual Swage IR](../language/swage-ir.md#swageextent) shows. The division
runs once per segment, after the threads of a task combined their sums:

- **Task kernels and the oracle.** The extent is the clamped end of the
  segment minus its clamped start. It is converted to the count type and
  then to the element type, and it divides the combined sum. Every thread
  computes the quotient, and the thread that stores the sum of a sum
  program stores it.
- **Split.** The partial kernel is the partial kernel of the sum: a chunk
  stores its raw sum. The merge sums the partial sums and divides once. It
  reads the extent of its segment from the range records of the partial
  tasks, as [Split Execution](split-execution.md) describes, because its
  bound range is scratch.
- **Persistent execution** refuses a mean.

The results follow from the composition:

- A mean is the sum of the same schedule on the same batch divided by the
  length as a value of the element type, bit for bit. The tests compare it
  with the sum program on every static schedule and on the one-CTA path, in
  both element types, at lengths that include 4097, 100,003, and 1,048,577.
- An empty segment gives NaN, zero divided by zero. The tests compare it
  with `isnan`, because the sign and the payload of that NaN are not
  specified.
- A NaN element gives NaN. A sum that overflows gives an infinite mean,
  also when the exact mean is finite.
- A mean lies within `(k + 1) * eps * sum(|x|) / n` of the exact mean, with
  the `k` and the `eps` of [Error bound](#error-bound) and `n` the length.
  The extra unit covers the division and the conversion of the length,
  which is exact up to 16,777,216 elements in f32 and always exact in f64.
  The bound holds when the sum does not overflow and the mean is not
  subnormal. For the public call `k + 1` is at most 71. The tests compare
  the public mean with the exactly rounded mean, formed in rational
  arithmetic.
- The schedule of a mean is the schedule of its sum: the division adds no
  element work, so the selection rule treats both alike.

All of this is measured on the RTX A6000 (`sm_86`).

## Rank-two values

A segment function over rank-two values reduces one column of one segment
per program instance. It declares a sixth role, `feature_count`, and its
program is the rank-one program with a second segment id:
`swage.segment_id 1` is the column, `swage.make_segment` binds it, and the
result is stored at `output[segment, column]`, as
[Textual Swage IR](../language/swage-ir.md#swagemake_segment) shows. The
segment is a column of the rows of a segment, a run of scalars, so every
reduction kind, both element types, and the mean epilogue apply as they are.

The kernel takes the parameters of the direct kernel and the number of
columns:

```text
values*, offsets*, output*, value_count:i32, segment_count:i32,
feature_count:i32
```

`value_count` is the number of rows. `values` holds `value_count` rows of
`feature_count` elements in row order, and `output` holds `segment_count`
rows of the same width.

The kernel is the column tile:

- **One block per segment.** The block index is the segment, compared with
  `segment_count`. The rows of the segment are loaded from the offsets and
  clamped to `value_count`, as every range into values is.
- **One column per thread at a time.** Thread `t` of a block of 128 runs
  the region for the columns `t`, `t + 128`, and so on, one after the
  other. The loop over those columns is bounded by `feature_count`.
- **A column is a strided run.** For the rows `[start, end)` and the column
  `c` of `D`, a thread reads the elements `start * D + c`,
  `(start + 1) * D + c`, and so on below `end * D`. The index is 64 bits
  wide, so the number of elements may exceed `2**31` while the number of
  rows stays below it.
- **One scalar per reduction.** A thread holds one accumulator per
  reduction stage, never one per column, so no runtime-sized array exists
  on the device or in the IR.
- **No combination across threads.** Each thread stores the results of its
  own columns at `output[segment * D + c]`. The kernel holds no shuffle, no
  barrier, and no shared memory, so its control flow may depend on the
  thread index, and the block reduction rules do not apply to it.
- **Every slot is written.** An empty segment stores the value of the kind
  in each of its columns.

The CPU oracle lowers the same program to a loop over the segments, a loop
over the columns, and a loop over the rows of the column. Both add a column
in row order, so the kernel and the oracle agree bit for bit, also on
values that are not exactly summable, and the tests compare them that way.

What follows from the tile:

- A column sum lies within `(n - 1) * eps * sum(|x|)` of the exact sum of
  its `n` rows, the bound of a sequential sum. It is weaker than the bound
  of the rank-one trees for a long segment. The bits of a column do not
  depend on the batch, the block width, or the GPU model.
- There is no split. A segment occupies one block for its whole length, so
  a batch with a heavy tail of long segments keeps few blocks busy for a
  long time.
- With few columns few threads of a block work: a segment of 10,000 rows
  and three columns is 10,000 additions, one after the other, in each of
  three threads.
- A launch classifies nothing, uploads no task record, and allocates no
  scratch. Its host work is the validation of the offsets.

`swage.segment_reduce` runs this kernel for `[N, D]` values with more than
one column. `[N, 1]` values take the rank-one schedules through a view, and
`[N, 0]` values launch nothing. Planning admission refuses a rank-two
function on every schedule that reads a task buffer.

The alternative, a column loop inside the row tiles of rank one, keeps the
split and the rank-one bound. It was not built: it doubles four schedules,
runs one cross-thread combination per column and segment, and loads a row
apart. [ADR-0022](../adr/ADR-0022-wider-data-model-for-segmented-reductions.md)
records the choice.

The driver-level tests of `python/tests/mlir/test_segmented_bounds.py`
launch the kernel below the Python validation: with row ranges that
validation rejects, with more blocks than segments, and with feature counts
of zero and below. Values sit between NaN guards and the output between
canaries. All of this is measured on the RTX A6000 (`sm_86`).

The softmax over rank-two values runs on the same tile, with three stages
per column in place of one reduction.
[Ragged Softmax](ragged-softmax.md#rank-two-values) describes it.

## Sum rounding

A sum is a fixed tree of IEEE-754 round-to-nearest additions in the
element type of the program, f32 or f64. The trees and the schedules below
are the same for both. The
compiled kernels contain no fused multiply-add, no other rounding mode, and
no flush-to-zero, which a compile-only test checks in the PTX of every sum
kernel. For one schedule the result is therefore a function of the values
of the segment and its length. The same schedule returns the same bits on
every launch, after a new preparation, in a second process, and wherever the
segment sits in the batch.

The tree depends on the schedule. The GPU schedules are the `warp`, `cta`,
and `mixed` callables that the private `_prepare_planned_reduction` helper
returns, described on [Task Planning](planning.md),
[Task Execution](task-execution.md), and
[Split Execution](split-execution.md), and the one-CTA `launch_gpu` path of
this page. Each strides the segment with a fixed number of lanes, where lane
`t` adds elements `t`, `t + lanes`, `t + 2 * lanes`, and so on in order,
and then adds the lanes pairwise.

| Schedule | Used by | Lanes | Additions on the longest path, `k` |
|---|---|---|---|
| Sequential | CPU oracle | 1 | `n - 1` |
| Warp | `warp`, and `mixed` up to the warp limit | 32 | `ceil(n / 32) + 4` |
| CTA | `cta`, `launch_gpu` at its default block size, and `mixed` up to the chunk limit | 128 | `ceil(n / 128) + 6` |
| Split | `mixed` above the chunk limit | 512 per chunk, then 512 over the chunk sums | `24 + ceil(ceil(n / 4096) / 512)` |

`n` is the segment length, and no schedule exceeds `n - 1`. The split row
is for the default 4096-element chunk limit, where `k` is 25 up to 2,097,152
elements.

### What the bits depend on

Segments of at most 32 elements have the same bits under the warp and CTA
schedules, because each lane holds at most one element and both trees then
add the same pairs. Longer segments can round differently under the warp,
CTA, and split trees, and under split trees with different chunk limits.
None of the GPU trees is the sequential order of the CPU oracle, so a GPU
sum and the oracle agree within the bound below and not bit for bit.

The schedule of a segment changes in these cases:

- The caller launches another callable of the prepared reduction: `warp`,
  `cta`, or `mixed`.
- Under `mixed`, the segment length crosses the warp limit (32 elements by
  default) or the chunk limit (4096 elements by default), or the caller
  passes other limits.
- Under `mixed` with `select_schedule=True`, which is the default of
  `_prepare_planned_reduction`, the batch meets the selection rule on
  [Task Planning](planning.md): every segment has 4097 to 8192 elements, the
  element program is small, and the batch has at least as many segments as
  the device has SMs. `mixed` is then the CTA schedule instead of the split
  schedule. The bits of a segment therefore depend on how many other
  segments are in the batch and on the GPU model. On the RTX A6000, which
  has 84 SMs, 83 such segments use the split tree and 84 use the CTA tree.
- The caller passes another block size to `launch_gpu`, which changes the
  number of lanes.
- The caller uses the experimental persistent kernel. It returns the bits
  of `mixed` up to the warp limit and above the chunk limit. Between them it
  strides the segment with its 512-thread block, which none of the prepared
  schedules does.

There is no selectable deterministic mode. These are the ways to pin the
schedule with the private helpers today:

- Launch `cta` or `warp`, which use one tree at every length.
- Pass `select_schedule=False`. `mixed` then chooses the tree from the
  length of each segment and the limits alone.
- Use `_prepare_planned_sum`, which always disables selection.

The public `swage.segment_reduce` offers none of them. Every call enqueues
the `mixed` schedule under the default limits with automatic selection, in a
one-shot step that shares the selection rule with the prepared path, so the
bits of a public sum can change with the composition of its batch, and its
bound is the one of the second case under [Error bound](#error-bound). Tests
compare the public result with the prepared `mixed` launch bit for bit over
the differential batches, and one shows the change between a batch of one
segment fewer than the device has SMs and a batch of as many.

### Error bound

Every schedule returns a sum within

```text
k * eps * sum(|x|)
```

of the exact sum of the values it adds, where `eps` is `eps32`, `2**-23`,
for an f32 program and `eps64`, `2**-52`, for an f64 program, `k` is the
entry of the table above, and `x` are the results of the element program. This is the worst case of a summation tree with `k` rounding
additions on its longest path. The bound is relative to the sum of
magnitudes, not to the sum: when the values cancel, the relative error of
the result is larger by the factor `sum(|x|) / |sum(x)|`.

Under the default limits, the bound of `mixed` depends on whether automatic
selection is on:

- With `select_schedule=False`, `mixed` has `k` of at most 38 for any
  segment up to 2,097,152 elements, which is `4.5e-06 * sum(|x|)` for f32
  and `8.4e-15 * sum(|x|)` for f64. The largest `k` belongs to a
  4096-element segment on the CTA schedule.
- With `select_schedule=True`, which is the default of
  `_prepare_planned_reduction`, a batch that meets the selection rule runs
  the CTA schedule on segments of up to 8192 elements. There `k` is at most
  70, which is `8.3e-06 * sum(|x|)` for f32 and `1.6e-14 * sum(|x|)` for
  f64. A batch that does not meet the rule has the bound of the first case.

A seeded test checks the bound against a float64 reference at 100,003,
300,001, and 1,048,577 elements, on mixed-sign values spread over 80
binades and on values whose sum is below `1e-12` of the sum of their
magnitudes. A second test checks the bound of the selected CTA schedule on
a batch of 8192-element segments that meets the selection rule, and a third
keeps the two `k` limits above equal to the trees the tests use. The
largest error the first test measures on the RTX A6000 is
`0.12 * eps32 * sum(|x|)`, and the test also fails above
`eps32 * sum(|x|)` on those inputs. That second limit is a regression guard
on the measurement, not a guarantee for other values.
The bound admits a dropped element on a long segment, so it is an accuracy
statement. The exact tests on position-dependent values check which
elements are read. Max and min involve no rounding and are compared exactly.

For f64 a float64 reference is not enough: summed in another order it
carries rounding of the size being bounded. The f64 tests compare with the
exactly rounded sum, `math.fsum`:

- On every static schedule and on the one-CTA path, at segment lengths from
  0 to 65,537 on both sides of the warp and chunk limits, over values
  spread across sixteen binades. The largest error measured on the RTX
  A6000 is `0.35 * eps64 * sum(|x|)`, and the test also fails above
  `eps64 * sum(|x|)`, the same regression guard as for f32.
- On the public call, over the differential batches and at 100,003 and
  1,048,577 elements.
- On values that are exactly summable in f64 and are not f32 values, where
  every schedule must return the exact sum. A kernel that loaded,
  accumulated, or stored in f32 fails that test.

The f64 sums of one batch differ between the schedules, as the f32 sums do,
and each schedule repeats its bits between launches. The sequential CPU
oracle returns the left-to-right float64 sum bit for bit.

### Special values

A sum follows IEEE-754 addition on every static schedule:

- A NaN element gives NaN.
- An infinity among finite elements gives that infinity.
- Infinities of both signs give NaN.
- Finite elements whose sum exceeds the range of the element type give
  infinity.
- Subnormal elements and results are kept and are not flushed to zero.

The f64 tests use the values of f64: the largest finite f64, the smallest
normal, `2**-1022`, and the smallest subnormal, `2**-1074`.

All of this section is measured on the RTX A6000 (`sm_86`). The PTX check
also compiles for `sm_80`. No other device has run these tests, so
agreement of the bits between two GPU models is not established.

Continue with [Ragged Softmax](ragged-softmax.md) for the fused
multi-phase case, or [Verification](verification.md) for the oracle
topology behind these claims.
