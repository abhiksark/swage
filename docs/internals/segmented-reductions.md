<!-- docs/internals/segmented-reductions.md -->

# Segmented Reductions

Canonical segmented sum, max, and min execute through a sequential CPU
oracle and a one-CTA GPU path. This page records the exact internal
contracts; none of them is a public API. The public
`swage.segment_reduce` runs the identity sum, max, and min through the
planned path, over f32 or f64 values, and [Sum rounding](#sum-rounding) says
which schedule it gets.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) for the executable evidence.*

An admitted segment function has one axis-zero segment ID, one segment over
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
