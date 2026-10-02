<!-- docs/internals/segmented-reductions.md -->

# Segmented Reductions

Canonical segmented sum and max execute through a sequential CPU
oracle and a one-CTA GPU path. This page records the exact internal
contracts; none of them is a public API. The public
`swage.segment_reduce` runs the identity sum and max through the planned
path, and [Sum rounding](#sum-rounding) says which schedule it gets.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) for the executable evidence.*

An admitted segment function has one axis-zero segment ID, one segment over
rank-one f32 values and rank-one i32 offsets, one capture-free
reduction of kind `sum` or `max`, with an optional element expression and
single-consumer map chains, one rank-one f32 output, and explicit i32
value and segment counts. A module may hold any number of segment functions
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
loads. Empty sums produce zero; empty maxima produce negative infinity. Max
uses NaN-propagating semantics. The same single-stage programs also support
private warp, fused mixed, and split execution; see [Task Planning](planning.md)
and [Split Execution](split-execution.md).

## Sum rounding

An f32 sum is a fixed tree of IEEE-754 round-to-nearest additions. The
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

The public `swage.segment_reduce` offers none of them. Every call launches
`mixed` of a preparation with the default limits and `select_schedule=True`,
so the bits of a public sum can change with the composition of its batch,
and its bound is the one of the second case under
[Error bound](#error-bound). A test compares the public sum with that
private launch bit for bit, and a second one shows the change between a
batch of one segment fewer than the device has SMs and a batch of as many.

### Error bound

Every schedule returns a sum within

```text
k * eps32 * sum(|x|)
```

of the exact sum of the f32 values it adds, where `eps32` is `2**-23`, `k`
is the entry of the table above, and `x` are the results of the element
program. This is the worst case of a summation tree with `k` rounding
additions on its longest path. The bound is relative to the sum of
magnitudes, not to the sum: when the values cancel, the relative error of
the result is larger by the factor `sum(|x|) / |sum(x)|`.

Under the default limits, the bound of `mixed` depends on whether automatic
selection is on:

- With `select_schedule=False`, `mixed` has `k` of at most 38 for any
  segment up to 2,097,152 elements, which is `4.5e-06 * sum(|x|)`. The
  largest `k` belongs to a 4096-element segment on the CTA schedule.
- With `select_schedule=True`, which is the default of
  `_prepare_planned_reduction`, a batch that meets the selection rule runs
  the CTA schedule on segments of up to 8192 elements. There `k` is at most
  70, which is `8.3e-06 * sum(|x|)`. A batch that does not meet the rule
  has the bound of the first case.

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
elements are read. Max involves no rounding and is compared exactly.

### Special values

A sum follows IEEE-754 addition on every static schedule:

- A NaN element gives NaN.
- An infinity among finite elements gives that infinity.
- Infinities of both signs give NaN.
- Finite elements whose sum exceeds the f32 range give infinity.
- Subnormal elements and results are kept and are not flushed to zero.

All of this section is measured on the RTX A6000 (`sm_86`). The PTX check
also compiles for `sm_80`. No other device has run these tests, so
agreement of the bits between two GPU models is not established.

Continue with [Ragged Softmax](ragged-softmax.md) for the fused
multi-phase case, or [Verification](verification.md) for the oracle
topology behind these claims.
