<!-- docs/user-guide/segmented-calls.md -->

# Segmented Calls

Two functions run a fixed program over every segment of a ragged batch:

- `swage.segment_reduce(values, offsets, kind, *, out=None)` returns one
  result per segment, in the dtype of `values`: a sum, a maximum, a
  minimum, or a mean of float32 or float64 values. The values are a run of
  scalars or `[N, D]` rows of `D` features, which are reduced per column.
- `swage.segment_softmax(values, offsets, *, out=None)` returns one float32
  result per value, the softmax within its segment. `[N, D]` rows are
  normalized per column.

These two calls are the whole public segmented surface. The programs are
fixed. There is no public segment syntax, so an element expression, another
reduction kind, a dtype outside the admitted ones, and values of rank three
or above cannot be written. The calls record no gradient.
[Ragged Data](ragged-data.md) defines the storage they read. This page
shows a call, states what it returns and what it costs, and lists where it
is refused.

Both calls need the CUDA GPU tier: the native wheel or a source build,
PyTorch 2.6 or newer, `numpy`, which the `pytorch` extra of the package
declares, and an NVIDIA GPU. The released `0.5.1` wheel has neither call;
[Installation](../getting-started/installation.md) lists what it lacks. The
[Support Matrix](../reference/support-matrix.md) lists the versions and the
GPU the tests run on. An artifact directory that `python -m swage.compile`
wrote ahead of time can take the place of the native bindings;
[Running Without the Compiler](deployment.md) describes that.

## A first call

```python
import swage
import torch

# Six values in four segments: [1, 2], [], [3, 4, 5], and [6].
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int32, device="cuda")

totals = swage.segment_reduce(values, offsets, "sum")  # [3, 0, 12, 6]
maxima = swage.segment_reduce(values, offsets, "max")  # [2, -inf, 5, 6]
minima = swage.segment_reduce(values, offsets, "min")  # [1, inf, 3, 6]
means = swage.segment_reduce(values, offsets, "mean")  # [1.5, nan, 4, 6]
weights = swage.segment_softmax(values, offsets)       # six weights
```

The committed script `examples/segment_reduce.py` runs these calls and
compares each result with PyTorch. Run it with the installed wheel, or from
a source build with `PYTHONPATH=build/python_packages` in front:

```bash
python examples/segment_reduce.py
```

A call returns after it enqueued its kernels on the current PyTorch CUDA
stream. It does not wait for them. Reading a result, as `tolist()` or
`cpu()` does, waits in the usual PyTorch way.

## Arguments

`values` and `offsets` follow
[the offsets contract](ragged-data.md#the-offsets-contract): contiguous
values and rank-one `torch.int32` or `torch.int64` offsets on the current
CUDA device. No values are cast, and nothing is moved to another device or
repaired. int64 offsets are checked and then narrowed on the host, as
[int64 offsets](ragged-data.md#int64-offsets) describes, so offsets that
PyTorch produced as int64 need no cast first. An argument outside the
contract raises before anything is enqueued.

The two calls differ in the values they take:

- `segment_reduce` takes `torch.float32` or `torch.float64` values. A
  float64 batch runs a float64 program: every value is loaded, combined,
  and stored as float64, and the result is float64. Any other dtype raises
  a `TypeError` that names the two.
- `segment_softmax` takes `torch.float32` values only. float64 values
  raise a `TypeError` that states the reason: the softmax kernel computes
  its exponential with the `exp2` instruction of the device, which exists
  for 32-bit values and not for 64-bit ones.

Both calls take values of rank one or of rank two.
[Rows of features](#rows-of-features) describes rank two. Any other rank
raises a `TypeError` that names the two.

They also differ in one offsets rule:

- `segment_reduce` admits offsets that end below the number of values, as
  `torch.segment_reduce` does. The values past the final offset belong to no
  segment and reach no result.
- `segment_softmax` requires offsets that end at the number of values, or
  of rows for `[N, D]` values. Its result has one element per value, and an
  element that no segment covers would be returned unwritten.

`out` is optional and keyword-only. When it is given, the call writes it and
returns the same tensor. It must meet all of these rules:

- It is a contiguous tensor of the dtype of `values`, on the device of
  `values`. A contiguous slice of a larger tensor is admitted.
- It has rank one and exactly one element per segment for `segment_reduce`
  of rank-one values, and exactly one element per value for
  `segment_softmax`. For `[N, D]` values it has the shape `[S, D]` for
  `segment_reduce`, one row per segment and one column per feature, and
  the shape `[N, D]` of the values for `segment_softmax`. It is never
  resized.
- It shares no memory with `values` or `offsets`.
- It does not require grad, and it is not a lazy negation or conjugate view.

Without `out`, the call allocates the result on the device of `values`.

After a call that enqueued a kernel, the version counter of the result has
advanced, as after an in-place PyTorch operation. A backward pass that saved
`out` before the call therefore raises instead of using the new values. A
call that enqueues nothing, because it is refused or its batch has no
segment, leaves the counter of `out` alone. The counters of `values` and
`offsets` do not move.

## Results

The results below are pinned by tests on the GPU. `torch.segment_reduce`
returns the same value in every case on PyTorch 2.12, the version the GPU
tests run with.

A sum follows IEEE-754 addition:

- An empty segment gives `0.0`.
- A NaN element gives NaN.
- One infinity among finite elements gives that infinity.
- Infinities of both signs give NaN.
- Finite elements whose sum exceeds the range of the dtype give positive
  infinity.
- Subnormal elements and results are kept and are not flushed to zero.

A maximum propagates NaN:

- An empty segment gives negative infinity.
- A NaN element gives NaN, wherever it sits in the segment.
- A positive infinity gives positive infinity, also beside a negative one.
- A negative infinity among finite elements gives the largest finite
  element.

A minimum is its mirror:

- An empty segment gives positive infinity.
- A NaN element gives NaN, wherever it sits in the segment.
- A negative infinity gives negative infinity, also beside a positive one.
- A positive infinity among finite elements gives the smallest finite
  element.

A mean is the sum divided by the length of the segment:

- An empty segment gives NaN, zero divided by zero. A mean has no identity
  to return instead.
- A NaN element gives NaN, and an infinity gives what its sum gives.
- A sum that overflows gives an infinite mean, also when the exact mean of
  the values is finite.
- The length is converted to the dtype of `values` once and divides the sum
  once. A float32 length above 16,777,216 is rounded to nearest by that
  conversion.

A batch with no segment, whose offsets are the single entry `0`, returns a
tensor of no elements and needs no kernel.
[Ragged Data](ragged-data.md#empty-segments-and-nan) shows how to replace
the result of empty segments after the call.

For `segment_softmax`, the tests compare with `torch.softmax` of each
segment:

- An empty segment has no result element.
- A segment that holds a NaN or a positive infinity, or nothing but negative
  infinities, gives NaN for every element of that segment.
- A negative infinity beside a finite maximum gives exactly `0.0` for that
  element.
- No segment affects the results of another.

A maximum and a minimum involve no rounding and are exact. The other
results are rounded:

- A sum lies within `k * eps * sum(|x|)` of the exact sum of its segment,
  where `eps` is `2**-23` for float32 values and `2**-52` for float64
  values, and `k` depends on the schedule and not on the dtype. For these
  calls `k` is at most 70 for a segment of up to 2,097,152 elements, which
  is `8.3e-06 * sum(|x|)` for float32 and `1.6e-14 * sum(|x|)` for float64.
  The bound is relative to the sum of magnitudes, not to the sum.
- A mean lies within `(k + 1) * eps * sum(|x|) / n` of the exact mean of a
  segment of `n` elements, with the `k` of its sum, so `k + 1` is at most
  71. The extra unit covers the division and the conversion of the length.
  The bound holds when the sum does not overflow and the mean is not
  subnormal. A mean is the sum of the same call on the same batch divided
  by the length, bit for bit.
- A softmax output has a relative error bound that grows with the distance
  of its logit below the segment maximum.
  [Ragged Softmax](../internals/ragged-softmax.md#accuracy) states it, for
  scalars and for rows.

## Sum rounding

The bits of a sum depend on the order of the additions, for float32 and
for float64 values, and a call selects that order from the batch it is
given. The dtype has no part in the selection. A mean adds in the order of
the sum, so everything in this section holds for it too:

- A segment of at most 32 elements is added by a 32-lane tree.
- A segment of 33 to 4096 elements is added by a 128-lane tree.
- A longer segment is split into 4096-element chunks whose sums are added by
  a second tree.
- One rule replaces the split tree by the 128-lane tree: every segment of
  the batch has 4097 to 8192 elements, and the batch has at least as many
  segments as the device has SMs.

The last rule makes the bits of a segment depend on the other segments of
its batch and on the GPU model. On the RTX A6000, which has 84 SMs, a test
batch of 83 segments of 8192 elements changes the bits of its sums when an
84th such segment joins it.

The public call has no argument that pins the schedule. A call on the same
batch returns the same bits each time, and every schedule stays inside the
bound above. A caller who needs bits that do not depend on the batch cannot
get them from these calls today.
[Sum rounding](../internals/segmented-reductions.md#sum-rounding) states the
trees, the bound, and its evidence.

A softmax uses one 128-thread CTA per segment at every length. It has no
warp, split, or selected schedule.

## Rows of features

Both calls take `[N, D]` values: `N` rows of `D` features. The offsets
delimit rows, and every column of a segment is taken on its own.

`segment_reduce` reduces each column, as
`torch.segment_reduce(values, kind, axis=0, ...)` does. Its offsets start
at zero and end at or below `N`, and the result is `[S, D]` for `S`
segments.

```python
# Four rows of two features in three segments: rows 0 and 1, none, 2 and 3.
rows = torch.tensor(
    [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], device="cuda"
)
offsets = torch.tensor([0, 2, 2, 4], dtype=torch.int32, device="cuda")

totals = swage.segment_reduce(rows, offsets, "sum")
# [[3, 30], [0, 0], [7, 70]]
```

`segment_softmax` normalizes each column over the rows of its segment, as
`torch.softmax(values[a:b], dim=0)` does for the rows `a` to `b` of a
segment. Its offsets start at zero and end at `N`, and the result is
`[N, D]`, the shape of the values. The values are float32, as for
scalars.

```python
weights = swage.segment_softmax(rows, offsets)
# Rows 0 and 1 share a segment, so each of their columns sums to one:
# [[0.2689, 0.0000], [0.7311, 1.0000], [0.2689, 0.0000], [0.7311, 1.0000]]
```

Everything under [Arguments](#arguments) and [Results](#results) holds per
column. Every column of an empty segment receives the value of the kind
from a reduction: `0.0`, an infinity, or NaN for a mean. A softmax has no
result row for an empty segment. The values must be contiguous in row
order: a transposed tensor and a slice of columns are refused, and nothing
is copied. Values of rank three or above are refused.

A call on rows runs another kernel than a call on scalars, with one
schedule:

- One block of 128 threads per segment. Thread `t` takes the columns `t`,
  `t + 128`, and so on, one after the other, each in row order.
- No split. A segment occupies one block for its whole length, whatever
  that length is.
- No thread combines with another, so a column is reduced, or normalized,
  by one thread. A softmax thread walks the rows of its column three
  times: for the maximum, for the sum of the exponentials, and for the
  results it stores.

That schedule has these consequences, which are limits of this call and not
of the data:

| | Few columns (1 to 8) | Many columns (64 to 1024) |
|---|---|---|
| Threads of a block that work | `D` of 128 | 64 to 128 |
| Additions of one thread per segment | `n` rows | `n * ceil(D / 128)` |
| Weakness | A long segment is taken by `D` threads: 10,000 rows of three columns are 10,000 additions, one after the other, in each of three threads | A long segment occupies one block for its whole length |

- **Rounding.** A column sum is added in row order, one addition per row.
  It lies within `(n - 1) * eps * sum(|x|)` of the exact sum of its `n`
  rows, with the `eps` of the dtype. For a long segment that bound is
  weaker than the bound of rank-one values, whose trees add in parallel. A
  column mean divides that sum once and lies within `eps * sum(|x|)` of the
  exact mean, under the conditions of the rank-one mean. A maximum and a
  minimum are exact. A softmax of rows has the bound of
  [Ragged Softmax](../internals/ragged-softmax.md#accuracy) with
  `k = n - 1`, the additions of its normalizer in row order.
- **Bits.** The bits of a column do not depend on the other segments of the
  batch, on the block width, or on the GPU model: a rank-two call has no
  schedule selection.
- **Host work.** A call on rows validates its offsets on the host and
  classifies nothing. It uploads no task record and allocates no scratch.
- **One column and no column.** `[N, 1]` values are a run of scalars. They
  are reduced by the schedules of rank-one values, with their rounding and
  their selection, and the result is `[S, 1]`. A softmax of `[N, 1]`
  values runs the kernel of rank-one values and returns `[N, 1]`. `[N, 0]`
  values return an `[S, 0]` result from a reduction and an `[N, 0]` result
  from a softmax, and launch nothing.

The schedules of rank-one values take a long run of scalars in parallel,
and those of a reduction split it. A caller reaches them for rows by
passing each column as a contiguous `[N]` or `[N, 1]` tensor, at the price
of one call per column.

## What a call costs

Each call prepares before it launches, and it keeps nothing of that
preparation for the next call:

1. It copies the offsets to the host. The copy waits for the work already
   queued on the current stream.
2. It validates the offsets on the host and, for `segment_reduce` of
   rank-one values, classifies every segment into warp, CTA, and split
   tasks.
3. For `segment_reduce` of rank-one values, it uploads the task records and
   allocates scratch for split segments. With int64 offsets it also uploads the narrowed
   int32 copy the kernels read.
4. It enqueues the kernels.

A second call with the same offsets tensor repeats all four steps. The task
records and the scratch are released when the call returns.

`torch.segment_reduce` does none of the host work. When the offsets change on
every call, expect `segment_reduce` to be slower than `torch.segment_reduce`.
One committed record measures that regime, on one NVIDIA RTX A6000 at
revision `453c56e`:

--8<-- "docs/internals/_generated/segmented-sum-a6000-sm86-453c56e-fresh-statement.inc"

The measured candidate is not this call. It is a private preparation of
three scheduling policies with schedule selection disabled, followed by the
mixed launch into a caller's output. `segment_reduce` validates and
classifies in the same way, prepares only the schedule it launches, selects
the schedule automatically, and allocates its result when no `out` is
passed. The harness has since gained a candidate that times the call
itself, and no committed record holds it. Read the figures as a measurement
of that private preparation, not of the call.
[Benchmarks](../internals/benchmarks.md#fresh-offsets-and-the-frozen-comparison-at-453c56e)
reports the record and its limits: one GPU, one seed per distribution, and a
machine that was not quiet.

The other recorded comparisons on that page were taken with a private
prepared launch, which prepares one layout once and launches it many times.
That path is not public, and its numbers do not describe these calls.

The first calls of a process cost more:

- A call compiles each kernel its batch needs that the process does not
  hold yet, and loads it into the CUDA context. A reduction kind has four
  kernels per dtype that a batch of rank-one values can need: one for
  segments of up to 4096 elements, two for longer segments, and one for a
  batch that the selection rule of [Sum rounding](#sum-rounding) sends to
  the 128-lane tree. A reduction kind has one more kernel per dtype for
  rows of features. The softmax has one kernel for scalars and one for
  rows. Kernels stay in the process for later calls and are never written
  to the persistent cache. With an artifact selected, a call
  compiles nothing: the first call reads and verifies the directory, and
  each kernel is loaded from it when a call first needs it.
- The first `segment_reduce` call on a device whose batch meets that
  selection rule uploads a table of segment ids that holds 4 MiB of device
  memory for the life of the process.

## Where a call is refused

Every refusal below raises before anything is enqueued.

- **Gradients.** `values` that require grad raise a `ValueError` that names
  `values.detach()`. The calls have no backward function, and a result is
  never part of an autograd graph.
- **CUDA graph capture.** A call on a stream that is capturing a CUDA graph
  raises a `RuntimeError`. Every call copies its offsets to the host, which
  a capturing stream cannot do, and a replay would not repeat the
  preparation. The check comes first, so the capture stays usable for the
  PyTorch work around the call.
- **A missing `numpy`.** A call raises a `BackendUnavailableError`, a
  `RuntimeError`, with the code `numpy-unavailable`, that names `numpy` and
  the installation page. The calls copy the offsets into a `numpy` array on
  the host, with the native bindings and with an artifact.
- **`SWAGE_NO_COMPILE=1`.** A call whose kernels the process does not hold
  raises a `RuntimeError`, because the segmented kernels are not in the
  persistent cache. A process that starts with the switch set cannot run a
  segmented call that has work to do, unless an artifact is selected: a
  kernel taken from an artifact is not compiled. A batch without segments
  needs no kernel and returns.
- **No native bindings and no artifact.** In a frontend-only install, or
  any process that cannot import `mlir_swage`, a call without a selected
  artifact raises a `BackendUnavailableError` with the code
  `native-unavailable` that names the installation page, after the
  argument checks that need no bindings.
- **An artifact that cannot serve the call.** With `SWAGE_ARTIFACT_DIR`
  set, a call raises a `RuntimeError` when the directory is damaged, unsafe,
  or written for another target, and when it does not hold the program of
  the call. It does not compile instead.
  [Running Without the Compiler](deployment.md#refusals) lists the cases.
- **PyTorch older than 2.6.** A call raises the `BackendUnavailableError`
  of `launch()`, with the code `pytorch-unsupported`, before it looks at an
  argument.

Three more rules follow from how PyTorch handles streams, threads, and
inference mode:

- A call enqueues on the stream that is current when it is made. Inputs
  that were produced on another stream must be complete before the call, as
  for any PyTorch operation that crosses streams.
- A call works on a thread that has not used CUDA before.
- A call works inside `torch.inference_mode()`, and `values`, `offsets`,
  and `out` may be tensors that were created inside it. A call enqueues
  what it classified before it returns, so it has no need to detect a later
  change to the offsets.

Continue with [Running Without the Compiler](deployment.md), which compiles
the kernels of these two calls ahead of time and serves the calls from the
result, in a process that holds no compiler.
