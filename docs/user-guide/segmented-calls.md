<!-- docs/user-guide/segmented-calls.md -->

# Segmented Calls

Two functions run a fixed program over every segment of a ragged batch:

- `swage.segment_reduce(values, offsets, kind, *, out=None)` returns one
  f32 result per segment, a sum or a maximum.
- `swage.segment_softmax(values, offsets, *, out=None)` returns one f32
  result per value, the softmax within its segment.

These two calls are the whole public segmented surface. The programs are
fixed. There is no public segment syntax, so an element expression, another
reduction kind, another dtype, and a trailing feature dimension cannot be
written. The calls record no gradient. [Ragged Data](ragged-data.md) defines
the storage they read. This page shows a call, states what it returns and
what it costs, and lists where it is refused.

Both calls need the CUDA GPU tier: the native build, PyTorch 2.6 or newer,
`numpy`, which the binding requirements in
[Installation](../getting-started/installation.md) include, and an NVIDIA
GPU. They are not part of the released `0.5.1` wheel. The
[Support Matrix](../reference/support-matrix.md) lists the versions and the
GPU the tests run on.

## A first call

```python
import swage
import torch

# Six values in four segments: [1, 2], [], [3, 4, 5], and [6].
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int32, device="cuda")

totals = swage.segment_reduce(values, offsets, "sum")  # [3, 0, 12, 6]
maxima = swage.segment_reduce(values, offsets, "max")  # [2, -inf, 5, 6]
weights = swage.segment_softmax(values, offsets)       # six weights
```

The committed script `examples/segment_reduce.py` runs these calls and
compares each result with PyTorch:

```bash
PYTHONPATH=build/python_packages python examples/segment_reduce.py
```

A call returns after it enqueued its kernels on the current PyTorch CUDA
stream. It does not wait for them. Reading a result, as `tolist()` or
`cpu()` does, waits in the usual PyTorch way.

## Arguments

`values` and `offsets` follow
[the offsets contract](ragged-data.md#the-offsets-contract): rank-one,
contiguous, `torch.float32` values and `torch.int32` offsets on the current
CUDA device. Nothing is cast, moved to another device, or repaired. An
argument outside the contract raises before anything is enqueued.

The two calls differ in one offsets rule:

- `segment_reduce` admits offsets that end below the number of values, as
  `torch.segment_reduce` does. The values past the final offset belong to no
  segment and reach no result.
- `segment_softmax` requires offsets that end at the number of values. Its
  result has one element per value, and an element that no segment covers
  would be returned unwritten.

`out` is optional and keyword-only. When it is given, the call writes it and
returns the same tensor. It must meet all of these rules:

- It is a contiguous rank-one `torch.float32` tensor on the device of
  `values`. A contiguous slice of a larger tensor is admitted.
- It has exactly one element per segment for `segment_reduce`, and exactly
  one element per value for `segment_softmax`. It is never resized.
- It shares no memory with `values` or `offsets`.
- It does not require grad, and it is not a lazy negation or conjugate view.

Without `out`, the call allocates the result on the device of `values`.

After a call that enqueued a kernel, the version counter of the result has
advanced, as after an in-place PyTorch operation. A backward pass that saved
`out` before the call therefore raises instead of using the new values. The
counters of `values` and `offsets` do not move.

## Results

The results below are pinned by tests on the GPU. `torch.segment_reduce`
returns the same value in every case on PyTorch 2.12, the version the GPU
tests run with.

A sum follows IEEE-754 addition:

- An empty segment gives `0.0`.
- A NaN element gives NaN.
- One infinity among finite elements gives that infinity.
- Infinities of both signs give NaN.
- Finite elements whose sum exceeds the f32 range give positive infinity.
- Subnormal elements and results are kept and are not flushed to zero.

A maximum propagates NaN:

- An empty segment gives negative infinity.
- A NaN element gives NaN, wherever it sits in the segment.
- A positive infinity gives positive infinity, also beside a negative one.
- A negative infinity among finite elements gives the largest finite
  element.

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

A maximum involves no rounding and is exact. The other two results are
rounded:

- A sum lies within `k * eps32 * sum(|x|)` of the exact sum of its segment,
  where `eps32` is `2**-23` and `k` depends on the schedule. For these calls
  `k` is at most 70 for a segment of up to 2,097,152 elements, which is
  `8.3e-06 * sum(|x|)`. The bound is relative to the sum of magnitudes, not
  to the sum.
- A softmax output has a relative error bound that grows with the distance
  of its logit below the segment maximum.
  [Ragged Softmax](../internals/ragged-softmax.md#accuracy) states it.

## Sum rounding

The bits of an f32 sum depend on the order of the additions, and a call
selects that order from the batch it is given:

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

## What a call costs

Each call prepares before it launches, and it keeps nothing of that
preparation for the next call:

1. It copies the offsets to the host. The copy waits for the work already
   queued on the current stream.
2. It validates the offsets on the host and, for `segment_reduce`,
   classifies every segment into warp, CTA, and split tasks.
3. For `segment_reduce`, it uploads the task records, allocates scratch for
   split segments, and records one CUDA event.
4. It enqueues the kernels.

A second call with the same offsets tensor repeats all four steps. The task
records, the scratch, and the event are released when the call returns.

`torch.segment_reduce` does none of the host work. When the offsets change on
every call, expect `segment_reduce` to be slower than `torch.segment_reduce`.
[Benchmarks](../internals/benchmarks.md#harness-methods) describes the
harness that times this regime. No committed record of it exists yet, so
this page states no number.

The recorded comparisons on that page were taken with a private prepared
launch, which prepares one layout once and launches it many times. That path
is not public, and its numbers do not describe these calls.

The first calls of a process cost more:

- A call compiles each kernel it needs that the process does not hold yet,
  and loads it into the CUDA context. A reduction kind has up to five
  kernels and the softmax has one. Kernels stay in the process for later
  calls and are never written to the persistent cache.
- The first `segment_reduce` call on a device uploads a table of segment
  ids that holds 4 MiB of device memory for the life of the process.

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
- **Offsets made under `torch.inference_mode()`.** `segment_reduce` raises a
  `ValueError` for offsets that are an inference tensor. Create the offsets
  before entering the context, or clone them outside it. The call itself
  runs inside the context, and `values` and the result may be inference
  tensors. `segment_softmax` also takes offsets made inside the context.
- **`SWAGE_NO_COMPILE=1`.** A call whose kernels the process does not hold
  raises a `RuntimeError`, because the segmented kernels are not in the
  persistent cache. A process that starts with the switch set cannot run a
  segmented call that has work to do. A batch without segments needs no
  kernel and returns.
- **A wheel-only install.** A call raises a `RuntimeError` that names the
  installation page, after the argument checks that need no native build.
- **PyTorch older than 2.6.** A call raises the `RuntimeError` of
  `launch()`, before it looks at an argument.

Two more rules follow from how PyTorch handles streams and threads:

- A call enqueues on the stream that is current when it is made. Inputs
  that were produced on another stream must be complete before the call, as
  for any PyTorch operation that crosses streams.
- A call works on a thread that has not used CUDA before.

Continue with [Writing Kernels](writing-kernels.md). That page turns to the
kernel language, which has no segment syntax: the one kernel it accepts is a
fixed-block vector add.
