<!-- docs/adr/ADR-0024-gradients-of-the-segmented-calls.md -->
# ADR-0024: Gradients of the segmented calls

- Status: accepted; step 1 of the migration sequence is implemented
- Date: 2026-10-03
- Accepted: 2026-10-03, with the recommended answer to every question at the
  end, except question 6, which the owner answered differently

This record decides how `swage.segment_reduce` and `swage.segment_softmax`
record a gradient for `values` that require grad: the mechanism, the
gradient of each call, how it is computed, and what stays unsupported.
"Migration sequence" says which steps exist.

## Context

The two calls return results that autograd does not see. They refuse
`values` that require grad with a `ValueError`, so a result is never cut
from a graph without an error, but a model that pools features with them
cannot train through the pooling. Reviewers asked for a registered backward
for the four reduction kinds and the softmax, checked with
`torch.autograd.gradcheck` in float64, compared with the gradients of
`torch.segment_reduce`, and run on a GPU before it is relied on.

ADR-0022 named autograd as out of scope. Since then each call body has been
wrapped by `torch.compiler.disable` at its first call, so a call is a graph
break inside `torch.compile` and runs eagerly.

Probes on the CPU with PyTorch 2.12.0 found these conventions, which the
decision below has to choose among:

- The backward of `torch.segment_reduce` for `max` and `min` divides the
  gradient among the tied elements only when it is positive. With a
  negative or NaN gradient, every tied element receives the whole of it.
  That is not linear in the upstream gradient, so it is not a
  vector-Jacobian product.
- `Tensor.scatter_reduce` with `"amax"` or `"amin"` shares equally for
  either sign, but counts an element of its destination that equals the
  result as a tie, and gives NaN to every element of a segment whose result
  is NaN.
- `Tensor.max()` without a dimension and `torch.amax` share ties equally.
- The `sum` and `mean` backward of `torch.segment_reduce` is a broadcast of
  the gradient, divided by the length for `mean`.
- A prototype of this design passes `gradcheck` and `gradgradcheck` in
  float64 for every kind and for the softmax, at ranks one and two, with
  int32 and int64 offsets, empty segments, and rows past the final offset.
  The output of an empty segment must be masked first: it is a constant
  NaN or infinity, whose numerical Jacobian column is NaN.

## Decision

### Mechanism

The gradient is recorded by `torch.autograd.Function` subclasses in a
private module, `swage._autograd`:

- `SegmentReduce` and `SegmentSoftmax` are the Functions of the two calls.
  Their nodes are named `SegmentReduceBackward` and
  `SegmentSoftmaxBackward`.
- `_SegmentSum` sums per segment with the sum kernel of `segment_reduce`,
  and `_SegmentBroadcast` copies a per-segment tensor to the rows of its
  segments. Each is the backward of the other.

The classes are made at the first call that records a gradient, after the
PyTorch check, and kept per PyTorch module. `import swage` loads no
PyTorch. The Functions are applied inside the call bodies that
`torch.compiler.disable` wraps, so Dynamo never traces them, and their
forwards call the shared launch helpers directly, never a public call. No
`torch.library` operator is registered.

A call records a gradient when `values` require grad and gradient recording
is on. Under `torch.no_grad()` and inside `torch.inference_mode()` a call
records nothing, also for `values` that require grad.

### The gradient of each call

For the gradient `g` of segment `j`, and per column for `[N, D]` values:

- `sum`: every element of the segment receives `g`, unchanged.
- `mean`: every element receives `g` divided once by the length of the
  segment, converted to the dtype of `values` as the forward converts it.
- `max` and `min`: the elements that equal the result share `g` equally:
  each receives `g` divided once by their number. `-0.0` and `0.0` are
  equal. When the result is NaN, the NaN elements share `g`. Every other
  element receives exactly `0.0`, also when `g` is infinite or NaN.
- An empty segment has no element, and its `g` reaches nothing. A row past
  the final offset receives `0.0`.
- `segment_softmax`: an element with result `y` receives `y * (g - s)`,
  where `s` is the sum of `g * y` over its segment.

The equal share is linear in `g`, symmetric, needs no index that the
forward does not have, and is a valid subgradient. It equals the gradient
of `torch.segment_reduce` wherever that gradient is linear.

### Computation

- PyTorch operations do the index work, the gather, and the elementwise
  arithmetic. The segment of every row comes from `torch.searchsorted` on
  the offsets on the device, the gather is `index_select` from the
  per-segment tensor with one appended zero row, and the tie count is a
  difference of an integer `cumsum`. The backward of a reduction copies
  nothing to the host.
- The sum kernel of `segment_reduce` does every floating-point segment sum:
  the `s` of the softmax, and the reduction inside every second derivative.
  It is deterministic for a batch and carries the bound of "Sum rounding".
  Like a forward call, it copies the offsets to the host.
- No new kernel, no artifact change, and no digest change. The sum programs
  are in the kernel table of every artifact.
- No index of an `argmax` is saved: the forward extreme is exact, so
  `values == result` recovers the tied set in the backward.

Nothing chooses between PyTorch and Swage at run time. The forward always
runs Swage kernels, the split of the backward is fixed and documented, and
a backward that cannot run its sum kernel raises.

### Second derivatives

They are supported. `_SegmentSum` and `_SegmentBroadcast` are each other's
adjoint, so every first-order backward is a composition of differentiable
pieces, and no second derivative uses an atomic.

### Interactions

- A call that records a gradient refuses `out` with a `ValueError`, before
  the bindings check. Under `torch.no_grad()` an `out` is accepted with
  `values` that require grad. An `out` that requires grad stays refused.
- The backward keeps the offsets for every call, the values and the result
  for `max` and `min`, and the result for the softmax. An in-place change to
  one of them after the call makes the backward raise, as PyTorch does.
  Offsets created inside `torch.inference_mode()` are kept as a copy.
- A backward of `segment_softmax`, and every second derivative, run the sum
  kernel, which refuses a capturing stream before its host copy with a
  message that names the backward.
- The backward of a reduction reads int32 or int64 offsets on the device as
  they are. The sum kernel narrows int64 offsets as a forward call does.
- `torch.compile`: a call that records a gradient is the same graph break,
  and its backward runs eagerly in the autograd engine. `fullgraph=True`
  stays refused, and compiled autograd is not supported.

## Alternatives considered

- **`torch.library.custom_op` with `register_autograd`.** It would make
  `fullgraph=True` work and let inductor fuse the backward. It replaces the
  graph break of the call bodies, moves every check that reads data into
  the operator, needs lazy registration, needs a second operator per call
  for `out`, and hides the host copy of the offsets inside a compiled graph.
  It is a separate later decision. The backward bodies are plain functions
  that would move into `register_autograd` unchanged.
- **A straight-through composition** with a PyTorch reference. It runs the
  forward twice, and its gradient is PyTorch's: a second backend behind the
  call.
- **PyTorch's sign-dependent tie rule, or one element per tie.** The first
  is not linear in `g`; the second needs an index the forward does not
  produce.
- **`torch.segment_reduce` for the floating-point sums of the backward.**
  Its numerics are not stated by this project and can change between
  PyTorch releases, and the backward would depend on the operation the
  calls are compared against.
- **New Swage kernels for the broadcast or an arg-reduction.** A broadcast
  needs a per-segment load inside a region, which ADR-0008 does not have,
  and every Swage call copies its offsets to the host. The PyTorch gather
  copies nothing.
- **`once_differentiable`.** It gives a clear error where the design gives
  a feature.

## Consequences

- Training through the pooling works for every kind and for the softmax,
  with first and second derivatives.
- The backward of a reduction does not depend on the batch, the schedule,
  or the GPU. The backward of a softmax follows the bits of the sum kernel.
- `max` and `min` keep their values and result alive until the backward
  runs, and the softmax keeps its result.
- The gradient on a tie with a negative or NaN `g` differs from that of
  `torch.segment_reduce`. The user guide states the difference.
- Not supported: an `out` while recording, forward-mode differentiation,
  `torch.func` transforms, compiled autograd, `fullgraph=True` and
  `torch.export`, gradients under CUDA graph capture, a float64 softmax
  gradient (there is no float64 softmax), a choice of tie rule, gradients
  for offsets, and a benchmark record of backward cost.

## Migration sequence

Each step keeps every kernel and digest, passes the test tiers, and lands
its documentation.

1. `sum` and `mean` for both dtypes and both ranks: the launch helpers, the
   private module with its four Functions, the `out` rule, and the capture
   naming. `max`, `min`, and the softmax refuse `values` that require grad
   with a `ValueError` that says they have no backward yet.
2. `max` and `min`: the tie rule, its tests, and the comparison with the
   CUDA behavior of PyTorch.
3. `segment_softmax`: the backward, its accuracy tests, special values, the
   second derivative, and the capture refusal.
4. Qualification: a recorded run of the gradient tests and of the whole
   GPU directory on the development GPU, stated in
   [Verification](../internals/verification.md), and the support-matrix
   row.

## Answers to the questions

1. Tie rule: equal shares.
2. Error class for `out` while recording: `ValueError`, as in the call's
   other refusals. PyTorch raises a `RuntimeError` for its own operations.
3. Floating-point segment sums in the backward: the Swage sum kernel.
4. Second derivatives: supported.
5. A `torch.library` operator: later, as its own decision.
6. Running the GPU tests on the trusted runner before merge: the owner
   keeps `ci-gpu.yml` running on `main` only. The qualification is instead
   a recorded run on the development GPU, which
   [Verification](../internals/verification.md) states with its commit and
   GPU, and which says that the trusted workflow has not run it.
7. Which offsets the backward keeps: the caller's tensor, checked by its
   version counter, and a copy only for an inference tensor.
8. Reusing the host offsets of the forward in the softmax backward: not
   before a measurement.
9. A backward candidate in the benchmark harness: after step 3, with no
   claim until a record exists.
10. A new record or an amendment of ADR-0022: this new record. ADR-0022
    keeps its scope line as history.
