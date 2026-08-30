<!-- docs/adr/ADR-0015-minimal-mixed-policy-execution.md -->
# ADR-0015: Minimal mixed-policy execution

- Status: accepted
- Date: 2026-08-24

## Context

The existing planning boundary records warp and CTA as legal policies for one
canonical identity segmented sum and classifies validated host metadata into
stable task descriptors. It does not connect the planning operation to that
classifier or execute either policy. Mixed-policy qualification needs the
smallest execution proof that preserves the existing semantic module and
public API boundaries while establishing a predeclared performance gate.

## Decision

### Reproducible segment distributions

This benchmark adds a standard-library-only generator under `benchmarks/`,
outside the installed package. `generate_lengths(name, count, seed)` uses a
private `random.Random(seed)` instance and supports these named distributions:

- `uniform`: inclusive uniform lengths from 0 through 4096;
- `log-normal`: median 32 and sigma 1.5, rounded to integer lengths and capped
  at 4096;
- `bimodal`: nine small lengths from 1 through 32 for each long length from
  1024 through 4096;
- `zipf-like`: exponent 1.2 over integer lengths from 1 through 4096;
- `many-tiny`: inclusive uniform lengths from 0 through 32;
- `few-huge`: nineteen short lengths from 0 through 4 for each long length
  from 1024 through 4096;
- `one-outlier`: one length of 4096 and all remaining lengths from 1 through
  32;
- `alternating-empty`: zero at each even position and a length from 1 through
  32 at each odd position.

For ratios that do not divide `count`, each complete group has the stated
ratio and the remainder is assigned to the short class. Generated positions
are shuffled for bimodal, few-huge, and one-outlier so policy classes are not
grouped. Counts must be positive and small enough that `count * 4096` fits in
a signed i32 total. This bound is checked before generation. Seeds must be
integers.

The generator reports count, total, minimum, median, nearest-rank p95, and
maximum. The nearest-rank p95 is the element at one-based rank
`ceil(0.95 * count)` in sorted order. The generator and its statistics are
covered by deterministic property-style pytest checks without Hypothesis or
another dependency.

### Policy execution boundary

The existing segmented-reduction GPU conversion gains optional internal task
ID indirection while preserving direct segment-ID mode. Task mode admits only
the canonical identity segmented sum. The x-block ID selects the original
segment ID used for offset loads and the output store, while the reduction
body remains the existing block-stride emitter. Qualification compiles
separate pure-warp and pure-CTA kernels. The exact private task ABI lives in
[Task Execution](../internals/task-execution.md).

### Classifier integration and launch ordering

The planning pass remains restricted to the identity-sum shape and accepts
a configurable nonnegative `warpMaxElements`, defaulting to 32. A private
native binding clones the semantic module, runs the planning pass, finds the
matching `swage_plan.classify`, reads its threshold, invokes the existing host
classifier, and returns stable warp and CTA segment-ID arrays. The source
module is not mutated.

The private Python preparation path validates tensors and offsets and
materializes both task classes before compilation, allocation, or launch. It
returns prepared `warp`, `cta`, and `mixed` callables. The mixed callable
launches warp tasks first and CTA tasks second on the current PyTorch stream.
An empty task list enqueues no kernel. Submitted values, offsets, output, and
task-ID tensors are recorded on that stream. Validation, classification,
compilation, allocation, or launch failure is reported directly without
backend or policy fallback.

This remains an internal qualification boundary. It adds no public Python
API and no SwagePlan operation, attribute, or type.

### Predeclared performance gate

The frozen timing run uses exactly:

- NVIDIA RTX A6000 at `sm_86`;
- the `bimodal` generator with seed 7 and 32,768 segments;
- a warp threshold of 32;
- 32 threads for the warp policy and 128 threads for the CTA policy;
- 25 warmups for each pure or mixed policy;
- 100 interleaved CUDA-event samples per policy;
- all-one f32 input values, making each exact expected sum equal to its
  segment length.

Compilation, module loading, classification, and allocation occur before
timing and are excluded. The committed record at
`benchmarks/results/mixed-sum-a6000-sm86.json` contains every raw sample,
environment metadata, generated distribution statistics, policy medians, and
the mixed-to-best-pure ratio.

The performance gate passes only when:

```text
mixed_median <= 1.05 * min(pure_warp_median, pure_cta_median)
```

If this gate fails, the distribution and policy constants above do not
change. The implementation must improve or qualification remains incomplete.

## Acceptance boundary

The boundary is accepted when tests prove pure-warp, pure-CTA, and mixed
correctness for empty, boundary, repeated-empty, skewed, and large segments;
failure paths enqueue no work and do not fall back; the committed A6000 record satisfies the
predeclared gate; and the final PR and merged tree pass the applicable Python,
MLIR, native binding, documentation, formatting, and GPU runtime checks.

## Explicit deferrals

This decision does not add packed warps, split CTAs, partial reductions,
ragged-softmax planning, device queues, persistent scheduling, public
segmented launch, general benchmark infrastructure, a release, or unrelated
semantic-dialect work.

## Consequences

- The benchmark measures the scheduling decision without widening the
  semantic or public runtime surface.
- Both pure policies use the same host contract, so the benchmark compares
  policy execution rather than different interfaces.
- Long segments still run in one CTA and short segments still use one whole
  warp. More general task decomposition remains future work.
