<!-- benchmarks/results/program-cost-selection.md -->

# Schedule selection using element work

The wider workload suite exposed a limitation in the previous selector: enough
moderately long segments do not make direct CTA execution profitable when each
element performs repeated exponentials or divisions. The new selector bounds
relative element work before choosing CTA. It retains the original shape rules
and reuses the existing native kernels.

| Evaluation | A6000 speedup | RTX 5090 speedup |
| --- | ---: | ---: |
| Calibration cases whose schedule changed | 1.20–3.32× | 1.77–3.70× |
| Independent validation cases whose schedule changed | 1.47–1.88× | 1.98–2.80× |

These are prepared GPU graph speedups over the previous shape-only selector,
not against an oracle or another framework. Each case uses the median of three
per-process medians. Ranges cover individual changed cases, not confidence
intervals. Every case whose selected schedule changed improved in this campaign.

## Change and calibration

The helper inspects typed operations in admitted MLIR map and reduction element
regions. Simple add, subtract, multiply, minimum, and maximum cost one relative
unit each; `math.exp2` costs eight and division sixteen. Constants and yields
cost zero. Work is summed across regions, and CTA selection requires a total
of at most 32. No second IR or text-pattern recognizer was introduced.

The existing shape requirement remains: default 4096-element chunks, every
segment in 4097–8192, and segment count at least the device SM count. Above the
work budget the prepared `mixed` callable retains split execution. These are
empirical scheduling weights, not instruction latency estimates or an occupancy
model. The legacy sum wrapper and `select_schedule=False` preserve fixed rules.

An initial candidate rejected all exponentials/divisions and expressions with
more than four simple operations. That candidate was rejected: on A6000 it made
some single-exp2 and 32-operation affine workloads roughly 25–43% slower. The
first complete paired run and its helper source are retained in the A6000 raw
record. The weighted budget preserves those CTA choices while splitting the
more expensive recurrence programs.

The `held-out` CLI suite was new relative to the earlier uniform benchmark, but
was used during calibration. It is therefore not independent validation of the
final weights. A separate `validation` suite was fixed before its measurements;
weights were not changed after observing that suite.

## Workloads and programs

Calibration uses seed 23 and six distributions: varied lengths in 4097–8192 at
counts 84 and 170; a shuffled 256-segment batch with 230 lengths of 4097 and 26
of 8192; 512 alternating 4097/8192 lengths; a sparse varied batch of 32; and a
256-segment varied batch with one length of 8193. Both sum and max evaluate:

- Identity `x`.
- A single `exp2(x)`.
- Eight iterations of `y = exp2(-0.5*y)`.
- Eight iterations of `y = (y + 0.125)/(1 + 0.25*y*y)`.
- Two or sixteen iterations of `y = -0.5*y + 0.125`, giving 4 or 32 operations.

The recurrence programs are synthetic arithmetic stress cases, not claims about
an application workload. Independent validation uses seed 91, counts 48, 192,
and 384, near-uniform lengths in 7168–8192, varied lengths in 4097–8192, sparse
batches, and lengths in 8193–12288. It evaluates identity plus previously unused
recurrence depths: two exponential steps, four rational steps, and eight affine
steps, each for sum and max.

## Independent validation details

Only the four-step rational cases below change schedule in independent
validation. Sparse and longer batches retain split execution; lower-work
programs retain the previous selector's choices.

| GPU | Workload | Reduction | Before, µs | After, µs | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| A6000 | near-uniform-192 | sum | 21.76 | 11.55 | 1.88× |
| A6000 | near-uniform-192 | max | 21.23 | 11.52 | 1.84× |
| A6000 | varied-384 | sum | 33.92 | 22.94 | 1.48× |
| A6000 | varied-384 | max | 33.92 | 23.04 | 1.47× |
| 5090 | near-uniform-192 | sum | 15.88 | 5.68 | 2.80× |
| 5090 | near-uniform-192 | max | 15.89 | 5.67 | 2.80× |
| 5090 | varied-384 | sum | 15.65 | 7.89 | 1.98× |
| 5090 | varied-384 | max | 15.67 | 7.89 | 1.99× |

Preparation is a tradeoff. For changed validation cases, the median preparation
sample rose from 11.8 to 23.4 ms on A6000 and from 18.3 to 35.9 ms on RTX 5090:
split compilation and scratch allocation are needed again. Median synchronized
call latency across those cases fell from 35.6 to 26.0 µs and from 26.8 to
19.9 µs, respectively. The change targets repeated execution; it does not
establish a cold-start or time-to-first-result improvement.

## Remaining limitations

The shape rule is still imperfect. In the RTX 5090 exploratory run, identity
sum on the varied-170 batch took about 3.81 µs with CTA and 3.36 µs with fixed
split execution. The new work budget leaves that low-cost program eligible for
CTA, so this shape-selection miss remains. This work does not claim that every
chosen schedule is optimal.

Unchanged schedule controls showed shared-run variation. On RTX 5090 their
before/after medians differed by less than 1%. A6000 controls included roughly
4% slowdowns and 8% speedups despite unchanged scheduling; those are not claimed
as effects of this change. NVML remains unavailable on A6000. RTX 5090 telemetry
and co-tenant process information are retained. Neither GPU was exclusive or
clock-locked, so all performance evidence remains provisional.

The work budget counts source operations conservatively, including dead or
foldable work. It is not a general cost model and is not calibrated for every
possible admitted expression, distribution, architecture, or reuse count.

## Evidence, reproduction, and validation

- [A6000 raw runs, calibration, rejected candidate, and tests](program-cost-selection-a6000-sm86.json)
- [RTX 5090 raw runs, calibration, telemetry, and tests](program-cost-selection-5090-sm120.json)
- [Previous selector and benchmark](split-selection.md)

Each GPU record includes three before and three after runs for each suite, the
exact runners, before/after helper source, relevant source and native-module
hashes, and the per-case median summaries. The baseline helper hash exactly
matches the previously reported selector. Both before and after source trees
use the same expanded benchmark and semantic fixtures. Source-isolated baseline
checkouts reuse the already qualified native builds; no native compiler code
changed in this task.

```sh
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
  python benchmarks/benchmark_composable_reductions.py \
  --suite held-out --samples 50 --warmups 25 --output /tmp/calibration.json
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
  python benchmarks/benchmark_composable_reductions.py \
  --suite validation --samples 50 --warmups 25 --output /tmp/validation.json
```

The new suites time prepared CTA and selected mixed execution only, through
synchronized Python calls and 32-call CUDA graph replay. The original default
suite and its exact comparisons remain available. New inputs are seeded
float32 quarter values. Reference evaluation applies the transform on CPU in
float32, then accumulates segments in float64 before casting the result back
to float32. New-suite comparisons use `rtol=1e-4, atol=1e-5`; native nontrivial
arithmetic qualification retains the tighter `1e-5` tolerance. Old exact tests
were not relaxed. New long affine sums use the existing floating-point tolerance
contract because different reduction trees can differ by an ulp.

All 1248 final benchmark program/workload cases passed their reference checks,
with 2496 timed-policy comparisons, 249600 positive timing samples, and every
graph capture available. The native Python suite passed all 351 tests on each
GPU, including typed operation-budget checks spanning map/reduce regions,
expensive-program parity, selected CTA output guards, and graph replay. The
35 benchmark contract/distribution tests, Ruff, whitespace checks, and strict
MkDocs passed. No tests reported skips. C++/lit and public packaging suites
were not rerun for this private Python/qualification change.

A6000 used `sm_86`, Python 3.13.13 and PyTorch 2.12.0+cu130. RTX 5090 used
`sm_120`, Python 3.13.11 and PyTorch 2.13.0+cu130. Both used CUDA 13.0 and the
existing LLVM 22.1.8 builds. Full environment and source provenance remain
inside each run; old evidence files are unchanged.

Changed files: the private preparation helper, semantic program fixtures,
codegen/runtime tests, benchmark suites, planning documentation and ADR-0019,
plus this report and two raw records. Scheduling behavior changes for expensive
programs; accepted mathematical semantics, public APIs, native descriptors,
LLVM pin, and compiler lowering are unchanged. No commits or publication were
performed, and unrelated work was preserved.

Follow-up issue, not filed: model segment-length imbalance and preparation
amortization, using the remaining RTX 5090 varied-170 miss as a regression case.
