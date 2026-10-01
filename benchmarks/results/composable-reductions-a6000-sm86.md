<!-- benchmarks/results/composable-reductions-a6000-sm86.md -->

# Composable reduction benchmark

Mixed scheduling is useful across the new sum/max element programs, but
length-only splitting is not consistently profitable. Across two local runs,
packing tiny segments was about 2.6 times faster than the best pure schedule,
and a workload with 40 long segments was about 1.9 times faster. Splitting
256 uniform 8,192-element segments was about 17% slower than direct CTA work.
This supports adding a profitability decision before splitting.

These are provisional engineering measurements from a modified worktree,
not a frozen performance gate or a before/after implementation comparison.

## Reproduce and inspect

```sh
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
  python benchmarks/benchmark_composable_reductions.py \
  --samples 50 --warmups 25 --output /tmp/composable-reductions.json
```

The [harness](../benchmark_composable_reductions.py) reuses the existing timing
helpers and the semantic programs from compiler/runtime qualification.
[First-run raw samples](composable-reductions-a6000-sm86.json) and
[repeat raw samples](composable-reductions-a6000-sm86-repeat.json) include
quartiles, source status, relevant source and native binary SHA-256 hashes,
and environment provenance. Neither run changes the frozen sum benchmark.

Both runs used NVIDIA RTX A6000 (`sm_86`), Python 3.13.13, PyTorch 2.12.0+cu130, and CUDA 13.0. The source base was `23fda044cf4851c7b4234f77f5a569441e0c25af` with the
recorded uncommitted changes. The recorded UTC start times were
`2026-09-13T09:12:52.363417+00:00` and `2026-09-13T09:13:40.078887+00:00`.

NVML failed with a driver/library version mismatch, so clocks, power,
utilization, and device exclusivity were not verified. CUDA execution and
correctness checks succeeded. Sequential timing samples and reused buffers
make these results representative of repeated execution, including warm-cache
effects, rather than isolated cold-memory performance.

## Workloads and measurement

The seed is 7. Each workload runs sum and max over `x`, `x*x`, and
`2*(x+1)`. Inputs are seeded float32 quarter values in `[-1, 0.75]`.
Every policy matches a CPU `torch.segment_reduce` reference exactly before
timing; Swage outputs are checked again after timing. This is 96 policy
comparisons per run, 192 across both runs, covering empty and split segments.
It is a finite-input benchmark, not exhaustive numerical qualification.

| Workload | Segments | Elements | Maximum length | Mixed launches |
| --- | ---: | ---: | ---: | ---: |
| many-tiny | 32,768 | 525,034 | 32 | 1 |
| bimodal | 32,768 | 8,835,812 | 4,095 | 1 |
| split-only | 256 | 2,097,152 | 8,192 | 2 |
| mixed-with-splits | 4,096 | 3,697,080 | 65,536 | 3 |

The mixed split workload has 3,600 segments of length 0–32, 456 of length
33–4,096, and 40 of length 65,536, shuffled deterministically. The tiny and
bimodal workloads reuse the existing distribution generator unchanged.

Pure warp uses one 32-thread block per segment. Pure CTA uses one 128-thread
block per segment. Mixed packs four small segments into a 128-thread block,
uses direct CTA work through length 4,096, and launches 512-thread partial
and merge kernels above that threshold. Thresholds were not tuned for these
measurements.

Each policy has 25 warmup calls and 50 timing samples. Graph measurements
capture 32 calls, warm up replay, and divide CUDA-event replay time by 32.
Synchronized Python-call timings are recorded separately. Policy order rotates
across cases; policies are sampled sequentially within each case.

The PyTorch baseline includes the eager transform and `segment_reduce`, with
output/intermediate allocations inside the call. Swage fuses the transform
and writes a preallocated output. Graph replay reuses captured allocations.
This comparison measures those execution paths; it is not a comparison with
a hand-optimized competing GPU kernel.

## Results

First-run median CUDA graph time per call, in microseconds. Preparation and
Python dispatch are excluded from this table. Lower is better.

| Workload | Program | Warp | CTA | Mixed | PyTorch |
| --- | --- | ---: | ---: | ---: | ---: |
| many-tiny | `sum(x)` | 23.20 | 39.31 | 8.54 | 50.88 |
| many-tiny | `sum(x*x)` | 22.11 | 37.18 | 8.54 | 52.86 |
| many-tiny | `sum(2*(x+1))` | 22.11 | 37.25 | 8.60 | 58.30 |
| many-tiny | `max(x)` | 22.11 | 37.16 | 8.57 | 95.82 |
| many-tiny | `max(x*x)` | 22.11 | 37.28 | 8.58 | 100.88 |
| many-tiny | `max(2*(x+1))` | 22.11 | 37.31 | 8.61 | 106.01 |
| bimodal | `sum(x)` | 65.34 | 65.50 | 60.81 | 91.30 |
| bimodal | `sum(x*x)` | 65.47 | 65.58 | 60.86 | 207.36 |
| bimodal | `sum(2*(x+1))` | 65.65 | 65.74 | 61.01 | 340.66 |
| bimodal | `max(x)` | 65.31 | 65.50 | 60.90 | 181.10 |
| bimodal | `max(x*x)` | 65.47 | 65.63 | 60.96 | 307.65 |
| bimodal | `max(2*(x+1))` | 65.68 | 72.87 | 60.93 | 431.86 |
| split-only | `sum(x)` | 23.84 | 14.50 | 16.93 | 14.62 |
| split-only | `sum(x*x)` | 24.03 | 14.46 | 16.90 | 36.93 |
| split-only | `sum(2*(x+1))` | 24.19 | 14.46 | 16.86 | 61.92 |
| split-only | `max(x)` | 23.84 | 14.50 | 16.93 | 15.45 |
| split-only | `max(x*x)` | 24.10 | 14.46 | 16.86 | 37.77 |
| split-only | `max(2*(x+1))` | 24.22 | 14.49 | 16.90 | 62.78 |
| mixed-with-splits | `sum(x)` | 196.88 | 54.27 | 28.51 | 27.94 |
| mixed-with-splits | `sum(x*x)` | 197.57 | 54.53 | 28.77 | 82.69 |
| mixed-with-splits | `sum(2*(x+1))` | 195.46 | 54.94 | 28.58 | 127.32 |
| mixed-with-splits | `max(x)` | 193.25 | 54.34 | 28.61 | 37.54 |
| mixed-with-splits | `max(x*x)` | 194.19 | 54.91 | 28.61 | 94.64 |
| mixed-with-splits | `max(2*(x+1))` | 195.39 | 55.03 | 28.70 | 138.67 |

Mixed graph medians changed by at most 0.99% in the repeat. Some pure-schedule and PyTorch cases had considerably wider within-run
quartiles, recorded in the raw files. The small bimodal advantage should be
rechecked under controlled clocks and device load before making a stronger
performance claim.

The entire prepared policy bundle took 19.4–32.4 ms across both runs.
This is one synchronized wall-time sample per case, including validation,
offset transfer, planning, compilation, module loading, and metadata/scratch
allocation. Input/output allocation and CUDA initialization are excluded.
The first case also includes native initialization; driver caches were not
cleared. These are not cold-compilation or per-schedule preparation costs.

| Workload | Mixed synchronized call range, µs | Bundle preparation range, ms |
| --- | ---: | ---: |
| many-tiny | 15.88–18.30 | 22.4–32.4 |
| bimodal | 68.81–69.86 | 21.3–27.5 |
| split-only | 25.26–27.70 | 19.4–28.7 |
| mixed-with-splits | 38.14–38.73 | 26.0–32.0 |

Ranges cover per-case medians for all six programs and both runs, rather
than confidence intervals. Preparation is much larger than warm execution,
so these results depend on reusing prepared work. The harness prepares all
three policies together; it does not measure incremental mixed preparation
or establish an amortization break-even point.

The results also show where fusion helps: transformed PyTorch programs pay
for intermediate passes, while Swage's transformed and identity graph times
are close. Identity sum is not universally faster than PyTorch: mixed work
with splits is slightly slower, and direct CTA is preferable for uniform
8,192-element segments.

## Validation and follow-up

Both benchmark processes completed: 48 program/workload cases, 192 initial
exact policy comparisons, all graph captures available, and 19,200 positive
timing samples. The 35 existing benchmark contract/distribution tests passed;
Ruff and whitespace checks passed. This benchmark-only work changes no
compiler or runtime semantics; the compiler suites were not rerun for it.

Follow-up issue, not filed: add and validate split profitability selection
using segment count, length distribution, and launch overhead. Start with
the uniform 8,192-element regression and retain the mixed long-segment win.
Add controlled hardware and broader workloads before selecting new defaults.
Softmax, persistent execution, automatic schedule search, and cold-cache
qualification were outside this benchmark.
