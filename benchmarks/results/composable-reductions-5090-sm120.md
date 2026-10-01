<!-- benchmarks/results/composable-reductions-5090-sm120.md -->

# Composable reductions on RTX 5090

The current programs passed correctness checks on RTX 5090 (`sm_120`).
Mixed scheduling outperformed the best pure schedule on tiny, bimodal,
and mixed long-segment workloads. Uniform 8,192-element segments still
ran faster with direct CTA execution, confirming the split-profitability
problem on a second GPU architecture.

| Workload | Mixed graph time, µs | Mixed speedup over best pure schedule |
| --- | ---: | ---: |
| many-tiny | 4.62–4.63 | 3.41–3.41× |
| bimodal | 9.99–10.04 | 1.80–1.82× |
| split-only | 4.11–4.11 | 0.91–0.92× |
| mixed-with-splits | 7.44–7.51 | 2.82–2.88× |

Each value is a median of three per-run medians. Ranges cover the six
programs, not confidence intervals. A speedup below 1 means mixed is slower.
Mixed graph medians varied by less than 0.5% across the three runs.

## Evidence and reproduction

The [raw evidence](composable-reductions-5090-sm120.json) contains all three
runs, 28 one-second telemetry samples, compiler/runtime test output, the
CMake configuration, native library hashes, and the exact remote runner.
The unchanged [benchmark harness](../benchmark_composable_reductions.py)
ran with 50 samples, 25 warmups, and 32 captured calls per graph replay:

```sh
CUDA_VISIBLE_DEVICES=GPU-8a89028f-7767-c5ce-7493-a6de33619f1a \
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
  python benchmarks/benchmark_composable_reductions.py \
  --samples 50 --warmups 25 --output /tmp/composable-5090-run-1.json
```

Repeat in three separate Python processes. The source base was
`23fda044cf4851c7b4234f77f5a569441e0c25af`, overlaid with the current local
changes and benchmark fixtures, in the isolated remote directory
`/tmp/swage-composable-5090-zq2URm/source`. All six recorded non-binary
source hashes match the A6000 benchmark sources and current local files.
The existing remote checkout was not modified. No changes were committed.

The build used Release mode, pinned LLVM `llvmorg-22.1.8`, and four parallel
build jobs. CMake needed the host's existing Conda zlib paths explicitly;
no system software or LLVM pin was changed.

## Environment and limits

The host was `dc-03-node21`, physical GPU 1, NVIDIA GeForce RTX 5090,
driver 580.173.02, CUDA 13.0, Python 3.13.11, and PyTorch 2.13.0+cu130.
The runs started on 2026-09-13 UTC. Three other compute processes occupied
about 4,752 MiB on the selected GPU before timing. This was a shared-device
engineering run, not an archival performance gate.

During timing, sampled SM clocks ranged from 2,557 to 2,955 MHz; memory
clocks stayed at 13,801 MHz. Temperature ranged from 22 to 34 °C, power from
65 to 334 W, and total GPU utilization from 0 to 99%. These samples include
the benchmark's own work and cannot separate its utilization from co-tenants.
Clocks were not locked and no other processes were stopped.

The [A6000 run](composable-reductions-a6000-sm86.md) used the same inputs,
programs, policies, thresholds, and timing controls. Its PyTorch version was
2.12.0+cu130 and its CPU/compiler environment differed, so a cross-host ratio
is not an isolated measurement of GPU hardware improvement. Both records
are provisional. Sequential samples and repeated buffers include warm-cache
effects; graph timing excludes Python dispatch and preparation.

PyTorch timings include the eager transform and `segment_reduce`. Swage
fuses the transform and uses preallocated output. Graph replay reuses captured
allocations. This is not a hand-optimized competing-kernel comparison.

## Program results

Median of three per-run graph medians, in microseconds per call. Lower is
better. `maps` denotes the two fused maps implementing `2*(x+1)`.

| Workload | Program | Warp | CTA | Mixed | PyTorch |
| --- | --- | ---: | ---: | ---: | ---: |
| many-tiny | `sum(x)` | 15.758 | 16.029 | 4.626 | 21.687 |
| many-tiny | `sum(x*x)` | 15.778 | 16.027 | 4.625 | 22.998 |
| many-tiny | `sum(2*(x+1))` | 15.774 | 16.027 | 4.629 | 24.268 |
| many-tiny | `max(x)` | 15.774 | 16.065 | 4.625 | 24.344 |
| many-tiny | `max(x*x)` | 15.792 | 16.024 | 4.630 | 25.646 |
| many-tiny | `max(2*(x+1))` | 15.781 | 16.028 | 4.624 | 26.969 |
| bimodal | `sum(x)` | 19.159 | 18.015 | 9.999 | 24.201 |
| bimodal | `sum(x*x)` | 19.257 | 18.185 | 9.998 | 36.114 |
| bimodal | `sum(2*(x+1))` | 19.319 | 18.158 | 10.014 | 64.757 |
| bimodal | `max(x)` | 19.181 | 18.013 | 9.988 | 32.394 |
| bimodal | `max(x*x)` | 19.329 | 18.226 | 10.043 | 44.047 |
| bimodal | `max(2*(x+1))` | 19.349 | 18.209 | 10.018 | 72.489 |
| split-only | `sum(x)` | 10.128 | 3.724 | 4.108 | 3.155 |
| split-only | `sum(x*x)` | 10.257 | 3.729 | 4.113 | 6.716 |
| split-only | `sum(2*(x+1))` | 10.378 | 3.782 | 4.114 | 10.219 |
| split-only | `max(x)` | 10.165 | 3.727 | 4.107 | 3.736 |
| split-only | `max(x*x)` | 10.368 | 3.737 | 4.112 | 7.262 |
| split-only | `max(2*(x+1))` | 10.459 | 3.791 | 4.111 | 10.798 |
| mixed-with-splits | `sum(x)` | 72.543 | 21.012 | 7.438 | 10.554 |
| mixed-with-splits | `sum(x*x)` | 73.628 | 21.331 | 7.443 | 16.128 |
| mixed-with-splits | `sum(2*(x+1))` | 74.413 | 21.469 | 7.513 | 21.601 |
| mixed-with-splits | `max(x)` | 72.836 | 21.072 | 7.444 | 14.169 |
| mixed-with-splits | `max(x*x)` | 74.467 | 21.449 | 7.456 | 19.719 |
| mixed-with-splits | `max(2*(x+1))` | 75.248 | 21.636 | 7.504 | 25.173 |

Preparing the complete three-policy bundle took 33–102 ms across individual
cases. The first case in each process took 97–102 ms; other preparations
were 33–53 ms. These are single wall-time samples including validation,
offset transfer, planning, compilation, module loading, and metadata/scratch
allocation. CUDA initialization and input/output allocation are excluded;
compiler and driver caches were not cleared. No cold-cache claim is made.

Prepared mixed synchronized Python-call medians ranged from 15.7 to 25.6 µs.
These include dispatch and synchronization and should not be substituted for
the graph times above. Preparing all policies together does not establish
incremental mixed-policy preparation cost or an amortization break-even.

## Validation and follow-up

Before benchmarking, `ninja -C build -j2 check-swage check-swage-unit
check-swage-python` passed: 37 lit tests, 9 C++ unit tests, and 309 native
Python tests, with no reported skips. All 72 benchmark program/workload cases
passed, totaling 288 initial exact comparisons against the CPU reference.
Swage outputs were also checked after timing. All graph captures succeeded;
the raw artifact contains 28,800 positive timing samples. Source hashes and
sample counts were verified after copying the evidence locally.

Only this report and its raw JSON were added locally; compiler/runtime
semantics and benchmark code are unchanged. Public Python, packaging, and
documentation-site suites were not rerun for these evidence-only additions.

Follow-up issue, not filed: choose split execution using measured workload
profitability, preserving the mixed long-segment wins while avoiding the
uniform 8,192-element regression. Keep architecture-specific evidence before
changing defaults. Softmax and persistent performance were not benchmarked.
