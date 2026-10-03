<!-- benchmarks/results/split-selection.md -->

# Direct CTA selection for regular split batches

Preparation now selects the existing pure CTA implementation for batches
containing only 4097–8192-element segments, using the default 4096-element
chunk size, when the segment count is at least the device SM count. It skips
split kernel compilation and scratch allocation for those batches. Selection
uses validated metadata and does not execute the user's program during setup.

The selected `mixed` callable aliases `cta`. Sparse batches, mixed direct/split
batches, larger segments, and custom chunk sizes keep the fixed schedule.
`select_schedule=False` explicitly opts out. The legacy `_prepare_planned_sum`
wrapper always opts out, so frozen benchmark contracts remain unchanged.

## Results against our fixed policy

The same four distributions and six programs were measured in three separate
processes per mode per GPU, alternating fixed and selected modes. Every run
used 25 warmups, 50 samples, and 32 calls per captured graph. Tables show
medians of the three per-run graph medians, summarized across the six programs.

| GPU | Workload | Fixed graph time, µs | Selected graph time, µs | Per-program latency change |
| --- | --- | ---: | ---: | ---: |
| A6000 | many-tiny | 8.70 | 8.75 | -0.14% to +0.74% |
| A6000 | bimodal | 60.99 | 60.97 | -0.06% to +0.06% |
| A6000 | split-only | 16.99 | 14.43 | -15.38% to -14.72% |
| A6000 | mixed-with-splits | 28.72 | 28.67 | -0.44% to +0.11% |
| 5090 | many-tiny | 4.63 | 4.63 | -0.15% to +0.14% |
| 5090 | bimodal | 10.01 | 10.01 | -0.02% to +0.09% |
| 5090 | split-only | 4.11 | 3.74 | -9.47% to -7.86% |
| 5090 | mixed-with-splits | 7.45 | 7.47 | -0.05% to +0.72% |

Negative changes mean lower latency. The uniform 256 × 8192 workload improved
by 14.7–15.4% on A6000 and 7.9–9.5% on RTX 5090. Other workloads retained their
schedules, with per-program graph differences under 1% in either direction.
That variation does not establish an improvement on unchanged schedules.

Preparation also improved for the uniform 8192 workload. The median across
its six programs and three processes fell from 22.2 to 11.5 ms on A6000 and
from 34.6 to 16.8 ms on RTX 5090. These are complete policy-bundle preparation
samples, including validation, planning, compilation, loading, and metadata
allocation. Input/output allocation and initial CUDA startup are excluded.
Caches were not cleared; these are not cold-cache measurements.

## Why include segment count

An exploratory identity-sum sweep covered segment counts 16, 64, 128, 256,
and 1024 at lengths 4097, 8192, 16384, and 65536 on both GPUs. It used
10 warmups and 20 graph samples per policy; raw samples and the script are
included in each evidence file.

At 16 segments of length 8192, direct CTA was slower than split work on both
GPUs. It also lost at 64 and 128 segments on RTX 5090. Raising the length
threshold alone would therefore introduce regressions. The selected rule uses
one segment per SM as a conservative parallelism requirement: 84 on the A6000
and 170 on the RTX 5090. This is a measured heuristic, not an occupancy model
or a general cost model. It does not claim optimality at every count or length.

## Evidence and reproduction

- [A6000 raw fixed/selected runs, exploration, and tests](split-selection-a6000-sm86.json)
- [RTX 5090 raw fixed/selected runs, exploration, tests, and telemetry](split-selection-5090-sm120.json)
- [Original A6000 baseline](composable-reductions-a6000-sm86.md)
- [Original RTX 5090 baseline](composable-reductions-5090-sm120.md)

The original baseline files are unchanged. The new fixed controls use the
same kernels and scheduling rules, with selection explicitly disabled, in the
current benchmark harness. Reproduce from a matching build-tree environment:

```sh
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
  python benchmarks/benchmark_composable_reductions.py \
  --samples 50 --warmups 25 --fixed-schedule --output /tmp/fixed.json
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
  python benchmarks/benchmark_composable_reductions.py \
  --samples 50 --warmups 25 --output /tmp/selected.json
```

Run each mode in three separate processes. The raw files include exact
commands, source status, relevant source/native module hashes, and environment
versions. All recorded non-binary source hashes were verified against the
local files after measurement. Source base remains
`23fda044cf4851c7b4234f77f5a569441e0c25af` plus recorded uncommitted changes.

A6000 used `sm_86`, Python 3.13.13, PyTorch 2.12.0+cu130; RTX 5090 used
`sm_120`, Python 3.13.11, PyTorch 2.13.0+cu130. Both used CUDA 13.0 and the
previously qualified native builds against pinned LLVM 22.1.8. The RTX 5090
used the same isolated temporary source directory and physical GPU 1 as the
original run. Device telemetry is included for RTX 5090; A6000 NVML remains
unavailable. These remain provisional shared-device/worktree measurements.
Comparisons in the table are within each GPU and software environment.

## Validation and scope

The native Python suite passed all 322 tests on each GPU, including 13 added
selection tests. These check every benchmark program, graph replay, output
guards, skipped split compilation, sparse/long/mixed/empty batches, custom
limits, explicit opt-out, and invalid selector configuration. The 35 benchmark
contract/distribution tests passed locally, along with Ruff, whitespace
checks, and strict MkDocs. No tests reported skips in these runs.

All 288 benchmark program/workload cases passed, totaling 1152 initial exact
policy comparisons and 115200 positive timing samples. Swage outputs were
also checked after timing. All graphs captured successfully. Native C++ and
lit suites were not rerun for this Python-only selection change; the same
native binaries passed qualification in the preceding benchmark task.

Changed files: the private preparation helper, native runtime tests, the
composable benchmark harness, planning documentation and ADR-0019, plus this
report and its two evidence files. Public APIs and native classifier/descriptor
contracts are unchanged. Schedule choice changes reduction trees within the
existing floating-point tolerance contract; no new arithmetic rewrites or
fast-math flags were added. Existing unrelated work was preserved; no commits
or remote publication were performed.

Follow-up issue, not filed: extend profitability selection using held-out
length distributions and element-program costs. The current rule is qualified
for correctness and measured on identity, square, and affine sum/max programs;
performance for more expensive element expressions requires further evidence.
