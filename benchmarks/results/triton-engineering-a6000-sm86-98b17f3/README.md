<!-- benchmarks/results/triton-engineering-a6000-sm86-98b17f3/README.md -->
# Swage and Triton comparison at 98b17f3

Swage has competitive prepared kernels for selected segment distributions, but its public call lost to the fixed Triton baseline in all 17 fresh-offset cases measured here. The mixed prepared kernel won 4 of 6 cases against the fixed Triton configuration and 3 of 6 against the fastest configuration selected after the sweep.

These are local engineering measurements, not release or archival qualification. Source and benchmark code were unchanged. Three independent processes ran each configuration, with separate empty compiler caches, 5 warmups and 30 timed samples per case. Values were float32 nonzero quarter multiples (seed 7); every measured result passed exact-reference checks, with zero bounded or unchecked result counts. The prepared harness also checks position-dependent values. This does not establish agreement for NaNs, infinities, arbitrary floating-point values, or gradients.

| Environment | Value |
|---|---|
| Source | `98b17f38d7feccb8a31d7e45b10db510db5aeed1` |
| GPU | NVIDIA RTX A6000, `sm_86`, 84 SMs |
| PyTorch / Triton | `2.12.0+cu130` / `3.7.0` |
| CPU | AMD Ryzen 9 7900X, amd-pstate-epp, powersave governor |
| GPU use | Runs serialized; no other compute process observed at sampled boundaries; exclusive allocation not asserted |
| Native compiler | Rebuilt Release bindings from the reviewed checkout; exact library and loaded PTX hashes are in each record |

Times below are microseconds, using the median of the three process medians. A ratio is calculated within each process first, then summarized by its median and [minimum, maximum]. A ratio below 1 favors Swage. The fixed Triton configurations were chosen before these runs: rank one uses block 256 and 4 warps; feature rows use 16 rows per loop step and 4 warps. The separate best-sweep ratio selects one configuration per case by the lowest median across processes. It is an optimistic bound for Triton, with tuning cost excluded.

Public-call timing includes validation, classification, allocation, launch, and synchronization with a new offsets layout on every sample. Prepared timing excludes compilation and preparation and uses CUDA graph replay to reduce host-launch gaps. These boundaries must not be mixed. Triton baselines are handwritten benchmark kernels, not a claim about every possible Triton implementation.

## Rank-one public calls

8,192 segments, all nine synthetic distributions. The private direct-CTA candidate is also recorded, but differs from the public call in validation and dispatch and is not treated as a pure planner ablation.

| Case | Swage us | Fixed Triton us | Swage / fixed Triton [range] | Swage / best sweep | Best sweep configuration |
|---|---:|---:|---:|---:|---|
| alternating-empty | 117.13 | 16.81 | 7.05 [6.93, 7.05] | 7.88 | `triton_looped_b512_w1` |
| bimodal | 127.56 | 24.38 | 5.24 [5.21, 5.28] | 5.32 | `triton_looped_b512_w1` |
| few-huge | 120.06 | 17.80 | 6.74 [6.71, 6.78] | 7.56 | `triton_looped_b256_w2` |
| log-normal | 136.13 | 17.52 | 7.80 [7.77, 7.80] | 8.76 | `triton_looped_b256_w2` |
| many-tiny | 117.08 | 17.11 | 6.93 [6.61, 7.01] | 7.99 | `triton_looped_b256_w1` |
| one-outlier | 118.43 | 17.25 | 6.82 [6.78, 7.09] | 7.82 | `triton_looped_b128_w2` |
| power-law | 155.57 | 23.56 | 6.68 [6.50, 6.73] | 7.68 | `triton_looped_b512_w4` |
| uniform | 209.14 | 106.29 | 1.97 [1.97, 1.97] | 1.98 | `triton_looped_b1024_w8` |
| zipf-like | 134.96 | 21.69 | 6.15 [6.14, 6.29] | 6.83 | `triton_looped_b128_w1` |

[Process summary](fresh-rank1/summary.json) contains every candidate and reference, including PyTorch.

## Rows-of-features public calls

2,048 segments, D=3 and D=64, four distributions. Uniform D=64 at this segment count exceeded the harness memory admission bound, so the attempted run was rejected before measurement; the final matrix uses bimodal instead. The failed attempt is retained under `fresh-rank2/`.

| Case | Swage us | Fixed Triton us | Swage / fixed Triton [range] | Swage / best sweep | Best sweep configuration |
|---|---:|---:|---:|---:|---|
| bimodal D=3 | 154.06 | 34.19 | 4.51 [4.42, 4.53] | 7.44 | `triton_rows_looped_r64_w4` |
| bimodal D=64 | 372.32 | 222.14 | 1.68 [1.67, 1.69] | 1.68 | `triton_rows_looped_r16_w4` |
| many-tiny D=3 | 85.28 | 12.82 | 6.56 [6.37, 6.87] | 6.94 | `triton_rows_looped_r16_w1` |
| many-tiny D=64 | 100.05 | 25.90 | 3.88 [3.81, 3.93] | 4.20 | `triton_rows_looped_r16_w1` |
| one-outlier D=3 | 141.55 | 22.27 | 6.41 [6.31, 6.46] | 9.25 | `triton_rows_looped_r64_w4` |
| one-outlier D=64 | 154.15 | 87.82 | 1.77 [1.72, 1.79] | 1.77 | `triton_rows_looped_r16_w4` |
| power-law D=3 | 143.70 | 33.86 | 4.30 [4.14, 4.34] | 7.83 | `triton_rows_looped_r64_w4` |
| power-law D=64 | 172.02 | 145.23 | 1.18 [1.16, 1.20] | 1.18 | `triton_rows_looped_r16_w4` |

[Process summary](fresh-rows/summary.json) contains every candidate and reference, including PyTorch.

## Prepared rank-one kernels

8,192 segments, six distributions, private prepared API. The raw records also contain synchronized host-call and batched-event timing.

| Case | Swage us | Fixed Triton us | Swage / fixed Triton [range] | Swage / best sweep | Best sweep configuration |
|---|---:|---:|---:|---:|---|
| bimodal seed=7 | 16.51 | 15.49 | 1.07 [1.07, 1.07] | 1.10 | `triton_looped_b256_w2` |
| few-huge seed=7 | 5.69 | 8.96 | 0.64 [0.64, 0.64] | 0.81 | `triton_looped_b256_w2` |
| log-normal seed=7 | 8.08 | 8.83 | 0.92 [0.92, 0.92] | 1.17 | `triton_looped_b256_w2` |
| many-tiny seed=7 | 3.30 | 8.32 | 0.40 [0.40, 0.40] | 0.51 | `triton_looped_b256_w1` |
| one-outlier seed=7 | 4.58 | 8.41 | 0.54 [0.54, 0.54] | 0.71 | `triton_looped_b256_w1` |
| uniform seed=7 | 97.65 | 96.54 | 1.01 [1.01, 1.01] | 1.02 | `triton_looped_b512_w8` |

[Process summary](prepared-validated/summary-torch_segment_reduce-triton_looped_b256_w4.json) contains every candidate and reference, including PyTorch.

## Interpretation and compiler idea

Triton already expresses runtime-length loops and programmer-written device schedules. Swage’s distinct proposal is to describe segment-local semantics and let its compiler derive admitted warp, CTA, and split decompositions. Its public segmented API currently runs fixed programs; the broader programmable segment surface remains private.

The prepared results support further work on this abstraction: the mixed schedule is substantially better than the same compiler’s direct CTA on several tested layouts. They do not establish a general advantage over Triton. In fresh calls, host orchestration and API work must repay their costs; that did not happen against the fixed Triton baselines in this matrix. Comparisons with earlier records cannot establish a regression because machine state and benchmark configurations differ.

Official references: [Triton runtime loops](https://triton-lang.org/main/python-api/generated/triton.language.range.html), [device scheduling in grouped GEMM](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html), and [configuration autotuning](https://triton-lang.org/main/python-api/generated/triton.autotune.html). These docs describe current main; the measurements above used installed Triton 3.7.0.

## Prepared-harness validation defect

Each prepared measurement wrote its complete JSON, then exited nonzero: `benchmark_provenance.start` retains `torch.__version__` as a `TorchVersion` string subclass, while `benchmark_campaign._string` requires `type(value) is str`. All three serialized records passed the full existing `validate_child` function when reloaded, since JSON stores the same version as a plain string. Timing data and compiler code were not changed. Copies of the three validated records were summarized using the existing campaign summarizer. This recovery is not a successful campaign/archival gate.

Raw process JSON is stored as lossless `.json.xz` files, following the existing benchmark-record convention. Decompress each process file to its original `.json` name before using the campaign summarizer; the summaries retain the original names and SHA-256 hashes of those uncompressed bytes.

Original records and failure logs are retained under `prepared-rank1/`, `prepared-repeat-1/`, and `prepared-repeat-2/`. Byte-identical collected records are under `prepared-validated/`. The two additional commands are in `prepared-repeat-commands.json`. Follow-up: normalize the recorded version to a plain string at the provenance boundary and add a regression test. No issue was filed.

## Reproduction and scope

Run from a clean checkout of the recorded revision with its rebuilt native bindings on `PYTHONPATH`. The full argument lists for fresh-offset campaigns are in their summaries; raw records include distributions, sample counts, candidate lists, source identity, native hashes, PTX hashes, timings, correctness checks, and environment observations. The temporary clean checkout used here was `/tmp/swage-triton-review-src-98b17f3`.

No compiler or benchmark source changed during measurement. This directory preserves the report, raw JSON, summaries and logs from the local engineering run. Original paths inside the records are retained for provenance; the measured source revision is 98b17f3, independently of the commit that stores these results. Existing untracked planning and history files were preserved.

Not measured: softmax, other reduction kinds, float64, gradients, pipelined public calls, another GPU, or held-out random seeds. No exhaustive numerical equivalence, universal performance, or release qualification claim follows from these measurements.
