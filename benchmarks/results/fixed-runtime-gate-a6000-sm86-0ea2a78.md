<!-- benchmarks/results/fixed-runtime-gate-a6000-sm86-0ea2a78.md -->

# Fixed-runtime gate calibration of the final launch path on the RTX A6000

This record sets the ceiling of the Swage/PyTorch median throughput ratio
at `2^18` elements of `python -m swage.bench vector-add --enforce` to
**1.80**. It supersedes the 1.85 calibration in
[fixed-runtime-gate-a6000-sm86.md](fixed-runtime-gate-a6000-sm86.md) as the
current basis of that ceiling. The earlier record stays as it was, as the
evidence for the move from 1.50. The ceiling at `2^20` elements stays 1.50,
and the cold, warm host, and memory gates are unchanged. The
[raw record](fixed-runtime-gate-a6000-sm86-0ea2a78.json) holds every run,
the machine state around it, the identity of each wheel, and the scripts
that ran them.

## Why a new calibration

The 1.85 ceiling was set from the integration wheel `5e2ee58`. The launch
path changed after that record. The native fixed launch now asks the CUDA
driver whether its stream is capturing (`cuStreamIsCapturing`) instead of
calling into PyTorch for that check, and it still advances the version
counter of its output through the public
`torch.autograd.graph.increment_version`. The ceiling is therefore set
again from a wheel of the final code, `0ea2a78`, against main's wheel in
the same campaign. `0ea2a78` differs from the commit that sets the new
ceiling only in the ceiling value, its tests, and its documentation.

## Results

Measured on the same host on 2026-10-04:

| Wheel | Median ratio at `2^18` (range) | Swage microseconds per launch (range) | Runs that pass the wheel's own ceiling |
|---|---|---|---|
| main, `22558af` | 1.613 (1.572 to 1.641) | 4.80 (4.68 to 4.86) | 0 of 12 (ceiling 1.50) |
| final, `0ea2a78` | 1.664 (1.635 to 1.701) | 4.95 (4.83 to 5.10) | 12 of 12 (ceiling 1.85) |

For comparison, the superseded record measured `5e2ee58` at 1.725 (1.707
to 1.744) and 5.07 microseconds per launch. In this campaign the final
wheel costs 0.155 microseconds, or 3.2%, more per launch than main. The
public version counter call alone measured 0.135 microseconds in the
superseded record.

The `2^20` ratio is 1.03 for both wheels in every run, because the kernel
time dominates there. Both wheels pass the cold (median 33.7 to 41.0 ms),
warm host (median 4.56 to 4.97 microseconds per call), and memory (85 MiB
at most) gates in every run.

## The rule and what it still catches

The rule is unchanged: the ceiling is the smallest multiple of 0.05 that is
at least 5% above the largest ratio the measured wheel showed. The largest
ratio of the final wheel is 1.701, and 1.701 times 1.05 is 1.786, which
rounds up to 1.80. The twelve runs of the final wheel span 4.0%.

With PyTorch at its median of 2.974 microseconds, the gate fails a Swage
launch slower than 5.35 microseconds at `2^18` elements. That is 0.40
microseconds, or 8.1%, above the final wheel's median of 4.95, and 11.6%
above main's median of 4.80. The warm host gate (median at most 15
microseconds) is far looser, so this ratio remains the tightest check of
the launch cost.

## Protocol

- Command: `python -m swage.bench vector-add --enforce --output FILE`, as
  the release `gpu` job and the `fixed-runtime-slo` GPU job run it, with
  `PYTHONPATH` unset, from a fresh scratch directory per run.
- Wheels: built with scikit-build-core 1.0.3 against LLVM 22.1.8 with
  `SWAGE_SOURCE_REVISION` set to the measured commit and
  `SWAGE_SOURCE_CLEAN=true`; the record names each wheel's SHA-256. Main's
  wheel is the one the superseded record measured, built from a copy of the
  tree of `22558af`. The final wheel was built from a fresh clone at
  `0ea2a78`.
- Environments: main's venv was made with `python3.13 -m venv
  --system-site-packages` (the runner's CUDA PyTorch) and the integration
  tree's hash lock as of the superseded record, before `7f9acac` locked
  cryptography 50.0.2, with the wheel installed with `--no-deps`. The final
  venv is a copy of it with only `swage-compiler` exchanged.
- Order: twelve pairs. Main ran first in odd pairs and second in even
  pairs, and `flock /tmp/swage-gpu.lock` was held for the whole campaign.
- Machine state, sampled before and after every run (48 samples) with
  `benchmarks/benchmark_provenance.py`: AMD Ryzen 9 7900X with the
  `amd-pstate-epp` driver, the `powersave` governor, and the `performance`
  energy preference on all 24 CPUs in every sample; driver 580.178.04,
  which reports CUDA 13.0; PyTorch 2.12.0+cu130; GPU temperature 57 to
  72 C; no other compute process on the GPU in any sample; one-minute load
  average 2.8 to 5.6.
- Gate verdicts in the runs: each run record carries the gates of its own
  wheel, 1.50 at `2^18` for main and 1.85 for the final wheel. The new
  ceiling comes from the measured ratios, not from those verdicts.
- Deviation from the release job: `SWAGE_CACHE_DIR` named a fresh empty
  directory per run, so the parent process did not write the user cache.
  The parent compiles before any timed section, and every cold child uses
  its own empty cache in both settings.

## Limits

These runs come from one host on one day. The ratio at `2^18` measures the
host cost of a launch, so it can move with the host. A later failure of the
gate on an unchanged tree calls for a new calibration record, not an edit
of this one.
