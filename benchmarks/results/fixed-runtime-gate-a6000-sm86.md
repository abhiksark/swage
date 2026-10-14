<!-- benchmarks/results/fixed-runtime-gate-a6000-sm86.md -->

# Fixed-runtime gate calibration on the RTX A6000

The installed-wheel release benchmark, `python -m swage.bench vector-add
--enforce`, gates the Swage/PyTorch median throughput ratio of the fixed
vector add at two sizes. This record sets the ceiling at `2^18` elements
to **1.85** (it was 1.50). The ceiling at `2^20` elements stays 1.50, and the
cold, warm host, and memory gates are unchanged. The
[raw record](fixed-runtime-gate-a6000-sm86.json) holds every run, the
machine state around it, the identity of each wheel, and the scripts that
ran them.

## Why the ceiling moved

At `2^18` elements PyTorch's add takes about 2.9 microseconds on this GPU,
less than one Swage launch, so the 32 back-to-back launches of a sample are
limited by the host and the ratio measures the host cost of a Swage
launch. Three changes to that cost, measured on the same host on
2026-10-04, account for the move:

| Wheel | Median ratio at `2^18` (range) | Swage microseconds per launch | Runs that pass 1.50 |
|---|---|---|---|
| `b7ce907`, which set the 1.50 ceiling | 1.498 (1.478 to 1.501) | 4.40 | 5 of 6 |
| main, `22558af` | 1.608 (1.596 to 1.642) | 4.74 | 0 of 18 |
| this tree, `5e2ee58` | 1.725 (1.707 to 1.744) | 5.07 | 0 of 12 |

1. **The host.** The record committed with the 1.50 ceiling,
   [fixed-runtime-a6000-sm86.json](fixed-runtime-a6000-sm86.json), measured
   1.436 on 2026-09-05 from a build whose native identity is `23fda04` with
   uncommitted changes. A clean build of `b7ce907`, the commit that added
   that record and the gate, measures 1.498 today, at the ceiling, and fails
   it once in six runs. The difference is host drift or a difference
   between that uncommitted build and `b7ce907`; the records cannot
   separate the two.
2. **Main's launch path since then.** Main's own wheel costs 0.36
   microseconds more per launch than `b7ce907` in the same campaign and
   fails 1.50 in all 18 of its runs, so the gate fails on main before this
   tree changes anything.
3. **This tree's launch path.** It costs 0.34 microseconds more per launch
   than main. Every launch now advances the version counter of its output
   for autograd correctness, so a backward pass that saved the output
   raises instead of reading overwritten values. That call alone measures
   0.135 microseconds (`torch.autograd.graph.increment_version` on a CUDA
   tensor, median of 21 batches of 100,000 calls). The rest is the other
   per-launch work of the merged warm path; this record does not split it
   further.

The `2^20` ratio is 1.03 for every wheel in every run, because the kernel
time dominates there. Every wheel passes the cold (median 32 to 38 ms),
warm host (median 4.3 to 5.0 microseconds per call), and memory (85 MiB at
most) gates in every run. The main row of the table gives the median and
range of the gate campaign; main's six baseline runs fall within that
range (1.597 to 1.633).

## The rule and what it still catches

The ceiling is the smallest multiple of 0.05 that is at least 5% above the
largest ratio this tree's wheel measured: 1.744 times 1.05 is 1.831, which
rounds up to 1.85. The 5% covers more than twice the range of this tree's
twelve runs (2.2%).

With PyTorch at its median of 2.944 microseconds, the gate fails a Swage
launch slower than 5.45 microseconds at `2^18` elements. That is 0.37
microseconds, or 7.3%, above this tree's median of 5.07, and 15% above
main's median. A regression of the size this tree adds over main (7.1%)
sits at the edge of what the gate catches. The warm host gate (median at
most 15 microseconds) is far looser, so this ratio remains the tightest
check of the launch cost.

## Protocol

- Command: `python -m swage.bench vector-add --enforce --output FILE`, as
  the release `gpu` job and the `fixed-runtime-slo` GPU job run it, with
  `PYTHONPATH` unset, from a fresh scratch directory per run. `b7ce907`
  predates that module, so it ran its own
  `benchmarks/benchmark_fixed_runtime.py --enforce --output FILE`, which
  measures the same workload.
- Wheels: built with scikit-build-core 1.0.3 against LLVM 22.1.8 with
  `SWAGE_SOURCE_REVISION` set to the measured commit and
  `SWAGE_SOURCE_CLEAN=true`; the record names each wheel's SHA-256. Each
  ran in a venv made with `python3.13 -m venv --system-site-packages` (the
  runner's CUDA PyTorch) and this tree's hash lock, with the wheel installed
  with `--no-deps`. The `b7ce907` venv is a copy of main's with only the
  wheel exchanged, because no network was available for a new venv.
- Order: the gate campaign ran twelve pairs of main and this tree, and the
  baseline campaign six pairs of `b7ce907` and main. The first wheel of a
  campaign ran first in odd pairs and second in even pairs, and
  `flock /tmp/swage-gpu.lock` was held for each campaign.
- Machine state, sampled before and after every run with
  `benchmarks/benchmark_provenance.py`: AMD Ryzen 9 7900X with the
  `amd-pstate-epp` driver and the `powersave` governor on all 24 CPUs in
  every sample; a driver that reports CUDA 13.0; PyTorch 2.12.0+cu130; GPU
  temperature 51 to 69 C; no other compute process on the GPU in any
  sample; one-minute load average 0.2 to 7.4 (the highest values follow
  the wheel builds that preceded the gate campaign).
- Deviation from the release job: `SWAGE_CACHE_DIR` named a fresh empty
  directory per run, so the parent process did not write the user cache.
  The parent compiles before any timed section, and every cold child uses
  its own empty cache in both settings.

## Limits

These runs come from one host on one day. The ratio at `2^18` measures the
host cost of a launch, so it can move with the host; a later failure of the
gate on an unchanged tree calls for a new calibration record, not an edit
of this one.
