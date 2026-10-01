<!-- docs/internals/benchmarks.md -->

# Benchmarks

!!! warning "Recorded evidence"

    This page reports recorded measurements from one benchmark campaign
    on one machine. It is not a continuously enforced gate and not a
    public performance contract.

## Environment and provenance

Unless a section names another campaign, numbers below were recorded on
2026-08-27 on an NVIDIA GeForce RTX 5090 (`sm_120`), driver 580.173.02,
CUDA 13.0, PyTorch 2.13.0+cu130, and Triton 3.7.1, on a co-tenant GPU. The
committed snapshot
[`benchmarks/results/perf-5090-sm120.json`](https://github.com/abhiksark/swage/blob/main/benchmarks/results/perf-5090-sm120.json)
records the aggregated medians, quartiles, and per-number provenance.
Each value is the median of three independent process runs unless its
provenance field states otherwise. The Triton baseline is the per-segment
kernel at the best of the swept and autotuned configurations.

## Timing methods

Three timing methods separate host dispatch cost from kernel quality:

- `call_us` is the synchronized wall clock per call, what a plain
  Python loop pays;
- `kernel_us` places CUDA events around 32 back-to-back launches, so
  the host launcher stays visible for short kernels;
- `graph_us` replays the same 32 launches from a captured CUDA graph,
  removing the host and isolating kernel quality.

<div class="doc-figure" tabindex="0" markdown="1">

![Timelines contrasting per-call wall clock, batched event timing, and graph replay](../assets/figures/timing-methods.svg)

</div>

*How each timing method sees dispatch and kernel time. [Open the full-size figure](../assets/figures/timing-methods.svg).*

## Harness methods

The scripts under `benchmarks/` are research harnesses, not CI gates. This
section states what they measure and what they write into a record. It
describes the harnesses as they are now. The recorded numbers on this page
come from earlier revisions of the harnesses, so a committed record carries
one of the fields below only if the harness wrote it at the time.

### What each harness measures

| Harness | Regime | Timed region |
|---|---|---|
| `benchmark_fresh_offsets.py` | Every iteration takes an offsets layout that no earlier iteration used. | From offsets in to result out: preparation, launch, and synchronize, on the host clock. |
| `benchmark_triton_comparison.py` | One layout per row is prepared once and launched repeatedly. | One launch, by the three timing methods above. Preparation and compilation are outside. |
| `benchmark_mixed_sum.py` | The frozen mixed-policy gate of ADR-0015. | One launch per policy between two CUDA events, in rotated order. |
| `benchmark_persistent_sum.py` | The frozen persistent gate of ADR-0018. | One launch per policy between two CUDA events, in rotated order. |
| `benchmark_composable_reductions.py` | Exploratory sum and max with element programs. | One preparation sample per row, then launches by call and graph timing. |

The fresh-offsets harness and the comparison harness time these candidates:

- The private planned Swage sum. Fresh offsets times the mixed policy with
  its preparation. The comparison times the warp, CTA, and mixed policies.
- `torch.segment_reduce` on the device offsets, with its output allocation.
- A pad-to-max baseline in pure PyTorch: every segment is padded with zeros
  to the longest one, and the masked matrix is summed per row. Fresh offsets
  times the padding with the sum. The comparison pads outside the timed
  launch.
- The looped Triton sum: one program per segment that walks the segment in
  fixed blocks. Every block and warp configuration of the sweep is timed and
  none is selected.
- In the comparison only, the fixed Triton sum and the planned Triton sum,
  which read one block per segment or task.

A baseline that cannot produce a correct sum on a row is not timed. The row
lists it under `skipped` with the reason. The fixed and planned Triton
baselines are skipped when the longest segment exceeds their block. The
pad-to-max baseline is skipped when padding does not fit the free device
memory, and the row then records the bytes it would need.

Triton is imported only when a harness runs. It is not a dependency of the
project, and fresh offsets leaves the looped candidates out when Triton is
not installed.

### Options

Both harnesses take the scale and the inputs as options. Neither names a
device or a target: they run on the current CUDA device, which
`CUDA_VISIBLE_DEVICES` selects, and derive the target from it.

| Option | Fresh offsets | Comparison |
|---|---|---|
| `--distributions` | Rows to run. The default is all nine. | Rows to run. The default is the seven of the recorded campaign; `alternating-empty` and `power-law` run when named. |
| `--segment-count` | Segments per layout. The default is 32,768. | Segments per row. The default is 32,768. |
| `--seed`, `--seeds` | `--seed` seeds the first layout, the values, and the candidate orders. Layout `i` of a row uses `seed + i`. | `--seeds` runs one row per distribution and seed. |
| `--values` | `quarters` (default) or `normal`. | `ones` (default), `quarters`, or `normal`. |
| `--samples`, `--warmups` | Timed and untimed fresh layouts per row. | Timed samples and warmup launches per candidate and method. |

A segment count is accepted when the largest total the distribution can
reach fits signed 32-bit offsets. One million segments fit `bimodal`,
`few-huge`, `one-outlier`, `many-tiny`, `alternating-empty`, and `power-law`.
They do not fit `uniform`, `log-normal`, and `zipf-like`, whose every length
can be 4096.

The value kinds are:

- `ones`: every value is one.
- `quarters`: seeded multiples of 0.25 from 0.25 to 1.75. No value is zero
  and neighbours differ, so a missing, extra, or shifted element changes a
  sum.
- `normal`: seeded standard normal values.

### Correctness check

Every timed candidate is checked against a float64 CPU reference on the
timed values. Fresh offsets checks every candidate in every iteration,
warmups included. The comparison checks every candidate once before timing.

The comparison is exact wherever f32 sums of the values are exact in any
order: all values lie on one grid and their magnitudes add up to at most
2^24 grid steps. Elsewhere a sum may differ from the reference by at most
`gamma(n - 1)` times the sum of magnitudes, with `gamma(k) = k u / (1 - k u)`
and `u = 2^-24`. This bound holds for `n` f32 values added in any order, so
it does not depend on the candidate. Beyond about 8.4 million elements the
bound says nothing, and such a segment is only required to have been
written. Each row counts its exact, bounded, and unchecked segments in
`check`.

With `normal` values the bound grows with the square of the segment length,
so the check detects a misplaced element in short segments and loses that
power in long ones. `quarters` keeps the exact check for every segment of up
to 2.3 million elements.

### Effective rate and timer resolution

Each row reports `effective_gb_per_s` beside every time. It is the bytes a
correct result has to move, divided by the time: for a segmented sum the f32
values and the i32 offsets read and the f32 sums written, for vector add two
inputs read and one output written. A baseline that moves more bytes, such
as a padded matrix, is rated by the same bytes.

The fresh-offsets and comparison harnesses also report the timer tick as a
fraction of one sample:

- The host clock tick is the smallest advance of back-to-back
  `time.perf_counter_ns` reads in the process.
- The CUDA event tick is the step that event readings favour. Readings are
  multiples of a fine step, but most intervals around a kernel are multiples
  of a coarser one, and the coarser step is what limits a sample. The
  harness measures it in the process and does not assume a value.

An event-timed sample is a batch of launches. The batch starts at 32. While
one tick is not below one percent of the median sample, the batch doubles
and the samples are taken again. Each timing records `launches_per_sample`,
`timer_tick_us`, and `tick_fraction_of_sample`. A fresh-offsets sample is
one call on one fresh layout and is never a batch, because a second call on
the same layout would not be fresh; the row records the tick fraction of
each candidate.

A ratio of two medians carries no more digits than the tick fraction and
the spread across processes support.

### Provenance

Every harness writes a `provenance` block:

| Field | Content |
|---|---|
| `gpu`, `gpu_uuid` | Device name from PyTorch and its UUID. |
| `cpu_model` | Processor model from `/proc/cpuinfo`. |
| `pytorch`, `triton` | Library versions. `triton` is null when it is not installed. |
| `swage`, `llvm_pin`, `llvm_linked` | Package version, the pinned LLVM tag, and the LLVM version the native extension was linked against. |
| `cuda_driver` | CUDA driver API version from `cuDriverGetVersion`. |
| `native_sha256` | SHA-256 of each native library file that produces PTX. |
| `loaded_ptx` | Kernel name, SHA-256, and size of every PTX module the process loaded, taken when the module was loaded. |
| `gpu_state_before`, `gpu_state_after` | `nvidia-smi` samples around the measurement: NVIDIA driver version, compute mode, performance state, temperature, power draw and limit, clocks, utilization, and the other compute processes on the device. |
| `other_compute_process_seen` | True when a sample lists another compute process, false when both samples are empty, null when a sample could not be read. |

A fact that cannot be read is recorded as null with the reason and never
fails the run, so an unreadable GPU is not reported as an exclusive one.
The source revision and the worktree state stay in `source`. A full
fresh-offsets record is refused when the worktree is dirty, when the
imported package belongs to another checkout, or when the native library
hashes or the loaded PTX hashes are missing.

### Independent processes

Samples from one process share its allocator state and its compiled kernels,
so their spread understates how far a median moves between runs.
`benchmark_processes.py` runs one harness command in several fresh
processes, one after another. The default is five.

```bash
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
python benchmarks/benchmark_processes.py \
  --processes 5 \
  --output-dir "$OUT/fresh-offsets-power-law" \
  -- benchmarks/benchmark_fresh_offsets.py --distributions power-law
```

The output directory must be new or empty and outside the checkout, because
a record written inside it would make the worktree dirty for the next
process. The driver passes `--output` itself. Each process writes
`process-<k>.json`, and `summary.json` then holds, for every row, timing
method, and candidate:

- the median of each process,
- the median, minimum, and maximum of those per-process medians,
- the same three for the ratio to the reference candidate (`--reference`,
  default `torch`), formed inside each process from its two medians,
- the same three for the effective rate.

Every candidate is reported and none is selected. A candidate that only
some processes timed is listed under `incomplete` and is not combined. The
driver refuses to summarize processes that differ in revision, device,
library versions, native library hashes, or loaded PTX hashes. It reads the
records of the fresh-offsets and comparison harnesses.

Several seeds are several configurations: run the driver once per `--seed`
for fresh offsets, or pass `--seeds` to the comparison.

### Gate scripts on another GPU

The two gate scripts keep their frozen inputs, timing loop, and decision.
Each gate is decided on its declared device. With `--any-device` the same
measurement runs on another GPU, the persistent run requests two resident
blocks per SM of that GPU, and the record is labelled
`rerun on another device; not gate evidence`.

A gate record also describes its samples without changing the decision:

- `position_matched` gives each policy's median at each place of the
  rotation and the gate ratio at each place. The first policy of a rotation
  starts on an idle stream, so its interval includes host dispatch; the
  later ones queue behind a running kernel.
- `timer` gives the observed tick, each median in ticks, and the range of
  the gate ratio that half a tick on each median allows.

## Persistent tail-skew gate

A separate 2026-09-02 campaign on the NVIDIA RTX A6000 (`sm_86`) evaluated
the frozen gate in
[ADR-0018](../adr/ADR-0018-private-persistent-task-queue.md). Its 32,768
segments contain 32,767 seeded lengths in `[1, 32]` and one final
16,777,216-element outlier. Both policies consume the same host-materialized
warp, partial, and merge plan. Timed static execution contains fused direct,
split partial, and split merge kernels; timed persistent execution contains
its counter reset and one 168-block resident kernel.

Adversarial testing invalidated the original 115.712-microsecond run: a final
publisher could observe completion before another CTA's scratch store was
globally visible. Compute Sanitizer then invalidated the first fenced run by
finding a shared queue-claim race at the CTA-to-partial phase handoff. Both raw
records remain in the repository, but neither is qualification evidence.

With publication fences and the phase barrier, the clean persistent median was
117.520 microseconds and static mixed measured 118.784 microseconds.
Persistent was 1.06% faster, but the 0.9894 ratio failed the predeclared
`persistent <= 0.95 * static_mixed` gate. Persistent qualification therefore
remains incomplete; this is an honest near miss, not a performance success.
The canonical
[raw record](https://github.com/abhiksark/swage/blob/main/benchmarks/results/persistent-sum-a6000-sm86.json)
and all earlier clean runs preserve every sample and source revision. Runs
that predate either synchronization fix are explicitly excluded from semantic
qualification.

## Segmented sum under graph timing

Under graph replay, the best Swage policy per distribution beats
`torch.segment_reduce` on all seven distributions and the Triton baseline on
six of seven. The snapshot records that baseline as a per-segment kernel
(`BLOCK=256`, `num_warps=8`, raw-log implementation name `triton-naive`), and
the campaign's raw log is not committed. The later
[A6000 comparison study](a6000-comparison.md), on a different GPU and
distribution set, adds a matched Triton scheduler that receives the same
heterogeneous tasks. There Swage is 16.2% faster on one distribution, within
1.3% on five, and matched Triton is 8.3% faster on one. The uniform-4k
row is parity, nominally
Triton (2.5 versus Swage's 2.6 microseconds). The bimodal and few-huge Swage
bars time one captured mixed sequence, the planner's fused warp launch
plus the 512-thread split kernels; their provenance fields in the
snapshot record the single-sequence caveat.

<div class="doc-figure" tabindex="0" markdown="1">

![Grouped bars of graph-replay medians for Swage, Triton, and torch across seven distributions](../assets/figures/segsum-graph-comparison.svg)

</div>

*Segmented sum graph-replay medians per distribution; lower is better. [Open the full-size figure](../assets/figures/segsum-graph-comparison.svg).*

## Dispatch cost

The campaign reduced warm per-launch dispatch from 7714.5 to 36.1
microseconds by caching the compiler identity and skipping emission on
cache hits, then to 24.7 microseconds through the compiled nanobind
launcher. Triton's compiled-C launcher measures 20.4 microseconds and
torch dispatch about 14, so pure dispatch narrowed but was not won.
Cold start went the other way: the first vector-add launch in a fresh
process took 144 milliseconds for Swage against 1116 milliseconds for
Triton's autotuning stack.

<div class="doc-figure" tabindex="0" markdown="1">

![Log-scale bars following warm dispatch cost across the campaign stages](../assets/figures/dispatch-ladder.svg)

</div>

*The warm dispatch ladder and the cold-start comparison. [Open the full-size figure](../assets/figures/dispatch-ladder.svg).*

## Honest losses

- Pure warm dispatch stays with Triton (20.4 versus 24.7 microseconds)
  and torch (about 14).
- uniform-4k segmented sum is parity, nominally Triton (2.5 versus
  2.6 microseconds).
- Vector add under graph timing shows two stable Swage losses to
  Triton: n = 2^18 (1.38 versus 1.19 microseconds, about 16 percent)
  and n = 2^20 (3.17 versus 2.49, about 27 percent); torch also leads
  Swage at both sizes. The other measured sizes are within a few
  percent, including an 8 percent Swage edge at n = 2^16 that stays
  below the snapshot's win bar of at least 10 percent over the best
  baseline. The snapshot records the full sweep; Swage claims no
  vector-add win.

Continue with the [A6000 comparison study](a6000-comparison.md) for a
separate exploratory Swage/Triton campaign across skewed segment
distributions, [Persistent Execution](persistent-execution.md) for the failed
resident-queue qualification, [Verification](verification.md) for the
executable proof behind each boundary, or
[Task Execution](task-execution.md) for the execution contracts these
measurements exercise.
