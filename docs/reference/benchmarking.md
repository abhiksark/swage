<!-- docs/reference/benchmarking.md -->

# Benchmarking

Native wheels ship one operational benchmark entry point:

```bash
python -m swage.bench vector-add --output fixed-runtime-slo.json
```

This is the frozen CUDA **float32 vector-add** workload, not a general kernel
benchmark API. The source tree also supports public fixed-vector multiplication
and low-precision execution, but neither is a benchmark selector. Segmented
comparison remains source-only private qualification; this CLI does not expose
segmented launch. No benchmark helper is added to the package exports or public
type stubs.

## Prerequisites

Install a native `swage-compiler` wheel containing the self-contained private
`mlir_swage` compiler, alongside CUDA-enabled PyTorch and an accessible NVIDIA
GPU and driver. Follow [Installation](../getting-started/installation.md) and
the [support matrix](support-matrix.md). A
frontend-only editable installation is not sufficient. The v0.5.2 source
contract remains pending publication and release qualification.

Run measurements outside the checkout, with `PYTHONPATH` unset, so `swage`
resolves from the installed wheel. The benchmark rejects a nonempty
`PYTHONPATH` or a `swage` package outside the interpreter's site-packages.
For example, from an external working directory:

```bash
env -u PYTHONPATH python -m swage.env --json --check cuda
env -u PYTHONPATH python -m swage.bench vector-add \
  --output fixed-runtime-slo.json
```

Help does not import PyTorch or `mlir_swage` and works without either optional
runtime dependency:

```bash
python -m swage.bench --help
python -m swage.bench vector-add --help
```

## Command and exit behavior

The public syntax is exactly:

```text
python -m swage.bench vector-add --output PATH [--enforce]
```

`--output` is required. There are no workload, sample-count, dtype, backend,
or threshold tuning options. A missing command or output, an unknown command
or option, or a segmented selector is a usage error with exit status **2**.
Help exits **0** without measuring anything.

The command creates the output parent directories and writes a sorted,
indented JSON record to `PATH`, replacing an existing file there. It then
prints a compact, key-sorted JSON summary to stdout containing `output`,
`passed`, and `valid`.

- Without `--enforce`, a valid measurement exits **0** even if a timing gate
  fails; inspect `passed` and the individual gates.
- With `--enforce`, a valid measurement exits **0** only if every frozen gate
  passes. Enforcement requires the device name **exactly NVIDIA RTX A6000**
  and target **exactly `sm_86`**; another admitted NVIDIA target is not
  interchangeable qualification evidence.
- Invalid measurement or correctness failure exits **1**, with or without
  enforcement. Caught execution errors retain the partial record, including
  an `error` object and any measurements already collected. Failed gates
  also retain their raw evidence. This retention does not cover parser
  errors, process termination, or an unwritable output destination.

## Frozen evidence contract

Records retain `schema_version: 1` and `benchmark: "fixed-runtime"`, the fixed
configuration, timestamp, enforcement and hardware-qualification flags,
correctness flags, raw measurements, and gates. Environment and native build
identity are recorded as execution reaches those checks. `qualified` denotes
only the exact A6000/`sm_86` hardware match, not release qualification;
`valid` denotes complete correct evidence, and `passed` denotes the timing
and memory gate result.

The workload and inclusive thresholds remain frozen:

| Measurement | Method | Passing threshold |
|---|---|---|
| Cold compile/load/first synchronized CUDA launch | Five fresh child processes and unique empty caches; `n=129`, `BLOCK=128` | Median <=250 ms; maximum <=400 ms |
| Warm host dispatch | `n=129`, `BLOCK=128`; 200 warmups and 20 batches of 500 launches, synchronized at batch boundaries | Median <=15 microseconds/call; p95 <=20 microseconds/call |
| Large-vector throughput | `n=2^18` and `2^20`, `BLOCK=256`; 25 warmups and 100 rotating interleaved CUDA-event samples of 32 launches, against `torch.add(out=...)` | Swage/PyTorch median ratio <=1.50 at each size |
| Native compiler memory | Linux `/proc/self/status` RSS increase from PyTorch/tensor setup to first compile/launch | Maximum <=512 MiB |

Correctness preflights precede every timing section. Cold children are
re-entered through this same installed module. Compilation, loading, and
first launch remain one cold measurement; there is no invented planning
field or segmented evidence in this schema.

A passing benchmark is one item of evidence, **not independent release
qualification**. The trusted release workflow must still qualify the actual
repaired installed artifact, its source/build identity, CPU and CUDA
correctness, persistent cache reuse, reproducibility, security, and publication
gates. Historical addition results do not qualify multiplication or
low-precision performance. See [Verification](../internals/verification.md)
for the complete release boundary and retained evidence.

## Source-tree comparison harness

The source tree also holds a research comparison of the private Swage paths
with Triton and PyTorch baselines. It is not part of the wheel, it is not a
CI gate, and its records are not release qualification. Triton is imported
only when the comparison runs; it is not a Swage dependency.

`benchmarks/benchmark_triton_comparison.py` writes one record per process.
Run it from a checkout with the native build, CUDA-enabled PyTorch, and
Triton available. The segmented-sum suite measures compilation first, so
`SWAGE_CACHE_DIR` and `TRITON_CACHE_DIR` must name two distinct empty
directories:

```bash
export PYTHONPATH="$PWD/python:$PWD/build/python_packages"
SWAGE_CACHE_DIR="$(mktemp -d)" TRITON_CACHE_DIR="$(mktemp -d)" \
python benchmarks/benchmark_triton_comparison.py --output result.json \
  --suite segmented-sum --distributions uniform power-law --seeds 7 11
```

The device is the current CUDA device; select another with
`CUDA_VISIBLE_DEVICES`.

| Option | Default | Effect |
|---|---|---|
| `--output PATH` | required | Record path. |
| `--suite` | `all` | `vadd`, `segmented-sum`, or both. |
| `--samples` | `100` | Timed rounds per timing method. |
| `--warmups` | `25` | Untimed rounds before each timing method. |
| `--distributions` | the seven synthetic distributions and `soc-epinions1-outdegree-v1` | Segmented-sum rows, in run order. `alternating-empty` and `power-law` run only when named. |
| `--segment-count` | `32768` | Segments of each synthetic row. The trace has 32,768 segments, so another count requires a list of distributions without it. |
| `--seeds` | `7` | One row per distribution and seed. The seed draws the lengths of a synthetic row and the values of any row; the first seed draws the vector-add inputs. |
| `--values` | `ones` | Timed values: all ones, seeded quarter multiples from 0.25 to 1.75, or seeded standard normal values. |
| `--candidates` | every candidate | Candidates or families to time. Requires `--suite segmented-sum`. |
| `--exclude-candidates` | none | Candidates or families to leave out. Requires `--suite segmented-sum`. |

The harness refuses a configuration whose offsets could exceed i32, a
repeated distribution or seed, and a filter name that matches no candidate
or family.

### Segmented-sum candidates

| Family | Candidates | Runs on a row when |
|---|---|---|
| Swage policies | `swage_warp`, `swage_cta`, `swage_mixed` | Always. |
| `torch_segment_reduce` | `torch_segment_reduce` | Always. |
| `torch_padded` | `torch_padded` | The matrix that pads every segment to the longest one fits the free device memory. |
| `triton_fused` | `triton_fused` | The longest segment has at most 4,096 elements. |
| `triton_fixed` | `triton_b{block}_w{warps}` | The block covers the longest segment. |
| `triton_matched_task_partition` | the base name and `_w2`, `_w4`, `_w8` | The longest segment has at most 4,096 elements. |
| `triton_looped` | `triton_looped_b{block}_w{warps}`, 15 configurations | Always. |
| `triton_planned_looped` | `triton_planned_looped_b{block}_w{warps}`, 15 configurations | Always. |

`triton_looped` runs one program per segment that loops over the segment in
fixed blocks. `triton_planned_looped` uses the task lists of
`triton_matched_task_partition`, packs the short tasks four per program, and
loops over each longer segment with the block and warp sweep of
`triton_looped`. `torch_padded` pads the segments with zeros outside the
timed launch and sums each row into a preallocated output inside it.

A row records a candidate it cannot run under `skipped`, with the reason,
and a candidate the filter left out under `excluded`. `candidate_order`
lists the timed candidates.

### Measurement

- Candidates are timed in deterministic rotating order. Each timing method
  has its own warmup and timed rounds.
- `call` is one synchronized Python call per sample. `batched_event` and
  `graph` start each sample as a batch of 32 launches, timed with CUDA events
  or replayed from a captured graph. Every graph of a pass is captured before
  the pass replays any of them.
- Each process measures the `time.perf_counter_ns` tick and the CUDA event
  tick. While one event tick is not below one percent of a candidate's
  median sample, that candidate's batch doubles and every candidate is
  sampled again in new rotating rounds. Every timing entry records
  `launches_per_sample`, `timer_tick_us`, and `tick_fraction_of_sample`.
- Every timing entry reports `effective_gb_per_s`: the bytes a correct
  result must move, divided by the median time.
- Before any timing, every candidate is checked against a float64 CPU
  reference on the timed values: exactly where the sums are exact in f32 in
  any order, otherwise within the bound for any summation order. Every
  output starts as NaN, so a sum that was never written fails. Every timed
  Triton candidate is also run on seeded quarter multiples, which expose a
  shifted, short, or long read that all-one values hide.
- The segmented-sum suite first compiles every Swage kernel and every Triton
  specialization that a timed candidate launches, and times each one. It then
  times planning and complete warm operations of `swage_mixed` and
  `triton_matched_task_partition` where the row times them, with fixed and
  with changing offsets.

Records carry `schema_version: 2`. `benchmarks/benchmark_campaign.py`
validates a record and recomputes every summary from its raw samples.

### Independent processes

`benchmarks/run_triton_comparison_campaign.py` runs a harness in several
fresh processes, one after another. Each process gets its own empty Swage and
Triton cache directories outside the checkout, and the driver records
`nvidia-smi` observations before and after each process. It requires a clean
worktree and an output directory that is new or empty and lies outside the
checkout or in a directory that Git ignores. Exactly one of
`--exclusive-gpu-allocated` and `--allow-shared-gpu-engineering` is
required:

```bash
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
python benchmarks/run_triton_comparison_campaign.py \
  --output-dir /tmp/swage-comparison \
  --exclusive-gpu-allocated --archival-source \
  --repetitions 5 --suite segmented-sum --samples 100 --warmups 25 \
  -- --distributions many-tiny power-law --values quarters
```

The driver passes `--suite`, `--samples`, `--warmups`, and `--output` to
the comparison harness and forwards the options after `--`. The output
directory then holds:

| File | Content |
|---|---|
| `process-NNN.json` | The record of each process, from `process-000.json`. |
| `manifest.json` | The controls, the observations at each process boundary, the archival eligibility, and the median of the process medians of every measurement, validated again from the raw records. |
| `summary.json` | For every row, timing method, and candidate, the process medians with their median, minimum, and maximum, and the same for the ratio to each reference candidate. |

`--reference` names the reference candidates. Without it the driver uses
`torch` and `torch_segment_reduce`, each where the records time it.
`--harness benchmarks/benchmark_fresh_offsets.py` repeats the fresh-offsets
harness instead; all of its options follow `--`, and the driver writes the
process records and `summary.json`.

`--summarize DIR` summarizes the records of a finished run again, for
example against another reference, without running anything. It writes
`DIR/summary-<references>.json` and never replaces a summary. It also reads
records of earlier revisions, which are named `process-1.json` onward.

Exact reproduction of a committed record requires the revision it names.
The record `benchmarks/results/segmented-sum-a6000-sm86-453c56e` was
produced at that revision by the driver of that revision; its summary page
lists the commands. `benchmarks/campaign_tables.py` regenerates that page
and its documentation fragments from the committed summaries, and
`python benchmarks/campaign_tables.py --check` verifies them and the digests
of the compressed process records.
