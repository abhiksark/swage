<!-- docs/internals/a6000-comparison.md -->

# Swage and Triton on Ragged Reductions

This study asks a narrow question: **when segment lengths vary at runtime, is
it useful to derive different fixed GPU tasks from one segment-local
program?** On an NVIDIA RTX A6000, the answer is yes—but a matched Triton
scheduler shows that task derivation, rather than an inherent code-generation
advantage, explains most of the result. Swage remains competitive while
making that scheduling model part of its compiler architecture.

!!! warning "Exploratory evidence, not a release claim"

    This page reports one local campaign from clean commit `80f222d` on one
    GPU. It is not a trusted GPU qualification, continuously enforced gate,
    or public performance contract. Segmented execution is private
    contributor machinery, not public Swage API. The complete local record is
    [`benchmarks/results/swage-triton-a6000-sm86.json`](https://github.com/abhiksark/swage/blob/main/benchmarks/results/swage-triton-a6000-sm86.json).

## Executive result

Against a swept **fixed-shape** Triton kernel, Swage is faster on six of seven
segmented distributions. The largest graph-replay differences are 6.60x on
`one-outlier`, 2.58x on `many-tiny`, and 1.90x on `few-huge`.

Against a **matched heterogeneous Triton scheduler** with the same 32-element
cutoff, packed four-warp programs, and separate CTA tasks, the result narrows:

- Swage is 16.2% faster on `log-normal`;
- Swage and matched Triton are within 1.3% on `uniform`, `bimodal`,
  `zipf-like`, `few-huge`, and `one-outlier`; and
- matched Triton is 8.3% faster on `many-tiny` under graph replay.

This is the central finding: **heterogeneous task derivation matters, and both
compiler stacks benefit when given it**. Swage's interesting property is that
one semantic segment program already lowers through this planning model; the
result is not evidence that Triton cannot express an equivalent schedule.

The fixed vector-add control is also close under graph replay, with Swage
between parity and 7.0% behind Triton across the sweep. Host-visible calls
still favor Triton because Swage's Python launch path costs more.

## What is being compared

All segmented implementations compute an identity f32 sum from one packed
values array and an i32 offsets array. Compilation, classification,
allocation, and module loading happen before timing. Every output is checked
exactly against the known all-one result before samples are collected.

### Swage

The same semantic segmented-sum module is classified from runtime offsets:

- lengths from 0 through 32 become 32-thread warp tasks;
- lengths from 33 through 4096 become 128-thread CTA tasks;
- longer segments would become 4096-element partial tasks plus a merge.

The `mixed` policy executes short and CTA-sized direct tasks in one fused
128-thread kernel. Each initial block contains four independent warp slots;
CTA tasks follow at one segment per block. See [Task Planning](planning.md),
[Task Execution](task-execution.md), and [Split Execution](split-execution.md)
for the exact private contracts.

### Triton

The campaign includes two Triton baselines.

The **fixed-shape baseline** assigns one program to each segment. A program
loads its begin and end offsets, masks a fixed power-of-two block, reduces the
loaded values, and stores one result. The harness sweeps legal combinations
from:

```text
BLOCK = 32, 64, 128, 256, 512, 1024, 2048, 4096
num_warps = 1, 2, 4, 8
```

Blocks smaller than the distribution's maximum segment length are excluded.
The best measured configuration is selected separately for each distribution
and timing method.

The **matched planned baseline** uses the same host lengths and 32-element
cutoff as Swage. It builds stable warp and CTA task-ID tensors before timing.
One Triton program handles four short tasks as a 4x32 reduction, matching the
four warp slots in Swage's fused block. A separate B4096 CTA-task kernel
handles long segments, with `num_warps` swept from 1 through 8. It needs one
launch when only short work exists and two launches for mixed work, whereas
Swage fuses direct warp and CTA work into one launch.

This matched baseline is still hand-written benchmark code, not Triton
compiler automation. It demonstrates that Triton can express the scheduling
strategy and separates the value of task derivation from the value of a
particular kernel language.

### PyTorch

`torch.segment_reduce(values, "sum", offsets=offsets)` is the framework
baseline. PyTorch 2.12 does not expose an `out=` argument for this operation,
so its measured Python call includes output allocation. Batched CUDA events
primarily expose device work but do not erase that semantic difference.

## Test distributions

Each distribution contains 32,768 segments generated with seed 7. Total work
and skew vary substantially.

| Distribution | Total values | Median | p95 | Maximum | Shape |
|---|---:|---:|---:|---:|---|
| many-tiny | 525,034 | 16 | 31 | 32 | Every segment fits one warp |
| uniform | 67,091,023 | 2,037 | 3,892 | 4,096 | Broadly CTA-sized |
| log-normal | 3,153,289 | 32 | 383 | 4,096 | Short center, long tail |
| bimodal | 8,835,812 | 18 | 2,564 | 4,095 | 90% short, 10% long |
| zipf-like | 6,697,989 | 8 | 1,331 | 4,093 | Many short, heavy tail |
| few-huge | 4,272,894 | 2 | 4 | 4,095 | 95% tiny, 5% large |
| one-outlier | 544,458 | 16 | 31 | 4,096 | One large segment |

The `uniform` case has over 15x more values than `few-huge`. Absolute latency
therefore should not be compared across rows as if every distribution had the
same amount of work. Ratios within a row are the meaningful comparison.

## Segmented-sum results

The primary table reports graph-replay median microseconds per semantic
launch. Each graph contains 32 launches and is replayed 100 times. Graph
replay removes Python dispatch and exposes the scheduled GPU work. Lower is
better. Each column selects its best measured policy or configuration.

| Distribution | Swage µs | Fixed Triton µs | Planned Triton µs | PyTorch µs | Swage / planned |
|---|---:|---:|---:|---:|---:|
| many-tiny | 8.480 | 21.850 | **7.776** | 51.360 | 1.091 |
| uniform | 386.336 | **379.520** | 383.136 | 382.013 | 1.008 |
| log-normal | **36.320** | 68.096 | 43.360 | 71.326 | **0.838** |
| bimodal | **60.760** | 72.416 | 61.056 | 83.584 | 0.995 |
| zipf-like | **48.762** | 71.936 | 48.992 | 80.192 | 0.995 |
| few-huge | **37.175** | 70.556 | 37.627 | 68.160 | 0.988 |
| one-outlier | 10.072 | 66.519 | **9.949** | 54.944 | 1.012 |

A ratio below 1 favors Swage. Comparing Swage only with fixed Triton makes the
planning result look like a language result. Planned Triton closes nearly all
of that gap. The remaining `log-normal` difference is the strongest kernel
result in this campaign; the other mixed distributions are effectively
parity under graph replay.

Batched CUDA events retain launcher submission while amortizing it over 32
calls. They show the same broad picture, with Swage at 0.840x planned Triton
on `log-normal`, between 0.967x and 1.004x on four other distributions,
1.072x on `many-tiny`, and 0.918x on `one-outlier`.

## Why the mixed policy helps

### It avoids maximum-length provisioning

For the one-program Triton baseline, a 4096-element maximum requires a
4096-lane logical block even when almost every segment contains only a few
elements. Masking preserves correctness, but most lanes perform no useful
loads. The `one-outlier` distribution is the cleanest demonstration: one
segment changes the fixed block required by all 32,768 programs.

Swage instead uses runtime lengths to derive fixed tasks. The large segment
gets CTA work while the remaining segments get warp work. Runtime segment
identity stays in SSA values and task IDs rather than types, preserving the
semantic program.

### It amortizes short-segment scheduling

The fused mixed kernel places four independent short segments into four warp
slots of one 128-thread block. It therefore avoids launching one full CTA per
tiny segment and avoids a second kernel launch between direct warp and CTA
work.

The `many-tiny` row is important here. Its maximum is only 32, so fixed
Triton can already select a small block but still launches one program per
segment. Packing four tasks per planned Triton program changes graph time from
21.850 to 7.776 microseconds and slightly beats Swage's 8.480 microseconds.
This isolates packed task organization as the source of the large gain.

### It retains a sensible uniform path

On uniformly distributed lengths through 4096, pure CTA is Swage's best
policy and is within 0.5% of the best Triton result. Classification does not
create a win when the workload has little exploitable shape separation, but
the selected homogeneous policy does not materially lose either.

## The vector-add control

Vector add uses the public fixed-block Swage path and direct equivalents in
Triton and PyTorch. Triton blocks 128, 256, 512, and 1024 are swept. These are
graph-replay medians.

| Elements | Swage µs | Best Triton µs | PyTorch µs | Swage / Triton |
|---:|---:|---:|---:|---:|
| 1,024 | **1.056** | **1.056** | 1.215 | 1.000 |
| 4,096 | 1.088 | **1.016** | 1.216 | 1.070 |
| 16,384 | 1.083 | **1.056** | 1.184 | 1.026 |
| 65,536 | 1.280 | **1.270** | 1.376 | 1.008 |
| 262,144 | 1.984 | **1.920** | 2.105 | 1.033 |
| 1,048,576 | 18.496 | **17.696** | 17.664 | 1.045 |
| 4,194,304 | 76.012 | **74.943** | 75.142 | 1.014 |

Kernel quality is close across this control. The host launch path remains a
separate Swage loss: batched event timing is roughly 2x slower than Triton for
small vectors. Graph replay shows that this is primarily dispatch overhead,
not a 2x device-kernel deficit.

## Host-visible call latency

Synchronized wall-clock timing includes Python dispatch and synchronization.
Compared with matched planned Triton, mixed Swage measures 15.529 versus
16.290 microseconds on `many-tiny`, 43.476 versus 49.763 on `log-normal`, and
17.372 versus 19.005 on `one-outlier`. Swage's fused direct schedule needs one
host launch where mixed planned Triton needs two, so host timing generally
favors Swage more than graph replay does.

Vector add still favors Triton's host path. At 1,024 elements, synchronized
calls measure about 14 microseconds for Swage, 9 for Triton, and 6 for
PyTorch. Device graph parity therefore should not be presented as dispatch
parity.

## What this result supports

The evidence supports discussing these propositions:

1. **Runtime shape information selects useful fixed GPU work shapes.** One
   semantic operation need not imply one physical tile for every segment.
2. **The scheduler explains most of the original win.** Giving Triton matched
   task lists changes six large fixed-baseline gaps into near parity.
3. **Packing matters for tiny tasks.** Four tasks per program turns Triton's
   `many-tiny` result from a large loss into a modest win.
4. **Swage remains a credible compiler architecture result.** It reaches
   matched hand-written performance while preserving one segment-local
   semantic program and automatic task derivation.
5. **The current result is specialized.** It covers one identity f32 sum on
   one NVIDIA architecture through private APIs.

It does **not** establish that Swage is generally faster than Triton, that
Triton cannot express a comparable scheduler, or that public Swage users can
run segmented kernels today.

## Suggested show-and-tell

A concise technical walkthrough can follow five beats:

1. **Problem:** draw packed values and offsets with thousands of unequal
   segment lengths. Ask what one fixed block should be sized for.
2. **Semantic program:** show one segment-local identity sum with no GPU
   thread or block IDs in semantic Swage IR.
3. **Task derivation:** classify the same offsets into warp, CTA, and split
   descriptors; then show the four-warp-slot fused block.
4. **Evidence:** first show the fixed Triton gaps, then reveal how matched
   planned Triton closes them. Use `log-normal` as Swage's remaining win,
   `many-tiny` as Triton's packed-task win, and `uniform` as the guardrail.
5. **Open question:** test whether Swage's compiler representation makes this
   scheduling strategy easier to generalize to new operations and device-side
   planning than equivalent hand-written Triton orchestration.

Good discussion questions include:

- Should classification remain on the host, or move into a device queue?
- Can task-list construction be amortized when offsets repeat?
- What policy features predict the best tile beyond a single length cutoff?
- How does the design extend to max, softmax, and non-identity map/reduce
  regions?
- Does persistent scheduling improve the tail without hurting tiny segments?

## Reproduce the campaign

Triton is an optional benchmark-time import and is not a Swage dependency.
With the native build, CUDA-enabled PyTorch, and Triton available:

```bash
PYTHONPATH="$PWD/python:$PWD/build/python_packages" \
python benchmarks/benchmark_triton_comparison.py \
  --suite all \
  --warmups 25 \
  --samples 100 \
  --output benchmarks/results/swage-triton-a6000-sm86.json
```

For publishable evidence, run from a clean revision on an idle or exclusively
allocated GPU, retain the complete raw JSON, repeat the process in independent
processes, and report clock/power policy. A future qualification should add
independent-process aggregation and a matched one-launch fused Triton variant;
the current planned Triton path uses separate packed-warp and CTA launches.

Continue with [Benchmarks](benchmarks.md) for the earlier RTX 5090 recorded
snapshot or [Verification](verification.md) for the exact status boundaries.
