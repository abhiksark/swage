<!-- docs/internals/a6000-comparison.md -->

# Swage and Triton on Ragged Reductions

This study asks a narrow question: **when segment lengths vary at runtime, is
it useful to derive different fixed GPU tasks from one segment-local
program?** Two records on one NVIDIA RTX A6000 answer it, and the answer has
three parts:

- Deriving tasks helps on the rows of short segments and on the one row
  with segments above 4,096 elements.
- It does not help on the four rows that mix short segments with segments
  of up to 4,096 elements. There a looped Triton kernel with no planner is
  faster than the Swage mixed policy.
- A planned Triton scheduler that receives the same task lists matches or
  beats the mixed policy on most rows, so the gain comes from the schedule
  and not from Swage's code generation.

!!! warning "Exploratory evidence, not a release claim"

    This page reports two local campaigns on one GPU. Neither is a trusted
    GPU qualification, a continuously enforced gate, or a public performance
    contract. Both time the private prepared launch with one frozen layout
    per row, not the public `swage.segment_reduce` call, which prepares on
    every call. With a new offsets layout on every call the private path is
    slower than `torch.segment_reduce`; [Benchmarks](benchmarks.md#fresh-offsets-and-the-frozen-comparison-at-453c56e)
    reports that regime.

## The two records

| Record | Revision | Processes | Rows | Triton baselines | Kernels |
|---|---|---|---|---|---|
| [`swage-triton-a6000-sm86.json`](https://github.com/abhiksark/swage/blob/main/benchmarks/results/swage-triton-a6000-sm86.json) | `80f222d` | 1 | 7 | One block per segment, and a matched planned scheduler whose longer tasks read one block | Before the LLVM pass pipeline and the device-side bounds |
| [`segmented-sum-a6000-sm86-453c56e`](https://github.com/abhiksark/swage/blob/main/benchmarks/results/segmented-sum-a6000-sm86-453c56e.md) | `453c56e` | 5 | 9 | The same two, a looped kernel, and a planned scheduler whose longer tasks loop | The ones generated now |

The older record is what it is: a comparison with a Triton kernel that
provisions one block for the longest segment, and with a matched planned
scheduler, at an older revision. Its tables are kept below unchanged. The
newer record repeats the comparison at revision `453c56e` with baselines
that do not provision for the longest segment. Where the two disagree, the
newer one describes the current code and the stronger baselines.

Under graph replay at 32,768 segments, the newer record says:

--8<-- "docs/internals/_generated/segmented-sum-a6000-sm86-453c56e-frozen-statements.inc"

Each "best" is chosen after the run from a sweep, separately for every row,
so the Triton figures are optimistic for a user who must pick one
configuration in advance.

## The `80f222d` record: one block per segment and matched planned Triton

Against a swept fixed-shape Triton kernel, which reads one block per
segment, Swage is faster on six of seven segmented distributions in this
record. The largest graph-replay differences are 6.60x on `one-outlier`,
2.58x on `many-tiny`, and 1.90x on `few-huge`. Those gaps measure the cost
of provisioning a block for the longest segment. They are not a comparison
with Triton in general.

Against a matched heterogeneous Triton scheduler with the same 32-element
cutoff, packed four-warp programs, and separate CTA tasks, the result
narrows:

- Swage is 16.2% faster on `log-normal`;
- Swage and matched Triton are within 1.3% on `uniform`, `bimodal`,
  `zipf-like`, `few-huge`, and `one-outlier`; and
- matched Triton is 8.3% faster on `many-tiny` under graph replay.

The fixed vector-add control is also close under graph replay, with Swage
between parity and 7.0% behind Triton across the sweep. Host-visible calls
still favor Triton because Swage's Python launch path costs more.

Every timing in this section describes the PTX of commit `80f222d`, one
process, and all-one values.

### What is being compared

All segmented implementations compute an identity f32 sum from one packed
values array and an i32 offsets array. Compilation, classification,
allocation, and module loading happen before timing. Every output is checked
exactly against the known all-one result before samples are collected.

#### Swage

The same semantic segmented-sum module is classified from runtime offsets:

- lengths from 0 through 32 become 32-thread warp tasks;
- lengths from 33 through 4096 become 128-thread CTA tasks;
- longer segments would become 4096-element partial tasks plus a merge.

The `mixed` policy executes short and CTA-sized direct tasks in one fused
128-thread kernel. Each initial block contains four independent warp slots;
CTA tasks follow at one segment per block. See [Task Planning](planning.md),
[Task Execution](task-execution.md), and [Split Execution](split-execution.md)
for the exact private contracts.

#### Triton

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

#### PyTorch

`torch.segment_reduce(values, "sum", offsets=offsets)` is the framework
baseline. PyTorch 2.12 does not expose an `out=` argument for this operation,
so its measured Python call includes output allocation. Batched CUDA events
primarily expose device work but do not erase that semantic difference.

### Test distributions

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

### Segmented-sum results

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
of that gap. In this record `log-normal` is the one row where Swage is ahead
of matched planned Triton by more than 2 percent, and the other mixed
distributions are effectively parity under graph replay. The matched
baseline is not the fastest Triton on that row. The `453c56e` record below
times a looped kernel and a planned scheduler whose longer tasks loop, and
both are faster than Swage on `log-normal`.

Batched CUDA events retain launcher submission while amortizing it over 32
calls. They show the same broad picture, with Swage at 0.840x planned Triton
on `log-normal`, between 0.967x and 1.004x on four other distributions,
1.072x on `many-tiny`, and 0.918x on `one-outlier`.

### Why the mixed policy helps

#### It avoids maximum-length provisioning

For the one-program Triton baseline, a 4096-element maximum requires a
4096-lane logical block even when almost every segment contains only a few
elements. Masking preserves correctness, but most lanes perform no useful
loads. The `one-outlier` distribution is the cleanest demonstration: one
segment changes the fixed block required by all 32,768 programs.

Swage instead uses runtime lengths to derive fixed tasks. The large segment
gets CTA work while the remaining segments get warp work. Runtime segment
identity stays in SSA values and task IDs rather than types, preserving the
semantic program.

A Triton kernel that loops over its segment in fixed blocks avoids the same
provisioning without a planner. This record has no such baseline. The
`453c56e` record measures it.

#### It amortizes short-segment scheduling

The fused mixed kernel places four independent short segments into four warp
slots of one 128-thread block. It therefore avoids launching one full CTA per
tiny segment and avoids a second kernel launch between direct warp and CTA
work.

The `many-tiny` row is important here. Its maximum is only 32, so fixed
Triton can already select a small block but still launches one program per
segment. Packing four tasks per planned Triton program changes graph time from
21.850 to 7.776 microseconds and slightly beats Swage's 8.480 microseconds.
This isolates packed task organization as the source of the large gain.

#### It retains a sensible uniform path

On uniformly distributed lengths through 4096, pure CTA is Swage's best
policy and is within 0.5% of the best Triton result. Classification does not
create a win when the workload has little exploitable shape separation, but
the selected homogeneous policy does not materially lose either.

### The vector-add control

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

### Host-visible call latency

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

## The `453c56e` record: looped and looping planned Triton

The newer record was made on 2026-10-02 at revision `453c56e` from a clean
tree, with five independent processes, 32,768 segments, seed 7, all-one
values, and nine distributions: the seven above, `alternating-empty`, and
`power-law`, whose longest segment exceeds 4,096 elements and takes the
split path. Its [summary page](https://github.com/abhiksark/swage/blob/main/benchmarks/results/segmented-sum-a6000-sm86-453c56e.md)
holds every table, the method, and the machine conditions. The machine was
not quiet: a desktop session ran on the same GPU and the CPU frequency
governor was `powersave`. Every number in this section is generated from
the committed summaries by `benchmarks/campaign_tables.py`.

A comparator cell gives the median time of the comparator in microseconds
under graph replay and, in parentheses, the median ratio of the Swage mixed
policy to it. A ratio above one means Swage took longer:

--8<-- "docs/internals/_generated/segmented-sum-a6000-sm86-453c56e-frozen-overview.inc"

The Swage policies and the PyTorch baselines of the same run:

--8<-- "docs/internals/_generated/segmented-sum-a6000-sm86-453c56e-frozen-swage.inc"

The padded column is the `torch_padded` baseline of revision `453c56e`,
which multiplied the padded matrix by its mask and allocated its output
inside the timed launch. The `torch_padded` of the current comparison
harness sums the padded rows into a preallocated output, so its times are
not comparable with this column.

Against the looped Triton kernel, with the range of the ratio across the
five processes and the number of the 15 configurations that are faster than
the Swage mixed policy:

--8<-- "docs/internals/_generated/segmented-sum-a6000-sm86-453c56e-frozen-looped.inc"

### What changed against the older record

- `log-normal` is no longer a Swage win. The mixed policy is still ahead
  of the matched planned baseline there, which reads one block of 4,096
  elements per longer task. A looped kernel and a planned scheduler whose
  longer tasks loop are both faster than the mixed policy on that row.
- The looped kernel, which has no planner, is faster than the mixed policy
  on four rows: `log-normal`, `bimodal`, `zipf-like`, and `few-huge`. These
  rows mix short segments with segments of up to 4,096 elements. The older
  record has no such baseline.
- Matched planned Triton is ahead of the mixed policy on the three rows of
  packed short segments, and the two are equal on four rows.
- The mixed policy keeps a clear lead over a looped kernel on the rows of
  short segments and on `power-law`, the one row that takes the split path.
  On `power-law` it is also ahead of the looping planned scheduler.
- The comparison now has five processes and ranges, where the older record
  has one process.

## What the two records support

The evidence supports these propositions:

1. **Runtime shape information selects useful fixed GPU work shapes.**
   Packing short segments and splitting very long ones beat a per-segment
   kernel, with one block or with a loop, on the rows where such segments
   dominate.
2. **The schedule explains the result, not the code generator.** A planned
   Triton scheduler that receives the same task lists matches or beats the
   Swage mixed policy on most rows.
3. **A per-segment loop is the better shape for CTA-sized segments.** A
   looped Triton kernel is faster than the mixed policy on the four rows
   that mix short segments with segments of up to 4,096 elements, so the
   CTA task of the mixed policy is its weak part.
4. **Swage derives the schedule from one segment-local program.** It needs
   no hand-written orchestration. In these records that does not buy speed
   over hand-written Triton.
5. **The result is specialized.** It covers one identity f32 sum on one
   NVIDIA architecture through private prepared launches with a frozen
   layout.

It does **not** establish that Swage is generally faster than Triton, that
Triton cannot express a comparable scheduler, or that the public
`swage.segment_reduce` call has these timings. The public call prepares on
every call, and [Benchmarks](benchmarks.md#where-swage-loses) reports what
that preparation costs.

## Presenting the result

A technical walkthrough can follow five steps:

1. **Problem:** draw packed values and offsets with thousands of unequal
   segment lengths. Ask what one fixed block should be sized for.
2. **Semantic program:** show one segment-local identity sum with no GPU
   thread or block IDs in semantic Swage IR.
3. **Task derivation:** classify the same offsets into warp, CTA, and split
   descriptors; then show the four-warp-slot fused block.
4. **Evidence:** first show the fixed Triton gaps of the older record, then
   the `453c56e` table. Matched planned Triton closes the gaps, a looped
   kernel is faster than Swage on four rows, and Swage keeps a lead over the
   looped kernel only on the rows of short segments and on `power-law`. Use
   `uniform` as the control.
5. **Open question:** test whether Swage's compiler representation makes this
   scheduling strategy easier to generalize to new operations and device-side
   planning than equivalent hand-written Triton orchestration.

Open questions for discussion:

- Should classification remain on the host, or move into a device queue?
- Can task-list construction be amortized when offsets repeat?
- What policy features predict the best tile beyond a single length cutoff?
- How does the design extend to max, softmax, and non-identity map/reduce
  regions?
- Does persistent scheduling improve the tail without hurting tiny segments?

## Reproduce the campaign

Triton is an optional benchmark-time import and is not a Swage dependency.
The `453c56e` record was produced by the commands below, with the native
build, CUDA-enabled PyTorch, and Triton available, from a clean worktree,
and with `OUT` set to a new directory outside the checkout. The last command
is the frozen comparison of this page; the six before it are the
fresh-offsets runs that [Benchmarks](benchmarks.md) reports:

--8<-- "docs/internals/_generated/segmented-sum-a6000-sm86-453c56e-reproduce.inc"

The block shows the driver of revision `453c56e`, as the record states it;
the current tree replaces that driver with
`benchmarks/run_triton_comparison_campaign.py`, and an exact rerun needs
that revision. Each driver passes `--output` to each process itself, writes
one record per process, and summarizes the per-process medians.
[Benchmarks](benchmarks.md#independent-processes) describes the current
driver and its summary. A rerun is a new measurement: it must not write
into `benchmarks/results/`, and the committed records are not replaced by
it.

The `80f222d` record is the output of one process of
`benchmarks/benchmark_triton_comparison.py` at that commit. It records 25
warmups and 100 samples and holds both the segmented-sum and the vector-add
results. The harness has changed since, so it cannot be reproduced bit for
bit from the current tree.

For publishable evidence, run on an idle or exclusively allocated GPU,
retain the complete raw JSON of every process, and report the clock and
power policy.

The planned Triton baselines of both records use separate packed-warp and
CTA launches. The current comparison harness also holds a one-launch
variant, `triton_fused`. It takes the task lists of the matched planned
baseline and launches one grid: each of the first programs sums four packed
short tasks with 32 of its 128 lanes each, and each later program sums one
longer task in 128-lane strides of at most 4,096 elements, with four warps.
A row whose longest segment exceeds 4,096 elements skips it. No committed
record holds `triton_fused`, so this page reports no result for it.

This is the last page of the internals section. Continue with the
[ADR Index](../decisions/index.md) for the decisions behind each boundary.
[Benchmarks](benchmarks.md) holds the other records, and
[Verification](verification.md) holds the exact status boundaries.
