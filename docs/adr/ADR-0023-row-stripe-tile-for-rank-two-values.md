<!-- docs/adr/ADR-0023-row-stripe-tile-for-rank-two-values.md -->
# ADR-0023: Row-stripe tile for rank-two values

- Status: accepted; steps 1 to 4 of the migration sequence are
  implemented
- Date: 2026-10-03
- Accepted: 2026-10-03, with the recommended answer to every open question
- Supersedes in part:
  [ADR-0022](ADR-0022-wider-data-model-for-segmented-reductions.md), whose
  rank-two decision was one column tile per segment with no split

## Context

ADR-0022 gave rank-two values, `[N, D]` rows of `D` features, one kernel:
the column tile. One block runs one segment, and thread `t` reduces the
columns `t`, `t + 128`, and so on, each alone in row order. Nothing is
combined across threads and no segment is split.

That tile is bound by the longest segment of a batch. A column of `n` rows
is `n` dependent loads in one thread, so a batch with a long segment waits
for that chain, and a long segment with few columns keeps few threads of
one block busy. Measurements of the public call on such batches were up to
17 times slower than `torch.segment_reduce` along axis 0, and looped Triton
row kernels were much faster than both.

## Decision

Rank-two values get a second tile, the row-stripe tile, reached through the
existing task-ids and split schedules, and the public calls run it. The
column tile stays in the compiler as the direct schedule of a rank-two
function and as a private launch path.

### The tile

- For `D` columns, the column-group width `W` is the smallest power of two
  that is at least `D`, capped at the subgroup width, and 1 for `D` of 1 or
  less. There are `ceil(D / W)` column groups.
- A work item is a task and a column group. A kernel loops over the items
  from its block index to `task_count * groups` by the grid size, and
  decodes the task and the group from the item index. The loop bounds are
  the same in every thread of a block, so the barriers of an item are
  legal, and a launch of one block per item runs one item per block.
- In a block of `T` threads, `T` a multiple of the subgroup width `S`,
  thread `t` owns column `(t mod S) mod W` of the group and is row stripe
  `(t / S) * (S / W) + (t mod S) / W`. There are `R = T / W` stripes.
- A stripe folds rows `s`, `s + R`, and so on of its column in row order
  into one scalar per reduction stage. A lane whose column is at or beyond
  `D` binds no element. Validity acts on addresses and on the final store,
  never on control flow.
- The stripes of one column combine in two steps: an XOR butterfly of five
  shuffles, each kept by a select only for an offset of at least `W`, and
  an exchange of one element per thread through a workgroup buffer of `T`
  elements between two barriers, after which every thread combines the
  results of its column from each subgroup as a pairwise tree. Every thread
  of a column ends with the same bits, which a later stage can capture.
- The thread of stripe zero stores the result of its column at
  `output[segment * D + column]`.

`W` is computed in the kernel from the feature count and on the host from
the same rule, `TargetDescription::columnGroupWidth`. It is not a
compile-time parameter: five kernels per program and role would multiply
the PTX, the digests, and the artifact.

### The split

- A segment longer than `floor(4096 / W)` rows is cut into chunks of that
  many rows, which keeps a stripe of a 512-thread block at eight rows.
- The partial kernel runs the tile at 512 threads over the rows of each
  range record and stores the result of column `c` of partial task `p` at
  `scratch[p * D + c]`. Scratch is `partial_count * D` elements.
- The merge kernel runs the tile at 512 threads over the scratch rows of
  each merge record, clamped to the partial count, and stores at
  `output[segment * D + c]`. The merge of a mean reads the row count of
  its segment from the range records.
- The split takes one capture-free reduction, as over rank-one values. The
  softmax has no split.

### The public calls

- `swage.segment_reduce` on `[N, D]` values with `D > 1` admits its program
  under the default planning limits, the ones an artifact records, and
  classifies the rows of each segment under the limits
  `(floor(32 / W), floor(4096 / W))`, which follow from the feature count.
- Both direct classes run on the task-id kernel at the CTA width: the warp
  ids and the CTA ids lie together at the start of the records. There is
  no warp tile of rank-two values.
- A longer segment runs the split. A batch without one is not classified:
  it uploads no record and launches the task-id kernel with the identity
  task list, which gives every segment the task and the bits that
  classification gives it. At 32,768 segments of up to 32 rows,
  classification under the row limits was the larger part of the host
  work of a call.
- `swage.segment_softmax` launches the task-id kernel of its program with
  the identity task list, one task per segment, and splits nothing.
- A launch runs one block per task and column group, up to `2**31 - 1`
  blocks; the item loop covers any items beyond.
- The kernels of a segment, and so its bits, follow from its row count and
  the feature count, and not from the rest of the batch.
- The private `launch_gpu` and `launch_softmax_gpu` keep the column tile.

### Schedules and plan IR

- No new plan operation and no new policy. `swage_plan.tasks policy<cta>`
  takes `feature_count`, with or without `ids`. `policy<warp>` stays
  rank-one. `swage_plan.partial_tasks` and `swage_plan.merge_tasks` take an
  optional `feature_count`, with which their buffers are rank-two.
- The task-ids schedule plans the row-stripe tile for every admitted
  rank-two program, including the softmax, whose ids then only name
  segments and which a launch runs with one task per segment. Over
  rank-one values the task-ids schedule still requires the one
  capture-free reduction that host classification describes.
- The split schedules plan the row-stripe tile of the partial and merge
  kernels for a single reduction.
- The fused-mixed and persistent schedules refuse rank-two values by name.
- A task-ids block of rank-two values is a whole number of subgroups.

### Kernel layouts and records

Four new layouts append `feature_count` to the task-id and split layouts:

| Kind | Arguments |
|---|---|
| `TaskIdsColumns` | values, offsets, output, task_ids, value_count, task_count, segment_count, feature_count |
| `SplitPartialColumns` | values, partial_ranges, scratch, value_count, partial_count, feature_count |
| `SplitMergeColumns` | scratch, output, merge_records, partial_count, merge_count, segment_count, feature_count |
| `SplitMergeExtentColumns` | scratch, output, merge_records, partial_ranges, partial_count, merge_count, segment_count, feature_count |

A launch of each runs `C` blocks per task, partial, or merge, for the
`C = ceil(D / W)` column groups.

Task records, the classifier, the C runtime, and its ABI version do not
change: the column group is the low part of the item index, and the host
and the kernel both derive the group count from the feature count. A
partial range record names rows.

### Numerics

A column sum adds within a stripe in row order, then over the stripe bits
of a subgroup, then over the subgroups as a tree, so its longest chain has
`k = ceil(n / R) - 1 + log2(R)` additions. Its bits depend on the values of
the column, on `n`, and on `D` through `W`, and not on the batch, the grid,
the device model, or the offset width. A maximum and a minimum are
order-free and keep their bits. A sum, a mean, and a softmax change bits
relative to the column tile.

A segment cut into `P` chunks of `4096 / W` rows adds eight rows per
stripe and combines in the partial kernel, then combines the partials in
the merge kernel, for `k = 6 + 2 log2(R) + ceil(P / R)` with
`R = 512 / W`. The merge depth grows with `P / R`, about `n / 2048` at
`W = 32`; a second merge level is left until a caller needs a tighter
bound in the millions of rows.

## Consequences

- The bits of a rank-two sum, mean, and softmax change from those of the
  column tile; a maximum and a minimum keep theirs. A sum has the `k` of
  the tile in place of `n - 1`.
- An artifact holds the `cta`, `partial`, and `merge` roles for each
  rank-two reduction and the `cta` role for the rank-two softmax, in place
  of the `column` role. The format version stays 2, and an artifact
  written before is refused at load time.

- The digest matrix gains the task-ids kernel of every rank-two program,
  and the partial and merge kernels of every rank-two reduction, on
  `sm_80` and `sm_86`. The 836 pairs before them do not move, which
  includes the 18 pairs of the column tile.
- The column tile keeps its bit equality with the CPU oracle, which adds
  in row order. The row-stripe tile does not have it; its tests compare
  bits with a host model of its order of additions instead.

## Migration sequence

Step 1. The block tile through task-ids, including the softmax. Implemented.

- `swage_plan.tasks policy<cta>` over rank-two values, the planner rules,
  the `TaskIdsColumns` layout, the column-group combination in
  `Emission.cpp`, the item loop and the exchange buffer in
  `SwagePlanToGPU.cpp`, and the width rule of the conversion's pre-check.
- `test/Conversion/SwagePlanToGPU/column-groups.mlir` pins the kernel;
  `python/tests/mlir/test_segmented_bounds.py` launches it below the Python
  validation and compares its bits with
  `python/tests/mlir/row_tile_model.py`; the racecheck runs it.
- The digest matrix gains 18 pairs.

Step 2. The split. Implemented.

- `feature_count` on `swage_plan.partial_tasks` and
  `swage_plan.merge_tasks`, the three split layouts, the planner rule, and
  the row-stripe paths of the partial and merge patterns with the width
  rule of the pre-check.
- `test/Conversion/SwagePlanToGPU/column-groups.mlir` pins both kernels;
  `python/tests/mlir/test_segmented_bounds.py` launches them below the
  Python validation, with stray merge records and ranges beyond the
  partial count, requires a mean to equal the merged sum divided by the
  row count bit for bit, and compares the bits of the split with the tile
  model; the racecheck runs both.
- The digest matrix gains 32 pairs.

Step 3. The public reductions. Implemented.

- `_launch_planned_rows`, `_column_group_width`, `_row_limits`, and
  `_row_grid` in `python/swage/_segmented_qualification.py`; the call in
  `python/swage/_segments.py`; the roles in `python/swage/_artifact.py`;
  and `python/swage/compile.py`, which now admits the rank-two reductions.
- `python/tests/mlir/test_segment_columns.py` compares the bits of the call
  with the tile model through the split, bounds it with the `k` of the
  tile, and checks the limits it classifies under and the kernels it
  compiles; the oracle test runs the private column tile;
  `tests/python/test_segments.py` checks the width rule and the limits on
  the host; the artifact tests run the new roles.

Step 4. The public softmax. Implemented.

- The `row_stripes` launch of `_launch_columns`, the call in
  `python/swage/_segments.py`, and the role in `python/swage/_artifact.py`.
- `python/tests/mlir/test_segment_columns.py` bounds every output with the
  `k` of one block, and keeps the block-size test on the private column
  tile.

Steps 5 and 6, a packed warp tile for very small `D` and a grid cap, are
made only on measurement.

## Rejected alternatives

- A compile-time `W`: more kernels, digests, and artifact entries for a few
  instructions per stage.
- A flat mapping, `column = flat index mod D`: for a `D` that is not a power
  of two the threads of one column are no butterfly.
- The column group in task records: records `C` times larger and a new
  classifier and runtime ABI.
- A two-dimensional grid: the runtime launches with `gridX` only.
- A new schedule and C API entry point for the softmax: the task-ids
  schedule with identity ids does the same.
