<!-- docs/user-guide/ragged-data.md -->

# Ragged Data

Variable-sized, internally dense segments appear in ragged softmax rows,
jagged batches, and graph neighborhoods. Fixed GPU work shapes handle
regular data well, but padding and one-shape scheduling waste work or
create load imbalance on ragged inputs. Everything in Swage grows from how
it stores and names this data.

Swage starts with ragged storage: one dense values buffer plus offsets that
divide it into logical segments. In the concrete example below, offsets
`[0, 2, 2, 5, 6]` describe ranges `[0, 2)`, `[2, 2)`, `[2, 5)`, and `[5, 6)`
over six dense values. The repeated offset makes the second segment empty.
Segment `i` is always the half-open slice
`values[offsets[i] : offsets[i + 1]]`; differing segment lengths do not add
padding or gaps to the values buffer.

<div class="doc-figure" tabindex="0" markdown="1">

![Dense values and offsets forming four half-open segments](../assets/diagrams/ragged-storage.svg)

</div>

*Dense ragged storage, including a repeated offset and empty segment. [Open the full-size figure](../assets/diagrams/ragged-storage.svg).*

## The offsets contract

Segments are not usable from Python yet, so nothing in this section is a
public call. It records what the private qualification path validates
today, on a host copy of the offsets, before it compiles or launches
anything. An input that breaks a rule is rejected with an error. Nothing is
cast, moved to another device, or repaired.

For `N` segments over one values buffer:

- `values` is a contiguous rank-one `torch.float32` tensor.
- `offsets` is a contiguous rank-one `torch.int32` tensor with `N + 1`
  entries.
- `offsets[0]` is zero.
- `offsets` never decreases. Two equal neighbors describe an empty segment.
- `offsets[N]` is at most the number of values. Values past it belong to no
  segment.
- The number of values and the number of segments are each below `2**31`.
- `output` is a contiguous rank-one `torch.float32` tensor with at least `N`
  elements, and it does not overlap `values` or `offsets` in memory.
- `values` and `output` do not require grad.
- For a GPU launch, all three tensors are CUDA tensors on the current
  device.

Ragged softmax writes one result per value instead of one per segment, so
its `output` needs at least `offsets[N]` elements.

## Converting other layouts to offsets

The contract admits one layout. Each layout below is not admitted and needs
a conversion in PyTorch first:

- `int64` offsets: check that the final offset is below `2**31`, then cast
  with `offsets.to(torch.int32)`. The cast wraps silently if the check is
  skipped.
- Offsets without the final entry, as `torch.nn.EmbeddingBag` takes them:
  append the number of values.
- Offsets that start above zero, as a slice of a larger batch has them:
  subtract the first offset and pass the matching slice of the values.
- A lengths vector: prefix-sum it behind a leading zero.
- A sorted index vector, with one segment id per value in nondecreasing
  order: count the ids and prefix-sum the counts, as shown below.
- An unsorted index vector: sort the values by segment id first. There is no
  scatter form, so the sort and the gather are PyTorch work.

A sorted index vector, such as the batch vector of a graph mini-batch,
becomes offsets with PyTorch alone:

```python
import torch

# index[i] is the segment of values[i]: sorted, with ids in [0, 4).
index = torch.tensor([0, 0, 2, 2, 2, 3])
segment_count = 4

lengths = torch.bincount(index, minlength=segment_count)
offsets = torch.zeros(segment_count + 1, dtype=torch.int64)
offsets[1:] = torch.cumsum(lengths, dim=0)
assert int(offsets[-1]) < 2**31
offsets = offsets.to(torch.int32)  # tensor([0, 2, 2, 5, 6])
```

`minlength` keeps segments that have no values, so segment 1 stays in the
result as an empty segment. The result is the offsets example at the top of
this page. For an unsorted index, sort first and carry the values along:

```python
order = torch.argsort(index, stable=True)
values, index = values[order], index[order]
```

## Empty segments and NaN

The private qualification path fixes three results:

- The sum of an empty segment is `0.0`.
- The maximum of an empty segment is negative infinity.
- The maximum of a segment that contains a NaN is NaN.

The qualification tests pin all three. On PyTorch 2.12,
`torch.segment_reduce` returns the same values, and
`torch.nn.functional.embedding_bag` with `mode="max"` differs on the last
two: it returns `0.0` for an empty bag and skips a NaN member. No option
changes the empty value. To get zero for empty segments, replace by length
after the call, not by value, because a segment whose values are all
negative infinity has the same maximum:

```python
empty = offsets[1:] == offsets[:-1]
output = output.masked_fill(empty, 0.0)
```

An f32 sum has one more property that a caller should know: its rounding
depends on the schedule. [Execution Model](execution-model.md) states what
is and is not guaranteed. The internal module shapes and ABIs behind this
contract are recorded in
[Segmented Reductions](../internals/segmented-reductions.md).

## Three questions, three levels

The rest of this guide follows one ladder of questions:

```text
Segment: What logical data and computation does one program instance mean?
    |
    v
Task:    What schedulable work is needed for the observed segment length?
    |
    v
Tile:    What fixed warp or CTA step executes that task?
```

A segment is runtime-sized. A task is derived work. A tile is a fixed
physical step. Keeping the levels separate prevents the runtime shape from
leaking into semantic types or hard-coding GPU indices into the program.

## Why the separation matters

If a segment were treated as a fixed tile, short rows would be padded and
long rows would not fit. If a segment always became one task, tiny rows could
underuse a CTA and a very long row could dominate the launch tail. If GPU IDs
were part of semantic IR, changing the schedule would require rewriting the
kernel's meaning.

Swage instead preserves the segment-local meaning and allows task derivation
to be qualified separately. Today, that separation is public for canonical
fixed vector add and privately qualified for selected segmented modules.
General public segmented execution remains planned.

Continue with [Writing Kernels](writing-kernels.md). That page writes the
one kernel the public frontend accepts today, a fixed-block vector add. It
uses no segment, because segment syntax is not public.
[Execution Model](execution-model.md) returns to segments, tasks, and tiles
and gives the invariants behind each level.
