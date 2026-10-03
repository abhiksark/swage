# python/tests/mlir/test_segment_columns.py
"""Rank-two values in `swage.segment_reduce` and `swage.segment_softmax`.

`[N, D]` values are `N` rows of `D` features. The offsets delimit rows, and
every column of a segment is reduced on its own, as `torch.segment_reduce`
does along axis 0, or normalized on its own, as `torch.softmax` does along
dimension 0 of the rows of a segment. A call runs the row-stripe tile of
ADR-0023: a block takes a group of `W` adjacent columns of one segment, `W`
the smallest power of two that is at least `D`, capped at 32, and its
threads are stripes of rows. A reduction cuts a segment of more than
`4096 / W` rows into chunks and merges them. The tests here pin, for the
reduction:

- Agreement with `torch.segment_reduce(axis=0)` at 1, 3, 64, 129, 200, and
  1024 columns, for every kind and both dtypes, with a maximum and a
  minimum exact and a sum inside the bound of the depth of the tile.
- Bit equality with `row_tile_model.py`, which adds in the order of the
  tile, through the split as well.
- Bit equality of the private column tile with the CPU oracle, which adds
  a column in the same row order.
- That a result reads its own rows and its own column, on values that
  depend on both, and that its bits do not depend on the batch.
- `[N, 1]` values through the schedules of rank one, and `[N, 0]` values
  without a launch.
- The refusals of the shape rules.

For the softmax, which takes float32 values only, they pin agreement with
float64 `torch.softmax` inside the bound of docs/internals/ragged-softmax.md
at the depth of the tile, agreement with the CPU oracle, the special
values, the two degenerate widths, and the refusals.

The device-side bounds of the kernels are in `test_segmented_bounds.py`.
"""

import fractions
import math
from itertools import pairwise

import numpy
import pytest
import row_tile_model
import swage
import torch
from swage import _runtime
from swage import _segmented_qualification as qualification
from swage._segmented_qualification import (
    cpu_oracle,
    cpu_softmax_oracle,
    launch_gpu,
    launch_softmax_gpu,
)
from test_segmented_numerics import _softmax_bound
from test_segmented_runtime import (
    _EPS32,
    _EPS64,
    _assert_softmax_matches_oracle,
    _bits,
    _offsets,
)

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)
KINDS = ["sum", "max", "min", "mean"]
DTYPES = [torch.float32, torch.float64]
_DTYPE_IDS = ["float32", "float64"]
# One column, fewer columns than a warp, half a block, one column more than
# a block, a partly filled second block of columns, and eight blocks.
FEATURES = [1, 3, 64, 129, 200, 1024]
# Rows per segment: empty segments, one row, both sides of the widths that
# matter to the rank-one schedules, and one segment of many rows.
LENGTHS = [0, 1, 31, 32, 33, 129, 0, 1500, 5]


def _depth(rows, features):
    """Return the rounding additions on the longest path of a column sum.

    A segment of `n` rows of at most `4096 / W` rows is one block of
    `R = 128 / W` stripes, `ceil(n / R) - 1 + log2(R)` additions, and a
    longer one is cut into `P` chunks of that many rows and merged by
    blocks of `R = 512 / W` stripes, `6 + 2 log2(R) + ceil(P / R)`. No path
    has more than the `n - 1` additions of nonzero terms. One column takes
    the schedules of rank one, whose bound is not tighter than `n - 1`
    for every length, so it keeps `n - 1`.
    """
    sequential = max(rows - 1, 0)
    if features == 1:
        return sequential
    width = row_tile_model.group_width(features)
    chunk = 4096 // width
    if rows <= chunk:
        return min(sequential, _block_depth(rows, features))
    stripes = row_tile_model.SPLIT_BLOCK_THREADS // width
    partials = -(-rows // chunk)
    tile = 6 + 2 * (stripes.bit_length() - 1) + -(-partials // stripes)
    return min(sequential, tile)


def _block_depth(rows, features):
    """Return `ceil(n / R) - 1 + log2(R)` for the stripes of one CTA block.

    It is the depth of a column sum of `n` rows in one block of the tile,
    the reduction of a segment that is not split and the normalizer of
    the softmax, which no segment splits.
    """
    width = row_tile_model.group_width(features)
    stripes = row_tile_model.CTA_BLOCK_THREADS // width
    return -(-rows // stripes) - 1 + stripes.bit_length() - 1


def _tile_model(kind, host_values, host_offsets):
    """Return the result of the row-stripe tile, as the call classifies it."""
    rows = host_values.numpy()
    chunk = 4096 // row_tile_model.group_width(rows.shape[1])
    results = []
    for begin, end in pairwise(host_offsets.tolist()):
        segment = rows[begin:end]
        if end - begin > chunk:
            results.append(row_tile_model.reduce_split(segment, chunk, kind))
        elif kind == "mean":
            results.append(row_tile_model.mean_rows(segment))
        else:
            results.append(row_tile_model.reduce_rows(segment, kind))
    return torch.from_numpy(numpy.stack(results))


def _rows(lengths, columns, seed, dtype):
    """Draw mixed-sign rows over many binades for the given segment lengths."""
    generator = torch.Generator().manual_seed(seed)
    shape = (sum(lengths), columns)
    exponents = torch.empty(shape, dtype=dtype).uniform_(
        -8, 8, generator=generator
    )
    values = torch.randn(shape, dtype=dtype, generator=generator)
    return values * torch.exp2(exponents), torch.tensor(
        _offsets(lengths), dtype=torch.int32
    )


def _reduce(kind, host_values, host_offsets, **keywords):
    """Run the public reduction on fresh device tensors."""
    return swage.segment_reduce(
        host_values.cuda(), host_offsets.cuda(), kind, **keywords
    ).cpu()


def _exact_column_sums(values, offsets):
    """Return the exactly rounded float64 sum of every segment and column."""
    columns = values.double().t().tolist()
    return torch.tensor(
        [
            [math.fsum(column[begin:end]) for column in columns]
            for begin, end in pairwise(offsets.tolist())
        ],
        dtype=torch.float64,
    ).reshape(offsets.numel() - 1, values.shape[1])


def _assert_same_results(actual, expected):
    """Require the same NaN positions and the same bits everywhere else."""
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.isnan(), expected.isnan())
    stored = ~expected.isnan()
    assert _bits(actual[stored]) == _bits(expected[stored])


def _assert_columns_match(kind, host_values, host_offsets, actual):
    """Compare one rank-two result with PyTorch and with a bound.

    A maximum and a minimum are exact. A column sum of `n` rows lies within
    `k * eps * sum(|x|)` of the exact sum, for the `k` of `_depth`. A mean
    is the sum of the same call divided by the number of rows, bit for bit,
    and NaN for no rows.
    """
    lengths = (host_offsets[1:] - host_offsets[:-1]).long()
    features = host_values.shape[1]
    covered = host_values[: int(host_offsets[-1])]
    theirs = torch.segment_reduce(
        covered.cuda(), kind, lengths=lengths.cuda(), axis=0
    ).cpu()
    assert actual.dtype == host_values.dtype
    assert actual.shape == theirs.shape == (lengths.numel(), covered.shape[1])
    if kind in ("max", "min"):
        assert _bits(actual) == _bits(theirs)
        return
    eps = _EPS64 if host_values.dtype == torch.float64 else _EPS32
    depth = torch.tensor(
        [_depth(length, features) for length in lengths.tolist()],
        dtype=torch.float64,
    )[:, None]
    magnitude = _exact_column_sums(covered.abs(), host_offsets)
    if kind == "mean":
        total = _reduce("sum", host_values, host_offsets)
        rows = lengths.to(host_values.dtype)[:, None]
        _assert_same_results(actual, total / rows)
        empty = (lengths == 0)[:, None].expand_as(actual)
        assert torch.equal(actual.isnan(), empty)
        kept = lengths > 0
        difference = (actual.double() - theirs.double()).abs()[kept]
        allowed = 2 * (depth + 1) * eps * magnitude / rows.double()
        assert (difference <= allowed[kept]).all()
        return
    error = (actual.double() - _exact_column_sums(covered, host_offsets)).abs()
    assert (error <= depth * eps * magnitude).all(), (
        f"largest error {(error / (eps * magnitude)).nan_to_num().max()} "
        "eps * sum(|x|) exceeds the bound of the tile"
    )
    difference = (actual.double() - theirs.double()).abs()
    assert (difference <= 2 * depth * eps * magnitude).all()


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("features", FEATURES)
def test_columns_match_pytorch_along_axis_zero(features, kind, dtype):
    """Reduce every column of every segment as PyTorch does along axis 0.

    At one column the call takes the schedules of rank one, whose sums are
    added by other trees and keep the rank-one bound, which is tighter
    than the sequential one for every length here. The segment of 1,500
    rows is split from 129 columns on.
    """
    host_values, host_offsets = _rows(LENGTHS, features, features, dtype)

    actual = _reduce(kind, host_values, host_offsets)

    _assert_columns_match(kind, host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("features", [2, 3, 33, 129])
def test_column_bits_equal_the_row_tile_model(features, kind, dtype):
    """Add in the order of the row-stripe tile, through the split as well.

    The values span many binades, so two orders of addition give other
    bits. The segments of 1,500 and 5,000 rows are split at every width
    here but two columns, at which 5,000 rows are.
    """
    lengths = [0, 1, 31, 129, 1500, 0, 5000]
    host_values, host_offsets = _rows(lengths, features, features, dtype)

    actual = _reduce(kind, host_values, host_offsets)

    _assert_same_results(actual, _tile_model(kind, host_values, host_offsets))


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
def test_the_column_tile_equals_the_cpu_oracle_bit_for_bit(kind, dtype):
    """The private column tile adds a column in row order, as the oracle.

    The values are not exactly summable, so two orders would differ. The
    oracle lowers the same program through the sequential schedule and runs
    it on the host. No public call launches the column tile.
    """
    lengths = [0, 1, 2, 33, 150, 0, 7]
    host_values, host_offsets = _rows(lengths, 5, 11, dtype)
    output = torch.full(
        (len(lengths), 5), float("nan"), dtype=dtype, device="cuda"
    )

    launch_gpu(host_values.cuda(), host_offsets.cuda(), output, kind)

    _assert_same_results(
        output.cpu(), cpu_oracle(host_values, host_offsets, kind)
    )


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("features", [3, 129, 300])
def test_columns_read_their_own_rows_and_their_own_column(
    features, kind, dtype
):
    """Reduce values that depend on the row and on the column, exactly.

    Every value is a small integer that changes with its row and with its
    column, so a sum is exact in both dtypes whatever its order, and a
    result that took a neighboring column, or a window of rows moved by
    one, is another number. The columns beyond the block width run in the
    second pass of a thread.
    """
    lengths = [3, 0, 130, 1, 64, 2000]
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    row = torch.arange(sum(lengths))[:, None]
    column = torch.arange(features)[None, :]
    host_values = ((row * 31 + column * 17) % 127 - 63).to(dtype)

    actual = _reduce(kind, host_values, host_offsets)

    # Every sum is an integer below 2**24, so the reference is exact in the
    # dtype of the values, and its mean is that sum divided once.
    expected = torch.segment_reduce(
        host_values, kind, lengths=torch.tensor(lengths), axis=0
    )
    _assert_same_results(actual, expected)
    # Neighboring columns do differ.
    assert not torch.equal(expected[:, :-1], expected[:, 1:])


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
def test_a_long_segment_of_few_columns_keeps_the_bound_of_the_split(dtype):
    """Reduce 100,003 rows of three columns through the split.

    Three columns are one group of four, so the segment is cut into 98
    chunks of 1,024 rows, each reduced by 128 stripes of eight rows, and
    the merge adds the 98 partials of each column: 21 additions on the
    longest path. The mean divides the merged sum once.
    """
    host_values, host_offsets = _rows([100_003, 2], 3, 3, dtype)

    for kind in ("sum", "mean"):
        actual = _reduce(kind, host_values, host_offsets)
        _assert_columns_match(kind, host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
def test_a_column_mean_is_within_its_bound_of_the_exact_mean(dtype):
    """Bound a column mean against the exactly rounded mean.

    The reference is formed in rational arithmetic and rounded once. A
    mean of `n` rows lies within `eps * sum(|x|)` of it, which is
    `(k + 1) * eps * sum(|x|) / n` for the at most `k = n - 1` additions
    of nonzero terms on a path of the tile and one unit for the division
    and the conversion of `n`.
    """
    lengths = [1, 2, 33, 400]
    host_values, host_offsets = _rows(lengths, 4, 5, dtype)
    columns = [
        [fractions.Fraction(value) for value in column]
        for column in host_values.double().t().tolist()
    ]
    bounds = list(pairwise(host_offsets.tolist()))
    exact = torch.tensor(
        [
            [
                float(
                    sum(column[begin:end], fractions.Fraction())
                    / (end - begin)
                )
                for column in columns
            ]
            for begin, end in bounds
        ],
        dtype=torch.float64,
    )
    magnitude = _exact_column_sums(host_values.abs(), host_offsets)
    eps = _EPS64 if dtype == torch.float64 else _EPS32

    actual = _reduce("mean", host_values, host_offsets)

    error = (actual.double() - exact).abs()
    assert (error <= eps * magnitude).all()


@_needs_cuda
@pytest.mark.parametrize("block_size", [32, 64, 512, 1024])
@pytest.mark.parametrize("kind", ["sum", "mean"])
def test_column_bits_do_not_depend_on_the_block_size(kind, block_size):
    """A column is added by one thread, whatever the width of its block.

    The private launch of the column tile takes the block size. With 32
    threads a thread reduces five of the 129 columns, and with 1024
    threads at most one. Every width returns the bits of the default width.
    """
    host_values, host_offsets = _rows([0, 5, 300, 33], 129, 7, torch.float32)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    def run(width):
        output = torch.full((4, 129), float("nan"), device="cuda")
        launch_gpu(values, offsets, output, kind, width)
        return output.cpu()

    _assert_same_results(run(block_size), run(128))


@_needs_cuda
@pytest.mark.parametrize("kind", KINDS)
def test_column_bits_do_not_depend_on_the_batch(kind):
    """A segment alone has the bits it has among other segments.

    The kernels of a segment follow from its row count and the feature
    count: six columns are one group of eight, the segment of 4,500 rows
    is split into chunks of 512 rows, and the others are tasks of the
    task-id kernel. Nothing about the other segments of a batch, their
    number, or the device changes the order in which a column is added.
    """
    lengths = [7, 4500, 0, 33]
    host_values, host_offsets = _rows(lengths, 6, 13, torch.float32)
    together = _reduce(kind, host_values, host_offsets)

    for index, (begin, end) in enumerate(pairwise(host_offsets.tolist())):
        alone = _reduce(
            kind,
            host_values[begin:end],
            torch.tensor([0, end - begin], dtype=torch.int32),
        )
        _assert_same_results(alone[0], together[index])


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
def test_empty_segments_give_every_column_the_value_of_the_kind(dtype):
    """Write each column of an empty segment: 0, the infinities, and NaN."""
    lengths = [0, 4, 0, 0, 33]
    host_values, host_offsets = _rows(lengths, 130, 2, dtype)
    empty = torch.tensor(lengths) == 0

    results = {
        kind: _reduce(kind, host_values, host_offsets) for kind in KINDS
    }

    assert _bits(results["sum"][empty]) == _bits(
        torch.zeros(3, 130, dtype=dtype)
    )
    assert (results["max"][empty] == float("-inf")).all()
    assert (results["min"][empty] == float("inf")).all()
    assert results["mean"][empty].isnan().all()
    for kind in KINDS:
        assert not results[kind][~empty].isnan().any()
        _assert_columns_match(kind, host_values, host_offsets, results[kind])


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
def test_column_extremes_order_signed_zeros_as_ieee(dtype):
    """Return +0.0 as the maximum and -0.0 as the minimum of both zeros.

    Each column of a segment holds zeros of both signs at other rows, or
    zeros of one sign, so a result that took the first zero of a column,
    as `torch.segment_reduce` does on PyTorch 2.12, or a neighboring
    column, shows in its sign.
    """
    rows = 5
    host_values = torch.full((2 * rows, 4), -0.0, dtype=dtype)
    host_values[rows - 1, 0] = 0.0
    host_values[0, 1] = 0.0
    host_values[:, 3] = 0.0
    host_values[rows + 2, 2] = 0.0
    host_offsets = torch.tensor([0, rows, 2 * rows], dtype=torch.int32)

    maximum = _reduce("max", host_values, host_offsets)
    minimum = _reduce("min", host_values, host_offsets)

    zero, negative = 0.0, -0.0
    assert _bits(maximum) == _bits(
        torch.tensor(
            [[zero, zero, negative, zero], [negative, negative, zero, zero]],
            dtype=dtype,
        )
    )
    assert _bits(minimum) == _bits(
        torch.tensor(
            [[negative, negative, negative, zero]] * 2, dtype=dtype
        )
    )


@_needs_cuda
@pytest.mark.parametrize("kind", KINDS)
def test_one_column_takes_the_schedules_of_rank_one(kind, monkeypatch):
    """Reduce `[N, 1]` values as the rank-one values they are.

    The result has the bits of the rank-one call on the same elements,
    with the shape `[S, 1]`, and no kernel of rank-two values is requested.
    """
    for name in ("_launch_columns", "_launch_planned_rows"):
        monkeypatch.setattr(
            qualification,
            name,
            lambda *arguments, **keywords: pytest.fail("a rank-two kernel ran"),
        )
    lengths = [0, 3, 40, 9000]
    host_values, host_offsets = _rows(lengths, 1, 17, torch.float32)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    column = swage.segment_reduce(values, offsets, kind).cpu()
    flat = swage.segment_reduce(values.view(-1), offsets, kind).cpu()
    out = torch.full((4, 1), float("nan"), device="cuda")
    returned = swage.segment_reduce(values, offsets, kind, out=out)

    assert column.shape == (4, 1)
    assert returned is out
    _assert_same_results(column.view(-1), flat)
    _assert_same_results(out.cpu(), column)


@_needs_cuda
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("rows", [0, 6])
def test_no_column_returns_an_empty_result_without_a_launch(
    rows, kind, monkeypatch
):
    """Return `[S, 0]` for `[N, 0]` values and enqueue nothing."""
    driver = _runtime._get_driver()
    for name in ("launch_segmented", "launch_segmented_tasks"):
        monkeypatch.setattr(
            driver,
            name,
            lambda *arguments: pytest.fail("a kernel was launched"),
        )
    values = torch.empty(rows, 0, device="cuda")
    offsets = torch.tensor([0, 0, rows], dtype=torch.int32, device="cuda")
    out = torch.empty(2, 0, device="cuda")
    version = out._version

    result = swage.segment_reduce(values, offsets, kind)
    returned = swage.segment_reduce(values, offsets, kind, out=out)

    assert result.shape == (2, 0)
    assert result.dtype == torch.float32
    assert result.device == values.device
    assert returned is out
    assert out._version == version


@_needs_cuda
@pytest.mark.parametrize("columns", [1, 3])
def test_a_batch_of_rows_without_a_segment_returns_no_row(columns):
    """Return `[0, D]` when the offsets hold no segment."""
    values = torch.ones(5, columns, device="cuda")
    offsets = torch.zeros(1, dtype=torch.int32, device="cuda")

    result = swage.segment_reduce(values, offsets, "sum")

    assert result.shape == (0, columns)


@_needs_cuda
def test_a_rank_two_call_classifies_its_rows_under_the_row_limits(
    monkeypatch,
):
    """Classify rows under the default limits divided by the group width.

    Four columns are one group of four: a segment of at most 1,024 rows is
    a task of the task-id kernel, and the segment of 9,000 rows is cut into
    chunks of 1,024 rows. The program is admitted under the default limits,
    and the call compiles the three kernels its batch launches.
    """
    limits, admitted = [], []
    validator = qualification._classifying_validator
    admit = qualification._admit_program

    def classify(*arguments):
        limits.append(arguments)
        return validator(*arguments)

    def admit_once(*arguments):
        admitted.append(arguments[2:])
        return admit(*arguments)

    monkeypatch.setattr(qualification, "_classifying_validator", classify)
    monkeypatch.setattr(qualification, "_admit_program", admit_once)
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    host_values, host_offsets = _rows([3, 9000, 0], 4, 23, torch.float32)

    actual = _reduce("sum", host_values, host_offsets)

    _assert_columns_match("sum", host_values, host_offsets, actual)
    _assert_same_results(actual, _tile_model("sum", host_values, host_offsets))
    assert limits == [(8, 1024)]
    assert admitted == [(32, 4096)]
    assert [
        (compile_ptx.__name__, dict(options))
        for compile_ptx, _, options in qualification._ptx_memo
    ] == [
        (
            "_compile_segmented_reduction_ptx",
            {
                "kernel_name": "segmented_sum_r2",
                "block_size": 128,
                "target": qualification._target(torch, 0),
                "use_task_ids": True,
            },
        ),
        *(
            (
                name,
                {
                    "kernel_name": "segmented_sum_r2",
                    "target": qualification._target(torch, 0),
                },
            )
            for name in (
                "_compile_split_partial_reduction_ptx",
                "_compile_split_merge_reduction_ptx",
            )
        ),
    ]


@_needs_cuda
@pytest.mark.parametrize(
    "lengths", [LENGTHS, [0, 1, 31, 32, 33, 5]], ids=["split", "unsplit"]
)
@pytest.mark.parametrize("kind", KINDS)
def test_columns_take_int64_offsets_and_give_the_bits_of_int32(kind, lengths):
    """Narrow int64 offsets on the host, as the rank-one calls do.

    Seven columns are one group of eight, so the segment of 1,500 rows is
    split and the call uploads its records with the narrowed offsets
    behind them. Without a split it uploads the narrowed offsets alone and
    reads the identity task list.
    """
    host_values, host_offsets = _rows(lengths, 7, 29, torch.float64)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    narrow = swage.segment_reduce(values, offsets, kind).cpu()
    wide = swage.segment_reduce(values, offsets.long(), kind).cpu()

    _assert_same_results(wide, narrow)


@_needs_cuda
def test_columns_ignore_rows_past_the_final_offset():
    """Rows that no segment covers reach no result."""
    host_values = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [float("nan"), float("inf")]]
    )
    host_offsets = torch.tensor([0, 1, 2], dtype=torch.int32)

    actual = _reduce("sum", host_values, host_offsets)

    assert actual.tolist() == [[1.0, 2.0], [3.0, 4.0]]


@_needs_cuda
def test_out_of_rows_is_written_in_place_and_its_version_advances():
    """Write a caller's `[S, D]` tensor and return it."""
    host_values, host_offsets = _rows([2, 0, 5], 3, 31, torch.float64)
    values, offsets = host_values.cuda(), host_offsets.cuda()
    out = torch.full((3, 3), -5.0, dtype=torch.float64, device="cuda")
    version = out._version

    returned = swage.segment_reduce(values, offsets, "max", out=out)

    assert returned is out
    assert out._version > version
    _assert_columns_match("max", host_values, host_offsets, out.cpu())


def _host_rows(rows=6, columns=3):
    """Return small host rows and the offsets of four segments."""
    values = torch.arange(rows * columns, dtype=torch.float32)
    return values.reshape(rows, columns), torch.tensor(
        [0, 2, 2, 5, 6], dtype=torch.int32
    )


def test_values_of_another_rank_are_refused():
    """Admit rank one and rank two, and name both."""
    values, offsets = _host_rows()

    for wrong in (values[None], torch.tensor(1.0)):
        with pytest.raises(
            TypeError, match="^values must have rank one or two$"
        ):
            swage.segment_reduce(wrong, offsets, "sum")


@pytest.mark.parametrize(
    ("shape", "found"),
    [
        ((4,), r"\(4,\)"),
        ((12,), r"\(12,\)"),
        ((3, 4), r"\(3, 4\)"),
        ((4, 2), r"\(4, 2\)"),
        ((4, 3, 1), r"\(4, 3, 1\)"),
        ((5, 3), r"\(5, 3\)"),
    ],
)
def test_out_of_another_shape_is_refused_with_both_shapes(shape, found):
    """Require one row per segment and one column per feature, exactly."""
    values, offsets = _host_rows()

    with pytest.raises(
        ValueError,
        match=(
            r"^out must have shape \(4, 3\), one row per segment and one "
            rf"column per feature; found {found}$"
        ),
    ):
        swage.segment_reduce(values, offsets, "sum", out=torch.empty(shape))


def test_out_of_another_dtype_is_refused_before_its_shape():
    """Name the dtype of the values for a `[S, D]` result too."""
    values, offsets = _host_rows()

    with pytest.raises(
        TypeError, match="^out must have the dtype of values, torch.float32$"
    ):
        swage.segment_reduce(
            values, offsets, "sum", out=torch.empty(4, 3, dtype=torch.float64)
        )


@_needs_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("transposed", "values must be contiguous"),
        ("sliced-columns", "values must be contiguous"),
        ("one-sliced-column", "values must be contiguous"),
        ("transposed-out", "out must be contiguous"),
        ("offsets-past-rows", "final offset 24 exceeds value count 6"),
        ("overlap", "out must not overlap values"),
    ],
)
def test_layouts_outside_the_contract_are_refused(case, message):
    """Refuse a view that is not contiguous instead of copying it.

    The offsets delimit rows: offsets that are valid for the number of
    elements and not for the number of rows are refused, with the number
    of rows as the value count.
    """
    values = torch.arange(24, dtype=torch.float32, device="cuda")
    rows = values.reshape(6, 4)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32, device="cuda")
    keywords = {}
    if case == "transposed":
        rows, offsets = rows.t(), offsets.clamp(max=4)
    elif case == "sliced-columns":
        rows = rows[:, ::2]
    elif case == "one-sliced-column":
        rows = rows[:, 1:2]
    elif case == "transposed-out":
        keywords["out"] = torch.empty(4, 2, device="cuda").t()
    elif case == "offsets-past-rows":
        offsets = torch.tensor([0, 2, 24], dtype=torch.int32, device="cuda")
    else:
        assert case == "overlap"
        keywords["out"] = rows[:2]

    with pytest.raises(ValueError, match=f"^{message}"):
        swage.segment_reduce(rows, offsets, "sum", **keywords)


def _logits(lengths, columns, seed):
    """Draw float32 logits with a standard deviation of four."""
    generator = torch.Generator().manual_seed(seed)
    values = 4 * torch.randn(sum(lengths), columns, generator=generator)
    return values, torch.tensor(_offsets(lengths), dtype=torch.int32)


def _softmax(host_values, host_offsets, **keywords):
    """Run the public softmax on fresh device tensors."""
    return swage.segment_softmax(
        host_values.cuda(), host_offsets.cuda(), **keywords
    ).cpu()


def _assert_softmax_columns_match(host_values, host_offsets, actual):
    """Compare a rank-two softmax with float64 `torch.softmax` along rows.

    Every output lies within the relative bound of
    docs/internals/ragged-softmax.md. The normalizer of a column of `n`
    rows is added by the stripes of one block, the `k` of `_block_depth`,
    and never more than `n - 1` additions of nonzero terms. One column runs
    the rank-one kernel and keeps the `k` of its tree. The outputs are
    normal f32 numbers, which the bound requires.
    """
    features = host_values.shape[1]
    assert actual.dtype == torch.float32
    assert actual.shape == host_values.shape
    for begin, end in pairwise(host_offsets.tolist()):
        if begin == end:
            continue
        logits = host_values[begin:end].double()
        reference = torch.softmax(logits, 0)
        assert (reference >= torch.finfo(torch.float32).tiny).all()
        relative = (actual[begin:end].double() - reference).abs() / reference
        depth = None
        if features > 1:
            depth = min(end - begin - 1, _block_depth(end - begin, features))
        bound = _softmax_bound(logits, reference, depth)
        assert (relative <= bound).all(), (
            f"rows [{begin}, {end}): relative error "
            f"{relative.max().item():.3e} exceeds its bound"
        )


@_needs_cuda
@pytest.mark.parametrize("features", FEATURES)
def test_softmax_columns_match_pytorch_along_the_rows_of_a_segment(features):
    """Normalize every column of every segment as `torch.softmax` does."""
    host_values, host_offsets = _logits(LENGTHS, features, features)

    actual = _softmax(host_values, host_offsets)

    _assert_softmax_columns_match(host_values, host_offsets, actual)


@_needs_cuda
def test_softmax_columns_agree_with_the_cpu_oracle():
    """Compare with the sequential lowering under the derived tolerance.

    The oracle adds the exponentials of a column in row order and the tile
    in the order of its stripes. They also differ in the exponential:
    `ex2.approx.f32` on the device and `exp2f` on the host. The tolerance
    that `test_segmented_runtime.py` derives covers any order of at most
    `n - 1` additions, so the comparison is not bitwise. The logits are
    quarter multiples with a spread of four, as that derivation needs, and
    depend on the row and on the column.
    """
    lengths = [0, 1, 2, 33, 150, 0, 7]
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    row = torch.arange(sum(lengths))[:, None]
    column = torch.arange(5)[None, :]
    host_values = ((row * 5 + column * 3) % 17 - 8).to(torch.float32) / 4

    actual = _softmax(host_values, host_offsets)

    expected = cpu_softmax_oracle(host_values, host_offsets)
    assert expected.shape == host_values.shape
    for index in range(host_values.shape[1]):
        _assert_softmax_matches_oracle(
            actual[:, index], expected[:, index], host_offsets
        )


@_needs_cuda
@pytest.mark.parametrize("features", [3, 129, 300])
def test_softmax_columns_read_and_write_their_own_rows_and_column(features):
    """Normalize logits that depend on the row and on the column.

    A result that took a neighboring column, or a window of rows moved by
    one, differs from the reference by far more than the bound. The columns
    beyond the block width run in the second pass of a thread.
    """
    lengths = [3, 0, 130, 1, 64, 2000]
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    row = torch.arange(sum(lengths))[:, None]
    column = torch.arange(features)[None, :]
    host_values = ((row * 31 + column * 17) % 127 - 63).to(torch.float32) / 8

    actual = _softmax(host_values, host_offsets)

    _assert_softmax_columns_match(host_values, host_offsets, actual)
    # Neighboring columns and neighboring rows do differ.
    beside = torch.isclose(actual[:, :-1], actual[:, 1:], rtol=1e-3)
    below = torch.isclose(actual[:-1], actual[1:], rtol=1e-3)
    assert not beside.all(0).any()
    assert not below.all(1).any()


@_needs_cuda
def test_softmax_columns_of_equal_logits_are_exactly_uniform():
    """Return exactly `1 / n` in every column of `n` equal logits.

    A shift by the maximum of equal logits is zero, `ex2.approx.f32` is
    exact at zero, and `n` ones add exactly in any order, so the only
    rounding is the division. One row gives exactly one.
    """
    lengths = [1, 2, 3, 7, 64, 100, 1]
    host_offsets = torch.tensor(_offsets(lengths), dtype=torch.int32)
    segment = torch.repeat_interleave(
        torch.arange(len(lengths)), torch.tensor(lengths)
    )
    column = torch.arange(130)[None, :]
    host_values = (segment[:, None] * 3 - column).to(torch.float32) / 4

    actual = _softmax(host_values, host_offsets)

    count = torch.tensor(lengths, dtype=torch.float32)[segment]
    expected = (1 / count)[:, None].expand_as(actual)
    assert _bits(actual) == _bits(expected.contiguous())


@_needs_cuda
@pytest.mark.parametrize("length", [2, 33, 129, 4097])
def test_softmax_columns_follow_pytorch_on_special_values(length):
    """Match `torch.softmax` where a column of a segment holds a special.

    A NaN, a positive infinity, or nothing but negative infinities makes
    every row of that column NaN in that segment, and no other column and
    no other segment. A negative infinity beside a finite maximum gives
    exactly zero.
    """
    ramp = torch.linspace(-2.0, 2.0, length)
    segment = torch.stack([ramp + column / 4 for column in range(6)], 1)
    special = segment.clone()
    special[length // 2, 1] = float("nan")
    special[-1, 2] = float("inf")
    special[:, 3] = float("-inf")
    special[0, 4] = float("-inf")
    host_values = torch.cat([segment, special, segment])
    host_offsets = torch.tensor(_offsets([length] * 3), dtype=torch.int32)

    actual = _softmax(host_values, host_offsets).reshape(3, length, 6)

    reference = torch.softmax(
        host_values.double().reshape(3, length, 6), 1
    )
    assert torch.equal(actual.isnan(), reference.isnan())
    assert actual[1][:, 1:4].isnan().all()
    assert actual.isnan().sum() == 3 * length
    assert actual[1, 0, 4] == 0.0
    finite = ~reference.isnan()
    torch.testing.assert_close(
        actual[finite].double(), reference[finite], rtol=1e-5, atol=0
    )


@_needs_cuda
def test_a_long_segment_of_few_logit_columns_keeps_the_softmax_bound():
    """Normalize 100,003 rows of three columns in one block.

    No segment of the softmax is split. Three columns are one group of
    four, so each column has 32 stripes, which walk about 3,126 rows each
    three times: for the maximum, for the sum, and for the store.
    """
    host_values, host_offsets = _logits([100_003, 2], 3, 3)

    actual = _softmax(host_values, host_offsets)

    _assert_softmax_columns_match(host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("block_size", [32, 64, 512, 1024])
def test_softmax_column_bits_do_not_depend_on_the_block_size(block_size):
    """A column is normalized by one thread, whatever the width of its block.

    The private launch of the column tile takes the block size; the public
    call launches the row-stripe tile at the CTA width.
    """
    host_values, host_offsets = _logits([0, 5, 300, 33], 129, 7)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    def run(width):
        output = torch.full((338, 129), float("nan"), device="cuda")
        launch_softmax_gpu(values, offsets, output, width)
        return output.cpu()

    _assert_same_results(run(block_size), run(128))


@_needs_cuda
def test_softmax_column_bits_do_not_depend_on_the_batch():
    """A segment alone has the bits it has among other segments.

    The call launches one task per segment, in segment order, whatever the
    batch.
    """
    lengths = [7, 4500, 0, 33]
    host_values, host_offsets = _logits(lengths, 6, 13)
    together = _softmax(host_values, host_offsets)

    for begin, end in pairwise(host_offsets.tolist()):
        alone = _softmax(
            host_values[begin:end],
            torch.tensor([0, end - begin], dtype=torch.int32),
        )
        _assert_same_results(alone, together[begin:end])


@_needs_cuda
def test_one_column_of_logits_takes_the_kernel_of_rank_one(monkeypatch):
    """Normalize `[N, 1]` values as the rank-one values they are.

    The result has the bits of the rank-one call on the same elements,
    with the shape `[N, 1]`, and the column kernel is never requested.
    """
    monkeypatch.setattr(
        qualification,
        "_launch_columns",
        lambda *arguments, **keywords: pytest.fail("the column kernel ran"),
    )
    host_values, host_offsets = _logits([0, 3, 40, 9000], 1, 17)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    column = swage.segment_softmax(values, offsets)
    flat = swage.segment_softmax(values.view(-1), offsets).cpu()
    out = torch.full((9043, 1), float("nan"), device="cuda")
    returned = swage.segment_softmax(values, offsets, out=out)

    assert column.shape == (9043, 1)
    assert returned is out
    _assert_same_results(column.cpu().view(-1), flat)
    _assert_same_results(out.cpu(), column.cpu())
    _assert_softmax_columns_match(host_values, host_offsets, column.cpu())


@_needs_cuda
@pytest.mark.parametrize("rows", [0, 6])
def test_no_column_of_logits_returns_an_empty_result_without_a_launch(
    rows, monkeypatch
):
    """Return `[N, 0]` for `[N, 0]` values and enqueue nothing."""
    driver = _runtime._get_driver()
    for name in ("launch_segmented", "launch_segmented_tasks"):
        monkeypatch.setattr(
            driver,
            name,
            lambda *arguments: pytest.fail("a kernel was launched"),
        )
    values = torch.empty(rows, 0, device="cuda")
    offsets = torch.tensor([0, 0, rows], dtype=torch.int32, device="cuda")
    out = torch.empty(rows, 0, device="cuda")
    version = out._version

    result = swage.segment_softmax(values, offsets)
    returned = swage.segment_softmax(values, offsets, out=out)

    assert result.shape == (rows, 0)
    assert result.dtype == torch.float32
    assert result.device == values.device
    assert returned is out
    assert out._version == version


@_needs_cuda
def test_logit_rows_without_a_segment_return_no_row():
    """Return `[0, D]` when there is no row and no segment."""
    values = torch.empty(0, 3, device="cuda")
    for segments in (0, 4):
        offsets = torch.zeros(segments + 1, dtype=torch.int32, device="cuda")

        result = swage.segment_softmax(values, offsets)

        assert result.shape == (0, 3)


@_needs_cuda
def test_a_rank_two_softmax_compiles_one_kernel(monkeypatch):
    """Compile the row-stripe kernel and launch one task per segment.

    The task list is the identity: the softmax classifies nothing, and its
    task-id kernel takes any segment length.
    """
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    host_values, host_offsets = _logits([3, 9000, 0], 4, 23)

    actual = _softmax(host_values, host_offsets)

    _assert_softmax_columns_match(host_values, host_offsets, actual)
    assert [
        (compile_ptx.__name__, dict(options))
        for compile_ptx, _, options in qualification._ptx_memo
    ] == [
        (
            "_compile_segmented_reduction_ptx",
            {
                "kernel_name": "ragged_softmax_r2",
                "block_size": 128,
                "target": qualification._target(torch, 0),
                "use_task_ids": True,
            },
        )
    ]


@_needs_cuda
def test_softmax_columns_take_int64_offsets_and_give_the_bits_of_int32():
    """Narrow int64 offsets on the host, as the rank-one calls do."""
    host_values, host_offsets = _logits(LENGTHS, 7, 29)
    values, offsets = host_values.cuda(), host_offsets.cuda()

    narrow = swage.segment_softmax(values, offsets).cpu()
    wide = swage.segment_softmax(values, offsets.long()).cpu()

    _assert_same_results(wide, narrow)


@_needs_cuda
def test_softmax_out_of_rows_is_written_in_place_and_its_version_advances():
    """Write a caller's `[N, D]` tensor and return it."""
    host_values, host_offsets = _logits([2, 0, 5], 3, 31)
    values, offsets = host_values.cuda(), host_offsets.cuda()
    out = torch.full((7, 3), -5.0, device="cuda")
    version = out._version

    returned = swage.segment_softmax(values, offsets, out=out)

    assert returned is out
    assert out._version > version
    _assert_softmax_columns_match(host_values, host_offsets, out.cpu())


def test_logits_of_another_rank_are_refused():
    """Admit rank one and rank two, and name both."""
    values, offsets = _host_rows()

    for wrong in (values[None], torch.tensor(1.0)):
        with pytest.raises(
            TypeError, match="^values must have rank one or two$"
        ):
            swage.segment_softmax(wrong, offsets)


@pytest.mark.parametrize(
    ("shape", "found"),
    [
        ((6,), r"\(6,\)"),
        ((18,), r"\(18,\)"),
        ((3, 6), r"\(3, 6\)"),
        ((4, 3), r"\(4, 3\)"),
        ((6, 3, 1), r"\(6, 3, 1\)"),
        ((6, 2), r"\(6, 2\)"),
    ],
)
def test_softmax_out_of_another_shape_is_refused_with_both_shapes(
    shape, found
):
    """Require the shape of the values, exactly."""
    values, offsets = _host_rows()

    with pytest.raises(
        ValueError,
        match=(
            r"^out must have shape \(6, 3\), the shape of values; found "
            rf"{found}$"
        ),
    ):
        swage.segment_softmax(values, offsets, out=torch.empty(shape))


def test_float64_logit_rows_are_refused_with_the_reason():
    """There is no float64 softmax kernel at rank two either."""
    values, offsets = _host_rows()

    with pytest.raises(TypeError, match="has no float64 kernel"):
        swage.segment_softmax(values.double(), offsets)


@_needs_cuda
@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("transposed", "values must be contiguous"),
        ("sliced-columns", "values must be contiguous"),
        ("one-sliced-column", "values must be contiguous"),
        ("transposed-out", "out must be contiguous"),
        ("offsets-past-rows", "final offset 24 exceeds value count 6"),
        (
            "offsets-short-of-rows",
            "offsets must end at the value count for a softmax: final "
            "offset 5, value count 6",
        ),
        ("overlap", "out must not overlap values"),
    ],
)
def test_softmax_layouts_outside_the_contract_are_refused(case, message):
    """Refuse a view that is not contiguous instead of copying it.

    The offsets delimit rows and cover every row: the row count is the
    value count of the refusal.
    """
    values = torch.arange(24, dtype=torch.float32, device="cuda")
    rows = values.reshape(6, 4)
    offsets = torch.tensor([0, 2, 6], dtype=torch.int32, device="cuda")
    keywords = {}
    if case == "transposed":
        rows, offsets = rows.t(), offsets.clamp(max=4)
    elif case == "sliced-columns":
        rows = rows[:, ::2]
    elif case == "one-sliced-column":
        rows = rows[:, 1:2]
    elif case == "transposed-out":
        keywords["out"] = torch.empty(4, 6, device="cuda").t()
    elif case == "offsets-past-rows":
        offsets = torch.tensor([0, 2, 24], dtype=torch.int32, device="cuda")
    elif case == "offsets-short-of-rows":
        offsets = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
    else:
        assert case == "overlap"
        keywords["out"] = rows

    with pytest.raises(ValueError, match=f"^{message}"):
        swage.segment_softmax(rows, offsets, **keywords)
