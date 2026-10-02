# python/tests/mlir/test_segment_columns.py
"""Rank-two values in `swage.segment_reduce`.

`[N, D]` values are `N` rows of `D` features. The offsets delimit rows, and
every column of a segment is reduced on its own, as `torch.segment_reduce`
does along axis 0. One kernel serves a call: one block per segment, in
which a thread reduces a column in row order. The tests here pin:

- Agreement with `torch.segment_reduce(axis=0)` at 1, 3, 64, 129, 200, and
  1024 columns, for every kind and both dtypes, with a maximum and a
  minimum exact and a sum inside the `(n - 1)` bound of a sequential sum.
- Bit equality with the CPU oracle, which adds a column in the same order.
- That a result reads its own rows and its own column, on values that
  depend on both.
- `[N, 1]` values through the schedules of rank one, and `[N, 0]` values
  without a launch.
- The refusals of the shape rules.

The device-side bounds of the column kernel are in
`test_segmented_bounds.py`.
"""

import fractions
import math
from itertools import pairwise

import pytest
import swage
import torch
from swage import _runtime
from swage import _segmented_qualification as qualification
from swage._segmented_qualification import cpu_oracle, launch_gpu
from test_segmented_runtime import _EPS32, _EPS64, _bits, _offsets

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

    A maximum and a minimum are exact. A column sum is added in row order,
    one addition per row, so it lies within `(n - 1) * eps * sum(|x|)` of
    the exact sum of its `n` rows. A mean is the sum of the same call
    divided by the number of rows, bit for bit, and NaN for no rows.
    """
    lengths = (host_offsets[1:] - host_offsets[:-1]).long()
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
    depth = (lengths - 1).clamp(min=0).double()[:, None]
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
        "eps * sum(|x|) exceeds the sequential bound"
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
    than the sequential one for every length here.
    """
    host_values, host_offsets = _rows(LENGTHS, features, features, dtype)

    actual = _reduce(kind, host_values, host_offsets)

    _assert_columns_match(kind, host_values, host_offsets, actual)


@_needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=_DTYPE_IDS)
@pytest.mark.parametrize("kind", KINDS)
def test_columns_equal_the_cpu_oracle_bit_for_bit(kind, dtype):
    """A thread adds its column in row order, the order of the oracle.

    The values are not exactly summable, so two orders would differ. The
    oracle lowers the same program through the sequential schedule and runs
    it on the host.
    """
    lengths = [0, 1, 2, 33, 150, 0, 7]
    host_values, host_offsets = _rows(lengths, 5, 11, dtype)

    actual = _reduce(kind, host_values, host_offsets)

    _assert_same_results(actual, cpu_oracle(host_values, host_offsets, kind))


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
def test_a_long_segment_of_few_columns_keeps_the_sequential_bound(dtype):
    """Reduce 100,003 rows of three columns in three threads.

    No segment of rank-two values is split, so each of the three threads
    adds 100,003 rows one after the other. The sum keeps the bound of a
    sequential sum, and the mean divides it once.
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
    `(k + 1) * eps * sum(|x|) / n` for the `k = n - 1` additions of a
    column and one unit for the division and the conversion of `n`.
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

    The private launch takes the block size. With 32 threads a thread
    reduces five of the 129 columns, and with 1024 threads at most one.
    Every width returns the bits of the default width.
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

    A rank-two call has one schedule, so nothing about the other segments
    of a batch, their number, or the device changes the order in which a
    column is added.
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
@pytest.mark.parametrize("kind", KINDS)
def test_one_column_takes_the_schedules_of_rank_one(kind, monkeypatch):
    """Reduce `[N, 1]` values as the rank-one values they are.

    The result has the bits of the rank-one call on the same elements,
    with the shape `[S, 1]`, and the column kernel is never requested: the
    long segment is split, which no rank-two kernel does.
    """
    monkeypatch.setattr(
        qualification,
        "_launch_columns",
        lambda *arguments, **keywords: pytest.fail("the column kernel ran"),
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
    monkeypatch.setattr(
        driver,
        "launch_segmented",
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
def test_a_rank_two_call_classifies_nothing_and_compiles_one_kernel(
    monkeypatch,
):
    """Validate the offsets, compile the column kernel, and launch it.

    The host work of a rank-two call is the validation of its offsets: no
    program is admitted for planning and no segment is classified, also
    for a segment that a rank-one call would split.
    """
    for name in ("_admit_program", "_classifying_validator"):
        monkeypatch.setattr(
            qualification,
            name,
            lambda *arguments, **keywords: pytest.fail("the call planned"),
        )
    monkeypatch.setattr(
        qualification,
        "_ptx_memo",
        _runtime._BoundedCache(_runtime._CACHE_LIMIT),
    )
    host_values, host_offsets = _rows([3, 9000, 0], 4, 23, torch.float32)

    actual = _reduce("sum", host_values, host_offsets)

    _assert_columns_match("sum", host_values, host_offsets, actual)
    assert [
        (compile_ptx.__name__, dict(options)["kernel_name"])
        for compile_ptx, _, options in qualification._ptx_memo
    ] == [("_compile_segmented_reduction_ptx", "segmented_sum_r2")]


@_needs_cuda
@pytest.mark.parametrize("kind", KINDS)
def test_columns_take_int64_offsets_and_give_the_bits_of_int32(kind):
    """Narrow int64 offsets on the host, as the rank-one calls do."""
    host_values, host_offsets = _rows(LENGTHS, 7, 29, torch.float64)
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
