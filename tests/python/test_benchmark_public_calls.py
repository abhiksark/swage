# tests/python/test_benchmark_public_calls.py
"""Tests for the public-call options of the fresh-offsets harness.

They cover the reduction kinds, rank-two values, float64 values, the
pipelined variant, and the record fields that keep the public call apart
from the private preparation candidates.
"""

import importlib
import itertools
import math
import pathlib
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_U32 = 2.0**-24
_U64 = 2.0**-53
_ROWS_CONFIGS = ("r4_w1", "r4_w4", "r16_w1", "r16_w4", "r64_w4")


@pytest.fixture
def fresh_offsets(monkeypatch):
    """Import the standalone benchmark as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_fresh_offsets")


@pytest.fixture
def comparison(monkeypatch):
    """Import the comparison harness, which holds the shared helpers."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_triton_comparison")


@pytest.fixture
def processes(monkeypatch):
    """Import the process driver."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_processes")


def _offsets(torch, lengths, dtype=None):
    dtype = dtype or torch.int32
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=dtype)


class _EmulatedRowsKernel:
    """Reduce each column block of each segment on the CPU, as the kernel.

    Args:
        extra_rows: Rows read past the end of every segment.
        shift_rows: Offset of the rows read, where the values leave room.
        shift_columns: Offset of the columns read, where they leave room.
    """

    def __init__(self, extra_rows=0, shift_rows=0, shift_columns=0):
        self._extra_rows = extra_rows
        self._shift_rows = shift_rows
        self._shift_columns = shift_columns
        self.launches = []

    def __getitem__(self, grid):
        def launch(values, offsets, output, features, *, KIND, FLOAT64,
                   BLOCK_ROWS, BLOCK_COLUMNS, num_warps):
            assert FLOAT64 is (values.dtype.itemsize == 8)
            import torch

            self.launches.append((grid, BLOCK_ROWS, BLOCK_COLUMNS, num_warps))
            segments, column_blocks = grid
            rows = values.shape[0]
            for segment in range(segments):
                begin, end = int(offsets[segment]), int(offsets[segment + 1])
                length = end - begin
                shift = self._shift_rows
                if 0 <= begin + shift and end + shift <= rows:
                    begin, end = begin + shift, end + shift
                end = min(end + self._extra_rows, rows)
                for block in range(column_blocks):
                    first = block * BLOCK_COLUMNS
                    last = min(first + BLOCK_COLUMNS, features)
                    columns = (
                        torch.arange(first, last) + self._shift_columns
                    ).clamp(0, features - 1)
                    window = values[begin:end][:, columns]
                    if KIND == 1:
                        result = (
                            window.max(dim=0).values
                            if len(window)
                            else torch.full((last - first,), -math.inf)
                        )
                    elif KIND == 2:
                        result = (
                            window.min(dim=0).values
                            if len(window)
                            else torch.full((last - first,), math.inf)
                        )
                    else:
                        result = window.sum(dim=0)
                        if KIND == 3:
                            result = result / length
                    output[segment, first:last] = result

        return launch


def _kernels(**changes):
    """Return CPU stand-ins for the Triton kernels of a rank-two row."""
    entries = {"rows": _EmulatedRowsKernel()}
    entries.update(changes)
    return types.SimpleNamespace(**entries)


def _swage(torch, calls=None, wrong=None):
    """Return stand-ins for the Swage entry points, the public call exact.

    Args:
        torch: The PyTorch module.
        calls: List that receives every public call's arguments.
        wrong: Callable that may change a result, for a broken call.
    """

    def public(values, offsets, kind, *, out):
        if calls is not None:
            calls.append((values, offsets, kind, out))
        result = torch.segment_reduce(
            values.double(), kind, offsets=offsets.long(), axis=0
        )
        if wrong is not None:
            result = wrong(result, kind)
        if result is not None:
            out.copy_(result)
        return out

    def prepare(values, offsets, output, *, warp_max_elements):
        def mixed():
            output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))

        return types.SimpleNamespace(mixed=mixed)

    def launch(values, offsets, output, kind):
        output.copy_(torch.segment_reduce(values, kind, offsets=offsets))

    return types.SimpleNamespace(prepare=prepare, launch=launch, public=public)


def _run(fresh_offsets, torch, name, **changes):
    """Run one row on the CPU with stand-in candidates."""
    options = {
        "device": "cpu",
        "synchronize": lambda: None,
        "free_bytes": lambda: 1 << 30,
        "clock_tick_us": 0.04,
        "warm_calls": 1,
    }
    options.update(changes)
    kernels = options.pop("kernels", None)
    swage = options.pop("swage", None) or _swage(torch)
    segment_count = options.pop("segment_count", 64)
    samples = options.pop("samples", 2)
    return fresh_offsets._run_distribution(
        torch, swage, kernels, name, segment_count, 1, samples, **options
    )


# ---- reference and check ----------------------------------------------------


@pytest.mark.parametrize("kind", ["sum", "max", "min", "mean"])
def test_reference_follows_the_semantics_of_each_kind(comparison, kind):
    """Give the float64 reference of every kind, empty segments included."""
    torch = pytest.importorskip("torch")
    lengths = [3, 0, 2]
    offsets = _offsets(torch, lengths)
    values = comparison._values(torch, "quarters", 5, 7).double()

    reference, tolerance = comparison._reduction_reference(
        torch, values, offsets, kind, quantum=0.25, dtype="float32"
    )

    expected = torch.segment_reduce(values, kind, offsets=offsets)
    assert reference.dtype == torch.float64
    assert torch.equal(reference.nan_to_num(), expected.nan_to_num())
    empty = {"sum": 0.0, "max": -math.inf, "min": math.inf}
    if kind == "mean":
        assert math.isnan(reference[1])
        # The sums are exact, so only the division rounds, once.
        assert tolerance.tolist() == pytest.approx(
            [2 * _U32 * abs(float(reference[0])), 0.0,
             2 * _U32 * abs(float(reference[2]))]
        )
    else:
        assert reference[1] == empty[kind]
        assert tolerance.tolist() == [0.0, 0.0, 0.0]


def test_reference_reduces_every_column_of_rank_two_values(comparison):
    """Reduce `[N, D]` values per column, as `torch.segment_reduce`."""
    torch = pytest.importorskip("torch")
    lengths = [4, 0, 3, 40]
    offsets = _offsets(torch, lengths)
    values = comparison._values(torch, "normal", 47 * 3, 7).reshape(47, 3)

    reference, tolerance = comparison._reduction_reference(
        torch, values.double(), offsets, "sum", quantum=None,
        dtype="float32",
    )

    assert reference.shape == tolerance.shape == (4, 3)
    assert torch.equal(
        reference,
        torch.segment_reduce(values.double(), "sum", offsets=offsets, axis=0),
    )
    magnitude = torch.segment_reduce(
        values.double().abs(), "sum", offsets=offsets, axis=0
    )
    steps = 39 * _U32
    assert tolerance[3].tolist() == pytest.approx(
        (steps / (1 - steps) * magnitude[3]).tolist()
    )
    assert tolerance[1].tolist() == [0.0, 0.0, 0.0]


def test_float64_tolerance_bounds_the_reference_too(comparison):
    """Allow the float64 reference its own rounding beside the candidate's."""
    torch = pytest.importorskip("torch")
    offsets = _offsets(torch, [1000])
    normal = comparison._values(torch, "normal", 1000, 7).double()
    quarters = comparison._values(torch, "quarters", 1000, 7).double()

    _, bounded = comparison._reduction_reference(
        torch, normal, offsets, "sum", quantum=None, dtype="float64"
    )
    _, exact = comparison._reduction_reference(
        torch, quarters, offsets, "sum", quantum=0.25, dtype="float64"
    )

    steps = 999 * _U64
    assert float(bounded[0]) == pytest.approx(
        2 * steps / (1 - steps) * float(normal.abs().sum())
    )
    assert exact.tolist() == [0.0]


def test_unwritten_fill_is_a_value_no_result_takes(fresh_offsets):
    """Fill with NaN, except for a mean, whose empty segments are NaN."""
    assert math.isnan(fresh_offsets._unwritten("sum"))
    assert math.isnan(fresh_offsets._unwritten("max"))
    assert math.isnan(fresh_offsets._unwritten("min"))
    assert fresh_offsets._unwritten("mean") == 1.0e30


def test_check_compares_every_kind_by_its_own_rule(comparison):
    """Pass rounding and special values; fail a misread or an unset slot."""
    torch = pytest.importorskip("torch")
    lengths = [40, 0, 7]
    offsets = _offsets(torch, lengths)
    values = comparison._values(torch, "normal", 47, 7)
    check = comparison._check_reduction

    for kind in ("sum", "max", "min", "mean"):
        reference, tolerance = comparison._reduction_reference(
            torch, values.double(), offsets, kind, quantum=None,
            dtype="float32",
        )
        flipped = torch.segment_reduce(
            values.flip(0), kind, offsets=_offsets(torch, lengths[::-1])
        ).flip(0)
        check(torch, "kernel", flipped, reference, tolerance, kind)
        wrong = flipped.clone()
        wrong[2] = flipped[2] + 0.5
        with pytest.raises(AssertionError, match="kernel: 1 of 3 .*segment 2"):
            check(torch, "kernel", wrong, reference, tolerance, kind)


def test_an_unwritten_mean_fails_where_nothing_else_is_checked(comparison):
    """Detect the marker of a mean even under an infinite tolerance."""
    torch = pytest.importorskip("torch")
    reference = torch.tensor([1.0, float("nan")], dtype=torch.float64)
    unchecked = torch.full((2,), math.inf, dtype=torch.float64)

    comparison._check_reduction(
        torch, "kernel", torch.tensor([5.0, math.nan]), reference,
        unchecked, "mean",
    )
    with pytest.raises(AssertionError, match="segment 0"):
        comparison._check_reduction(
            torch, "kernel", torch.tensor([1.0e30, math.nan]), reference,
            unchecked, "mean",
        )
    with pytest.raises(AssertionError, match="segment 1"):
        comparison._check_reduction(
            torch, "kernel", torch.tensor([1.0, 0.0]), reference,
            unchecked, "mean",
        )


def test_a_rank_two_failure_names_the_segment_and_the_column(comparison):
    """Point at the one wrong element of an `[S, D]` result."""
    torch = pytest.importorskip("torch")
    reference = torch.zeros(3, 4, dtype=torch.float64)
    result = torch.zeros(3, 4)
    result[1, 2] = 0.25

    with pytest.raises(AssertionError, match="segment 1, column 2 "):
        comparison._check_reduction(
            torch, "kernel", result, reference, torch.zeros(3, 4), "sum"
        )


def test_padded_bytes_follow_the_element_size(comparison):
    """Size the pad-to-max budget for the dtype of the values."""
    assert comparison._padded_bytes_per_element(4) == 17
    assert comparison._padded_bytes_per_element(8) == 29


# ---- arguments --------------------------------------------------------------


def test_defaults_keep_the_recorded_configuration(fresh_offsets):
    """Produce the recorded rows unless a new option is given."""
    arguments = fresh_offsets._arguments(["--output", "x.json"])

    assert arguments.kinds == ["sum"]
    assert arguments.features is None
    assert arguments.dtype == "float32"
    assert arguments.pipeline_depth == 0


def test_arguments_select_kinds_features_dtype_and_a_pipeline(
    fresh_offsets,
):
    """Take every new option together."""
    arguments = fresh_offsets._arguments(
        [
            "--output",
            "x.json",
            "--distributions",
            "power-law",
            "many-tiny",
            "--kinds",
            "sum",
            "max",
            "min",
            "mean",
            "--features",
            "3",
            "64",
            "--dtype",
            "float64",
            "--pipeline-depth",
            "8",
        ]
    )

    assert arguments.kinds == ["sum", "max", "min", "mean"]
    assert arguments.features == [3, 64]
    assert arguments.dtype == "float64"
    assert arguments.pipeline_depth == 8


@pytest.mark.parametrize(
    "extra",
    [
        ["--kinds", "prod"],
        ["--features", "0"],
        ["--dtype", "float16"],
        ["--pipeline-depth", "-1"],
        # 4096 x 2048 rows of 64 features may reach 2^29 elements.
        ["--distributions", "uniform", "--segment-count", "2048",
         "--features", "64"],
        ["--distributions", "power-law", "--features", "768"],
    ],
)
def test_arguments_reject_what_cannot_be_measured(fresh_offsets, extra):
    """Refuse a configuration before any device work starts."""
    with pytest.raises(SystemExit):
        fresh_offsets._arguments(["--output", "x.json", *extra])


@pytest.mark.parametrize(
    ("name", "count", "features", "admitted"),
    [
        ("power-law", 32_768, 64, True),
        ("many-tiny", 32_768, 64, True),
        ("one-outlier", 32_768, 64, True),
        ("alternating-empty", 32_768, 64, True),
        ("bimodal", 8_192, 64, True),
        ("few-huge", 8_192, 64, True),
        ("bimodal", 32_768, 64, False),
        ("uniform", 2_048, 64, False),
        ("uniform", 2_048, 3, True),
        ("power-law", 2_048, 768, True),
        ("power-law", 8_192, 768, False),
    ],
)
def test_rank_two_rows_are_admitted_by_their_worst_case_size(
    fresh_offsets, name, count, features, admitted
):
    """Bound the values of a rank-two row before generating any layout."""
    elements = fresh_offsets._rank_two_elements(name, count, features)

    assert (elements <= fresh_offsets._MAX_RANK_TWO_ELEMENTS) is admitted


# ---- candidate universes ----------------------------------------------------


def test_rank_one_candidates_are_the_recorded_ones(fresh_offsets):
    """Keep every rank-one name, in its order, for the recorded commands."""
    looped = [
        f"b{block}_w{warps}"
        for block in (128, 256, 512, 1024)
        for warps in (1, 2, 4, 8)
        if warps <= block // 32
    ]
    assert fresh_offsets._candidate_names(triton_available=True) == (
        "swage_mixed",
        "swage_cta_call",
        "swage_public_call",
        "swage_public_call_int64",
        "torch",
        "torch_pad_to_max",
        *(f"triton_looped_{config}" for config in looped),
        *(f"triton_planned_w{warps}" for warps in (1, 2, 4, 8)),
        *(f"triton_planned_looped_{config}" for config in looped),
    )


def test_rank_two_candidates(fresh_offsets, comparison):
    """Time the public call, torch along axis 0, and looped Triton."""
    names = fresh_offsets._candidate_names(triton_available=True, rank=2)

    assert names == (
        "swage_public_call",
        "swage_public_call_int64",
        "torch",
        *(f"triton_rows_looped_{config}" for config in _ROWS_CONFIGS),
    )
    assert fresh_offsets._candidate_names(
        triton_available=False, rank=2
    ) == names[:3]
    assert {comparison._family(name) for name in names[3:]} == {
        "triton_rows_looped"
    }
    arguments = fresh_offsets._arguments(
        ["--output", "x.json", "--distributions", "power-law",
         "--features", "3", "--candidates", "triton_rows_looped", "torch"]
    )
    assert arguments.candidates == ["triton_rows_looped", "torch"]


# ---- rank-one rows of other kinds and of float64 ----------------------------


@pytest.mark.parametrize("kind", ["max", "min", "mean"])
def test_a_rank_one_kind_row_times_the_public_call_and_torch(
    fresh_offsets, kind
):
    """Leave the float32-sum candidates out with the reason, not silently."""
    torch = pytest.importorskip("torch")
    calls = []

    row = _run(
        fresh_offsets, torch, "alternating-empty",
        swage=_swage(torch, calls), kind=kind,
    )

    assert row["candidates"] == [
        "swage_public_call",
        "swage_public_call_int64",
        "torch",
    ]
    assert set(row["skipped"]) == {
        "swage_mixed",
        "swage_cta_call",
        "torch_pad_to_max",
    }
    assert all("sum" in reason for reason in row["skipped"].values())
    assert (row["kind"], row["rank"], row["features"], row["dtype"]) == (
        kind, 1, None, "float32"
    )
    assert {call[2] for call in calls} == {kind}
    assert row["check"]["exact_segments"] + row["check"][
        "bounded_segments"
    ] == 3 * 64


def test_a_wrong_public_kind_is_rejected(fresh_offsets):
    """Check a minimum as a minimum."""
    torch = pytest.importorskip("torch")

    def as_maximum(result, kind):
        return result.abs() + 1.0

    # Both public candidates are wrong; the random order picks the first.
    with pytest.raises(
        AssertionError, match=r"swage_public_call(_int64)? on bimodal"
    ):
        _run(
            fresh_offsets, torch, "bimodal",
            swage=_swage(torch, wrong=as_maximum), kind="min",
        )


def test_a_public_mean_that_writes_nothing_is_rejected(fresh_offsets):
    """Catch an unwritten mean, though an empty mean is NaN."""
    torch = pytest.importorskip("torch")

    with pytest.raises(AssertionError, match="swage_public_call"):
        _run(
            fresh_offsets, torch, "alternating-empty",
            swage=_swage(torch, wrong=lambda result, kind: None),
            kind="mean", only=["swage_public_call"],
        )


def test_a_float64_sum_row_keeps_pad_to_max_and_sizes_it(fresh_offsets):
    """Time float64 sums with the public call, torch, and pad-to-max."""
    torch = pytest.importorskip("torch")
    calls = []

    row = _run(
        fresh_offsets, torch, "bimodal", swage=_swage(torch, calls),
        dtype="float64",
    )

    assert row["candidates"] == [
        "swage_public_call",
        "swage_public_call_int64",
        "torch",
        "torch_pad_to_max",
    ]
    assert set(row["skipped"]) == {"swage_mixed", "swage_cta_call"}
    assert {call[0].dtype for call in calls} == {torch.float64}
    assert {call[3].dtype for call in calls} == {torch.float64}
    assert row["pad_to_max"]["padded_bytes"] == (
        64 * row["pad_to_max"]["longest_segment"] * 29
    )
    assert row["dtype"] == "float64"


# ---- rank-two rows ----------------------------------------------------------


def test_a_rank_two_row_times_the_public_call_torch_and_triton(
    fresh_offsets,
):
    """Reduce `[N, D]` values per column with every rank-two candidate."""
    torch = pytest.importorskip("torch")
    calls = []
    kernel = _EmulatedRowsKernel()

    row = _run(
        fresh_offsets, torch, "power-law", swage=_swage(torch, calls),
        kernels=_kernels(rows=kernel), features=3,
    )

    assert row["candidates"] == [
        "swage_public_call",
        "swage_public_call_int64",
        "torch",
        *(f"triton_rows_looped_{config}" for config in _ROWS_CONFIGS),
    ]
    assert (row["rank"], row["features"]) == (2, 3)
    assert row["skipped"] == {}
    assert row["check"]["exact_segments"] == 3 * 64 * 3
    assert {tuple(call[0].shape[1:]) for call in calls} == {(3,)}
    assert {tuple(call[3].shape) for call in calls} == {(64, 3)}
    assert {call[1].dtype for call in calls} == {torch.int32, torch.int64}
    assert row["triton_rows_columns"] == {
        "block_columns": 4,
        "column_blocks": 1,
    }
    assert {launch[0] for launch in kernel.launches} == {(64, 1)}
    assert {launch[1:] for launch in kernel.launches} == {
        (4, 4, 1), (4, 4, 4), (16, 4, 1), (16, 4, 4), (64, 4, 4)
    }
    entry = row["iterations"][-1]
    rows = entry["layout_statistics"]["total"]
    assert entry["useful_bytes"] == 4 * rows * 3 + 4 * 65 + 4 * 64 * 3


def test_wide_features_split_into_column_blocks(fresh_offsets):
    """Cover 100 columns with two blocks of 64."""
    torch = pytest.importorskip("torch")
    kernel = _EmulatedRowsKernel()

    row = _run(
        fresh_offsets, torch, "many-tiny", kernels=_kernels(rows=kernel),
        features=100, only=["triton_rows_looped_r16_w4"], warm_calls=0,
    )

    assert row["triton_rows_columns"] == {
        "block_columns": 64,
        "column_blocks": 2,
    }
    assert {launch[0] for launch in kernel.launches} == {(64, 2)}


@pytest.mark.parametrize(
    "kernel",
    [
        _EmulatedRowsKernel(extra_rows=1),
        _EmulatedRowsKernel(shift_rows=-1),
        _EmulatedRowsKernel(shift_columns=1),
    ],
)
def test_a_wrong_rows_kernel_is_rejected(fresh_offsets, kernel):
    """Catch a row past the end, a shifted row, and a shifted column."""
    torch = pytest.importorskip("torch")

    with pytest.raises(
        AssertionError, match=r"triton_rows_looped_r\d+_w\d+ on bimodal"
    ):
        _run(
            fresh_offsets, torch, "bimodal", kernels=_kernels(rows=kernel),
            features=3, only=["triton_rows_looped"], warm_calls=0,
        )


@pytest.mark.parametrize("kind", ["sum", "max", "min", "mean"])
def test_rank_two_rows_take_every_kind_in_float64(fresh_offsets, kind):
    """Run the rank-two candidates on every kind and on float64."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets, torch, "alternating-empty", kernels=_kernels(),
        features=3, kind=kind, dtype="float64", values_kind="normal",
    )

    assert len(row["candidates"]) == 3 + 5
    assert (row["kind"], row["dtype"]) == (kind, "float64")


# ---- pipelined variant ------------------------------------------------------


def test_a_pipeline_sample_is_k_steps_between_two_synchronizes(
    fresh_offsets, monkeypatch
):
    """Enqueue K productions and calls with no wait between them."""
    torch = pytest.importorskip("torch")
    log = []
    now = [0]

    def advance(step, nanoseconds):
        log.append(step)
        now[0] += nanoseconds

    produce = fresh_offsets._produce_offsets

    def recording_produce(torch, layout):
        advance("produce", 1_000)
        produce(torch, layout)

    monkeypatch.setattr(fresh_offsets, "_produce_offsets", recording_produce)
    calls = []
    swage = _swage(torch, calls)
    public = swage.public

    def logged(values, offsets, kind, *, out):
        advance("call", 10_000)
        return public(values, offsets, kind, out=out)

    swage.public = logged

    row = _run(
        fresh_offsets, torch, "bimodal", swage=swage,
        only=["swage_public_call"], warm_calls=0, pipeline_depth=3,
        synchronize=lambda: advance("synchronize", 100_000),
        clock=lambda: now[0],
    )

    sample = ["synchronize", *["produce", "call"] * 3, "synchronize"]
    assert log == sample * 3
    # The span of three steps, 3 x 11 us plus the closing 100 us wait,
    # over three steps.
    assert row["raw_samples_us"]["swage_public_call"] == [
        pytest.approx((33_000 + 100_000) / 3 / 1_000)
    ] * 2
    assert row["pipeline_enqueue_samples_us"]["swage_public_call"] == [
        pytest.approx(33.0 / 3)
    ] * 2
    outputs = [call[3] for call in calls]
    assert len({id(output) for output in outputs[:3]}) == 3
    assert [id(output) for output in outputs[3:6]] == [
        id(output) for output in outputs[:3]
    ]
    assert row["pipeline_depth"] == 3
    assert row["swage_mixed_prepare_samples_us"] == []
    assert row["triton_planned_partition_samples_us"] == {}
    assert [len(entry["layout_seeds"]) for entry in row["iterations"]] == [
        3, 3, 3
    ]
    seeds = [seed for entry in row["iterations"] for seed in entry[
        "layout_seeds"
    ]]
    assert seeds == list(range(7, 16))


def test_a_pipeline_checks_the_result_of_every_step(fresh_offsets):
    """Fail a candidate that is wrong only in one step of the pipeline."""
    torch = pytest.importorskip("torch")
    count = [0]

    def wrong_on_third(result, kind):
        count[0] += 1
        return result + 1.0 if count[0] % 3 == 0 else result

    with pytest.raises(AssertionError, match="swage_public_call on bimodal"):
        _run(
            fresh_offsets, torch, "bimodal",
            swage=_swage(torch, wrong=wrong_on_third),
            only=["swage_public_call"], warm_calls=0, pipeline_depth=3,
        )


def test_a_pipeline_produces_the_offsets_on_the_device_each_step(
    fresh_offsets,
):
    """Recompute both offsets widths from the lengths of the layout."""
    torch = pytest.importorskip("torch")
    lengths = torch.tensor([3, 0, 5], dtype=torch.int32)
    layout = types.SimpleNamespace(
        lengths=lengths,
        offsets=torch.full((4,), -1, dtype=torch.int32),
        long_offsets=torch.full((4,), -1, dtype=torch.int64),
    )
    layout.offsets[0] = 0
    layout.long_offsets[0] = 0

    fresh_offsets._produce_offsets(torch, layout)

    assert layout.offsets.tolist() == [0, 3, 3, 8]
    assert layout.long_offsets.tolist() == [0, 3, 3, 8]


def test_a_pipelined_rank_two_row_runs(fresh_offsets):
    """Pipeline `[N, D]` calls as well."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets, torch, "power-law", kernels=_kernels(), features=4,
        pipeline_depth=2, kind="mean",
    )

    assert row["pipeline_depth"] == 2
    assert set(row["pipeline_enqueue_samples_us"]) == set(row["candidates"])


# ---- record -----------------------------------------------------------------


def test_default_rows_state_their_identity(fresh_offsets):
    """Carry kind, rank, features, dtype, and depth on every row."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets, torch, "bimodal", only=["swage_public_call", "torch"],
        swage=_swage(torch),
    )

    assert (
        row["kind"], row["rank"], row["features"], row["dtype"],
        row["pipeline_depth"],
    ) == ("sum", 1, None, "float32", 0)


def test_configuration_keeps_public_and_private_candidates_apart(
    fresh_offsets,
):
    """Describe the surface each candidate times, and every option."""
    configuration = fresh_offsets._configuration(
        segment_count=2048,
        warmups=1,
        samples=3,
        triton_available=True,
        kinds=["sum", "mean"],
        features=[3, 64],
        dtype="float64",
        pipeline_depth=8,
    )

    described = configuration["candidate_descriptions"]
    assert described["swage_public_call"]["surface"] == "public"
    assert "swage.segment_reduce" in described["swage_public_call"]["entry"]
    assert described["swage_public_call_int64"]["surface"] == "public"
    for private in ("swage_mixed", "swage_cta_call"):
        assert described[private]["surface"] == "private"
        assert "_segmented_qualification" in described[private]["entry"]
    for baseline in (
        "torch",
        "torch_pad_to_max",
        "triton_looped",
        "triton_planned",
        "triton_planned_looped",
        "triton_rows_looped",
    ):
        assert described[baseline]["surface"] == "baseline"
    assert "axis=0" in described["torch"]["entry"]
    assert configuration["kinds"] == ["sum", "mean"]
    assert configuration["features"] == [3, 64]
    assert configuration["dtype"] == "float64"
    assert configuration["pipeline"]["depth"] == 8
    assert "cumsum" in configuration["pipeline"]["step"]
    assert "synchronize" in configuration["pipeline"]["timer"]
    assert configuration["candidates_by_rank"]["2"][0] == "swage_public_call"


def test_options_record_every_argument(fresh_offsets):
    """Write every option of the run into the record."""
    arguments = fresh_offsets._arguments(
        ["--output", "x.json", "--distributions", "many-tiny",
         "--features", "3", "--kinds", "max"]
    )

    options = fresh_offsets._options(arguments)

    assert options["output"] == "x.json"
    assert options["features"] == [3]
    assert options["kinds"] == ["max"]
    assert set(options) == set(vars(arguments))


# ---- process driver ---------------------------------------------------------


def _fresh_row(distribution, **identity):
    return {
        "distribution": distribution,
        **identity,
        "summary_us": {"torch": {"median": 10.0}},
        "effective_gb_per_s": {"torch": {"median": 1.0}},
    }


def test_driver_labels_keep_default_rows_and_name_the_new_ones(processes):
    """Tell the new rows apart without renaming the recorded ones."""
    record = {
        "benchmark": "fresh-offsets-segmented-sum",
        "results": [
            _fresh_row("power-law"),
            _fresh_row(
                "power-law", kind="sum", rank=1, features=None,
                dtype="float32", pipeline_depth=0,
            ),
            _fresh_row("power-law", kind="max", rank=1, features=None,
                       dtype="float32", pipeline_depth=0),
            _fresh_row("power-law", kind="mean", rank=2, features=64,
                       dtype="float64", pipeline_depth=0),
            _fresh_row("many-tiny", kind="sum", rank=2, features=3,
                       dtype="float32", pipeline_depth=8),
        ],
    }

    series = processes._series(record)

    assert set(series) == {
        "power-law",
        "power-law max",
        "power-law D=64 mean float64",
        "many-tiny D=3",
    }
    assert set(series["power-law max"]) == {"end_to_end"}
    assert set(series["many-tiny D=3"]) == {"pipelined"}


def test_mean_tolerance_is_the_sum_bound_over_the_length(comparison):
    """Divide the bound of the sum by n, then add the division's rounding."""
    torch = pytest.importorskip("torch")
    offsets = _offsets(torch, [40])
    values = comparison._values(torch, "normal", 40, 7).double()

    _, sum_tolerance = comparison._reduction_reference(
        torch, values, offsets, "sum", quantum=None, dtype="float32"
    )
    reference, mean_tolerance = comparison._reduction_reference(
        torch, values, offsets, "mean", quantum=None, dtype="float32"
    )

    assert float(mean_tolerance[0]) == pytest.approx(
        float(sum_tolerance[0]) / 40 + 2 * _U32 * abs(float(reference[0]))
    )


@pytest.mark.parametrize("kind", ["max", "min"])
def test_an_extreme_is_exact_whatever_the_values(comparison, kind):
    """Give a maximum and a minimum no tolerance, also on random values."""
    torch = pytest.importorskip("torch")
    offsets = _offsets(torch, [40, 0, 7])
    values = comparison._values(torch, "normal", 47, 7).double()

    _, tolerance = comparison._reduction_reference(
        torch, values, offsets, kind, quantum=None, dtype="float32"
    )

    assert tolerance.tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    ("features", "columns"),
    [(1, (4, 1)), (2, (4, 1)), (5, (8, 1)), (64, (64, 1)), (65, (64, 2)),
     (768, (64, 12))],
)
def test_column_blocks_are_floored_and_capped(comparison, features, columns):
    """Keep the column block between 4 and 64 lanes."""
    assert comparison._rows_columns(features) == columns


def test_a_pipelined_row_records_no_preparation_or_partition(fresh_offsets):
    """Time no part of a step on its own when the steps overlap."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets, torch, "bimodal", only=["swage_mixed"],
        pipeline_depth=2,
    )

    assert row["candidates"] == ["swage_mixed"]
    assert row["swage_mixed_prepare_samples_us"] == []
    assert row["triton_planned_partition_samples_us"] == {}
