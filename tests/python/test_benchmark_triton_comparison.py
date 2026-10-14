# tests/python/test_benchmark_triton_comparison.py
"""Tests for the options, methods, and evidence of the comparison harness."""

import contextlib
import importlib
import itertools
import math
import pathlib
import types
from types import SimpleNamespace

import pytest
from benchmark_campaign_fixtures import TICKS, make_child, provenance

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SYNTHETIC_DISTRIBUTIONS = [
    "many-tiny",
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "few-huge",
    "one-outlier",
]
_TRACE = "soc-epinions1-outdegree-v1"
_MATCHED = "triton_matched_task_partition"


@pytest.fixture
def comparison(monkeypatch):
    """Import the comparison harness as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_triton_comparison")


def _offsets(torch, lengths):
    return torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int32)


class _Kernels:
    """CPU stand-ins that read the same windows as the Triton kernels.

    Args:
        looped_extra: Elements the looped kernel reads past a segment end.
        task_extra: Elements the looping task kernel reads past the end.
        task_shift: Offset of the window the looping task kernel reads for
            every segment that leaves room for it.
        fused_shift: The same offset for the windows of the fused kernel.
    """

    def __init__(
        self, looped_extra=0, task_extra=0, task_shift=0, fused_shift=0
    ):
        self._looped_extra = looped_extra
        self._task_extra = task_extra
        self._task_shift = task_shift
        self._fused_shift = fused_shift

    @staticmethod
    def _window(values, begin, end, shift, limit):
        """Sum up to ``limit`` elements from ``begin``, shifted if room."""
        if 0 <= begin + shift and end + shift <= values.numel():
            begin, end = begin + shift, end + shift
        return values[begin : min(end, begin + limit)].sum()

    @staticmethod
    def _launcher(body):
        class Kernel:
            def __getitem__(self, grid):
                return lambda *arguments, **options: body(
                    grid[0], *arguments, **options
                )

        return Kernel()

    @property
    def fixed(self):
        """One masked block per segment: it stops after BLOCK elements."""

        def body(programs, values, offsets, output, count, *, BLOCK, num_warps):
            for sid in range(programs):
                begin, end = int(offsets[sid]), int(offsets[sid + 1])
                output[sid] = values[begin : min(end, begin + BLOCK)].sum()

        return self._launcher(body)

    @property
    def looped(self):
        """A loop over the whole segment in blocks."""

        def body(programs, values, offsets, output, *, BLOCK, num_warps):
            for sid in range(programs):
                begin, end = int(offsets[sid]), int(offsets[sid + 1])
                stop = min(end + self._looped_extra, values.numel())
                output[sid] = values[begin:stop].sum() if end > begin else 0.0

        return self._launcher(body)

    @property
    def packed(self):
        """Warp tasks: one window of WARP elements per task."""

        def body(
            programs,
            values,
            offsets,
            output,
            ids,
            count,
            *,
            TASKS,
            WARP,
            num_warps,
        ):
            for sid in ids.tolist():
                begin, end = int(offsets[sid]), int(offsets[sid + 1])
                output[sid] = values[begin : min(end, begin + WARP)].sum()

        return self._launcher(body)

    @property
    def cta(self):
        """CTA tasks: one masked block per task, as the fixed kernel."""

        def body(
            programs, values, offsets, output, ids, count, *, BLOCK, num_warps
        ):
            for sid in ids.tolist():
                begin, end = int(offsets[sid]), int(offsets[sid + 1])
                output[sid] = values[begin : min(end, begin + BLOCK)].sum()

        return self._launcher(body)

    @property
    def fused(self):
        """Packed warp tasks, then CTA tasks of at most MAX_CTA_ELEMENTS."""

        def body(
            programs,
            values,
            offsets,
            output,
            warp_ids,
            warp_count,
            cta_ids,
            cta_count,
            *,
            WARP_PROGRAMS,
            LOGICAL_LANES,
            SHORT_TASK_SLOTS,
            WARP_LANES,
            MAX_CTA_ELEMENTS,
            num_warps,
        ):
            assert programs == WARP_PROGRAMS + cta_count
            for ids, limit in (
                (warp_ids, WARP_LANES),
                (cta_ids, MAX_CTA_ELEMENTS),
            ):
                for sid in ids.tolist():
                    begin, end = int(offsets[sid]), int(offsets[sid + 1])
                    output[sid] = self._window(
                        values, begin, end, self._fused_shift, limit
                    )

        return self._launcher(body)

    @property
    def cta_looped(self):
        """CTA tasks that loop over the whole segment in blocks."""

        def body(programs, values, offsets, output, ids, *, BLOCK, num_warps):
            for sid in ids.tolist():
                begin, end = int(offsets[sid]), int(offsets[sid + 1])
                shift = self._task_shift
                if 0 <= begin + shift and end + shift <= values.numel():
                    begin, end = begin + shift, end + shift
                stop = min(end + self._task_extra, values.numel())
                output[sid] = values[begin:stop].sum()

        return self._launcher(body)


def _reference_prepare(torch, calls=None):
    """Return a stand-in for the planned preparation that sums on the CPU."""

    def prepare(values, offsets, output, *, warp_max_elements):
        if calls is not None:
            calls.append(output)

        def launch():
            output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))

        return types.SimpleNamespace(warp=launch, cta=launch, mixed=launch)

    return prepare


def _not_run():
    return {"status": "not-run", "reason": "the CPU tests time nothing"}


def _row(comparison, torch, name, **changes):
    """Run one segmented row on the CPU with stand-in kernels."""
    options = {
        "segment_count": 96,
        "seed": 7,
        "values_kind": "quarters",
        "device": "cpu",
        "free_bytes": lambda: 1 << 30,
        "synchronize": lambda: None,
    }
    options.update(changes)
    kernels = options.pop("kernels", _Kernels())
    prepare = options.pop("prepare", _reference_prepare(torch))
    segment_count = options.pop("segment_count")
    seed = options.pop("seed")
    case = {
        "distribution": name,
        "seed": seed,
        "lengths": comparison.generate_lengths(name, segment_count, seed),
    }
    measured = []

    def measure(launches, useful_bytes):
        timings = {}
        for candidate, launch in launches.items():
            measured.append(useful_bytes)
            launch()
            timings[candidate] = {"call": {"summary_us": {"median": 1.0}}}
        return timings, {"base_candidate_order": list(launches)}

    def orchestrate(launches, lengths, values, offsets):
        assert lengths == case["lengths"]
        return {"planning": _not_run(), "end_to_end": _not_run()}

    row = comparison._segmented_row(
        torch, kernels, prepare, measure, orchestrate, case, **options
    )
    return row, measured


def test_default_arguments_keep_the_recorded_configuration(comparison):
    """Run the recorded campaign's rows unless an option says otherwise."""
    arguments = comparison._arguments(["--output", "x.json"])

    assert arguments.suite == "all"
    assert arguments.distributions == [*_SYNTHETIC_DISTRIBUTIONS, _TRACE]
    assert arguments.segment_count == 32_768
    assert arguments.seeds == [7]
    assert arguments.values == "ones"
    assert (arguments.samples, arguments.warmups) == (100, 25)


def test_arguments_select_scale_seeds_values_and_distributions(comparison):
    """Offer 10^6 segments, random values, seeds, and the heavy tail."""
    arguments = comparison._arguments(
        [
            "--output",
            "x.json",
            "--suite",
            "segmented-sum",
            "--distributions",
            "power-law",
            "alternating-empty",
            "--segment-count",
            "1000000",
            "--seeds",
            "7",
            "11",
            "13",
            "--values",
            "normal",
        ]
    )

    assert arguments.distributions == ["power-law", "alternating-empty"]
    assert arguments.segment_count == 1_000_000
    assert arguments.seeds == [7, 11, 13]
    assert arguments.values == "normal"


@pytest.mark.parametrize(
    "extra",
    [
        ["--distributions", "normal"],
        ["--distributions", "uniform", "--segment-count", "1000000"],
        ["--segment-count", "0"],
        ["--values", "zeros"],
        ["--samples", "0"],
        ["--warmups", "-1"],
        # The trace has 32,768 segments, and it is in the default run.
        ["--segment-count", "2048"],
        ["--distributions", "uniform", "uniform"],
        ["--seeds", "7", "7"],
    ],
)
def test_arguments_reject_what_cannot_be_measured(comparison, extra):
    """Refuse a configuration before any device work starts."""
    with pytest.raises(SystemExit):
        comparison._arguments(["--output", "x.json", *extra])


def test_values_are_seeded_per_kind(comparison):
    """Offer all ones, exact quarter multiples, and random normal values."""
    torch = pytest.importorskip("torch")

    ones = comparison._values(torch, "ones", 100, 7)
    quarters = comparison._values(torch, "quarters", 10_000, 7)
    normal = comparison._values(torch, "normal", 10_000, 7)

    assert torch.equal(ones, torch.ones(100))
    assert torch.equal(quarters, comparison._exact_values(torch, 10_000))
    assert quarters.min() == 0.25 and quarters.max() == 1.75
    assert normal.dtype == torch.float32
    assert torch.equal(normal, comparison._values(torch, "normal", 10_000, 7))
    assert not torch.equal(
        normal, comparison._values(torch, "normal", 10_000, 8)
    )
    assert not torch.equal(
        quarters, comparison._values(torch, "quarters", 10_000, 8)
    )
    assert abs(float(normal.mean())) < 0.05
    with pytest.raises(ValueError, match="zeros"):
        comparison._values(torch, "zeros", 1, 7)


def test_reference_is_exact_where_any_order_sums_are_exact(comparison):
    """Demand equality for quarter multiples whose sums fit 24 bits."""
    torch = pytest.importorskip("torch")
    lengths = [3, 0, 1, 500]
    offsets = _offsets(torch, lengths)
    values = comparison._values(torch, "quarters", sum(lengths), 7)

    reference, tolerance = comparison._sum_reference(
        torch, values, offsets, comparison._QUANTUM["quarters"]
    )

    assert reference.dtype == torch.float64
    assert torch.equal(
        reference,
        torch.segment_reduce(values.double(), "sum", offsets=offsets),
    )
    assert tolerance.tolist() == [0.0, 0.0, 0.0, 0.0]
    assert comparison._check_modes(tolerance) == {
        "exact_segments": 4,
        "bounded_segments": 0,
        "unchecked_segments": 0,
    }


def test_reference_bounds_random_values_by_the_any_order_error(comparison):
    """Allow what any summation order of f32 adds can differ by, no more."""
    torch = pytest.importorskip("torch")
    lengths = [1, 0, 2, 1000]
    offsets = _offsets(torch, lengths)
    values = comparison._values(torch, "normal", sum(lengths), 7)

    _, tolerance = comparison._sum_reference(
        torch, values, offsets, comparison._QUANTUM["normal"]
    )

    unit = 2.0**-24
    magnitude = torch.segment_reduce(
        values.double().abs(), "sum", offsets=offsets
    )
    assert tolerance[0] == 0.0
    assert tolerance[1] == 0.0
    assert tolerance[2] == pytest.approx(
        unit / (1 - unit) * float(magnitude[2])
    )
    assert tolerance[3] == pytest.approx(
        999 * unit / (1 - 999 * unit) * float(magnitude[3])
    )
    assert comparison._check_modes(tolerance) == {
        "exact_segments": 2,
        "bounded_segments": 2,
        "unchecked_segments": 0,
    }


def test_tolerance_follows_the_exactness_limit_of_the_values(comparison):
    """Switch from equality to the bound, then to unchecked, by length."""
    torch = pytest.importorskip("torch")
    lengths = torch.tensor(
        [2_000_000, 3_000_000, 1 << 24, (1 << 24) + 1, 1 << 23, (1 << 23) + 2]
    ).double()
    magnitude = lengths.clone()

    quarters = comparison._sum_tolerance(torch, lengths, magnitude * 1.75, 0.25)
    ones = comparison._sum_tolerance(torch, lengths, magnitude, 1.0)
    unit = 2.0**-24

    # 1.75 * 2,000,000 quarters-valued elements still fit 24 bits; 3,000,000
    # do not, and their bound is gamma(n - 1) times the sum of magnitudes.
    assert quarters[0] == 0.0
    steps = (3_000_000 - 1) * unit
    assert quarters[1] == pytest.approx(steps / (1 - steps) * 5_250_000.0)
    assert ones[:3].tolist() == [0.0, 0.0, 0.0]
    # Past half of 2 ** 24 elements the bound says nothing, so the segment
    # is reported as unchecked instead of being given a vacuous tolerance.
    assert math.isinf(ones[3])
    assert math.isfinite(quarters[4])
    assert math.isinf(quarters[5])
    assert comparison._check_modes(quarters) == {
        "exact_segments": 1,
        "bounded_segments": 2,
        "unchecked_segments": 3,
    }


def test_check_accepts_a_reordered_sum_and_rejects_a_misread(comparison):
    """Pass rounding differences, fail a missing element or an unset slot."""
    torch = pytest.importorskip("torch")
    lengths = [40, 7, 0, 300]
    offsets = _offsets(torch, lengths)
    values = comparison._values(torch, "normal", sum(lengths), 7)
    reference, tolerance = comparison._sum_reference(
        torch, values, offsets, None
    )
    reversed_sums = torch.stack(
        [
            values[begin:end].flip(0).sum()
            for begin, end in itertools.pairwise(offsets.tolist())
        ]
    )

    comparison._check_sums(torch, "kernel", reversed_sums, reference, tolerance)
    short = reversed_sums.clone()
    short[3] = values[int(offsets[3]) : int(offsets[4]) - 1].sum()
    with pytest.raises(AssertionError, match="kernel: 1 of 4 .* segment 3 "):
        comparison._check_sums(torch, "kernel", short, reference, tolerance)
    unset = reversed_sums.clone()
    unset[2] = float("nan")
    with pytest.raises(AssertionError, match="segment 2 "):
        comparison._check_sums(torch, "kernel", unset, reference, tolerance)
    unchecked = torch.full_like(tolerance, float("inf"))
    with pytest.raises(AssertionError, match="segment 2 "):
        comparison._check_sums(torch, "kernel", unset, reference, unchecked)


def test_effective_rate_counts_the_bytes_a_correct_sum_must_move(comparison):
    """Report GB/s over values and offsets read and sums written."""
    assert comparison._useful_bytes(1_000, 10) == 4 * (1_000 + 11 + 10)
    assert comparison._gb_per_s(35_343_248, 63.488) == pytest.approx(
        556.69, abs=0.01
    )
    assert comparison._gb_per_s(1_000, 0.0) is None


def test_batch_doubles_until_the_tick_is_below_one_percent(comparison):
    """Batch enough launches that one timer tick is under 1% of a sample."""
    batches = []

    def elapsed_us(launches):
        batches.append(launches)
        return 2.5 * launches

    # 32 launches take 80 us, 78 ticks of 1.024 us; 64 take 160 us.
    assert comparison._resolved_samples(elapsed_us, 3, 1.024) == (
        [2.5] * 3,
        64,
    )
    assert batches == [32] * 3 + [64] * 3
    assert comparison._resolved_samples(elapsed_us, 3, 0.032)[1] == 32
    assert comparison._resolved_samples(elapsed_us, 3, None)[1] == 32
    # Exactly one percent is not below one percent.
    assert (
        comparison._resolved_samples(lambda n: 100.0 * n / 32, 3, 1.0)[1] == 64
    )


def test_kept_samples_satisfy_the_limit_themselves(comparison):
    """Judge the batch by the samples that are kept, not by a pilot."""
    durations = iter([110.0, 101.0, 101.0, 210.0, 205.0, 204.0])

    timings, launches = comparison._resolved_samples(
        lambda launches: next(durations), 3, 1.024
    )

    # The first batch has one long sample, but its median of 101 us is
    # under 100 ticks, so the batch doubles and is sampled again.
    assert launches == 64
    assert timings == [210.0 / 64, 205.0 / 64, 204.0 / 64]


def test_batch_stops_at_a_timer_that_never_resolves(comparison):
    """Fail instead of looping when samples never outgrow the tick."""
    with pytest.raises(RuntimeError, match="below one percent"):
        comparison._resolved_samples(lambda launches: 0.0, 3, 1.0)


class _Event:
    """A CUDA event stand-in that reads a shared fake clock."""

    clock = [0.0]

    def __init__(self, enable_timing):
        assert enable_timing
        self._time = None

    def record(self):
        self._time = self.clock[0]

    def synchronize(self):
        pass

    def elapsed_time(self, end):
        return (end._time - self._time) / 1_000.0


def _event_torch():
    """Return a torch stand-in with fake events and a counting clock."""
    _Event.clock = [0.0]
    return types.SimpleNamespace(
        cuda=types.SimpleNamespace(Event=_Event, synchronize=lambda: None)
    )


def test_batched_event_timing_records_its_batch_and_resolution(comparison):
    """Carry the launches per sample and the tick fraction in the entry."""
    torch = _event_torch()

    def launch():
        _Event.clock[0] += 2.5

    timing = comparison._interleaved_batched_event_us(
        torch, {"a": launch}, 2, 4, tick_us=1.024
    )["a"]

    assert timing["launches_per_sample"] == 64
    assert timing["samples_us"] == [2.5] * 4
    assert timing["timer_tick_us"] == 1.024
    assert timing["tick_fraction_of_sample"] == pytest.approx(1.024 / 160)
    assert timing["tick_fraction_of_sample"] < 0.01

    untouched = comparison._interleaved_batched_event_us(
        torch, {"a": launch}, 2, 4
    )["a"]
    assert untouched["launches_per_sample"] == 32
    assert untouched["timer_tick_us"] is None
    assert untouched["tick_fraction_of_sample"] is None


def test_a_doubled_batch_keeps_every_candidate_interleaved(comparison):
    """Sample every candidate again, in rotating order, when one doubles."""
    order = []

    def elapsed_us(name, launches):
        order.append((name, launches))
        return {"fast": 0.5, "slow": 10.0}[name] * launches

    timings, batches = comparison._interleaved_batches(
        ["fast", "slow"], 2, 1.024, elapsed_us
    )

    # 32 fast launches take 16 us, below 100 ticks; 256 take 128 us.
    assert batches == {"fast": 256, "slow": 32}
    assert timings == {"fast": [0.5, 0.5], "slow": [10.0, 10.0]}
    assert order[-4:] == [
        ("fast", 256),
        ("slow", 32),
        ("slow", 32),
        ("fast", 256),
    ]
    assert len(order) == 4 * 4


def test_event_tick_is_measured_from_back_to_back_events(comparison):
    """Estimate the event timer tick on the device, not from a constant."""
    elapsed = itertools.cycle([3.072, 4.096, 3.072, 3.104, 5.12, 4.096])

    class Tensor:
        def add_(self, value):
            _Event.clock[0] += next(elapsed)

    torch = _event_torch()
    torch.zeros = lambda count, device: Tensor()

    # The finest gap is 0.032 us, but five readings in six are multiples
    # of 1.024 us, and that is the step a sample is resolved to.
    assert comparison._event_tick_us(torch, "cuda", pairs=12) == (
        pytest.approx(1.024)
    )


def test_padded_baseline_sums_a_masked_matrix(comparison):
    """Reduce a pad-to-max matrix exactly as the segment sums."""
    torch = pytest.importorskip("torch")
    lengths = [3, 0, 7, 1]
    offsets = _offsets(torch, lengths)
    values = comparison._values(torch, "quarters", sum(lengths), 7)

    padded, mask = comparison._padded_inputs(torch, values, offsets)

    assert padded.shape == mask.shape == (4, 7)
    assert mask.sum(dim=1).tolist() == lengths
    assert torch.equal(padded[~mask], torch.zeros(28 - sum(lengths)))
    assert torch.equal(
        comparison._padded_sum(padded, mask),
        torch.segment_reduce(values, "sum", offsets=offsets),
    )
    empty = comparison._padded_inputs(
        torch, values[:0], _offsets(torch, [0, 0])
    )
    assert comparison._padded_sum(*empty).tolist() == [0.0, 0.0]


@pytest.mark.parametrize(
    ("lengths", "expected_rows"),
    [
        ([0, 1, 3], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        ([0, 0, 0], [[], [], []]),
        ([], []),
    ],
)
def test_padded_reduction_and_storage(comparison, lengths, expected_rows):
    """Padding preserves exact sums and reports actual storage, even empty."""
    torch = pytest.importorskip("torch")
    offsets = _offsets(torch, lengths)
    values = torch.ones(sum(lengths))

    padded, _ = comparison._padded_inputs(torch, values, offsets)
    layout = comparison._padded_layout(lengths)
    output = torch.full((len(lengths),), float("nan"))
    torch.sum(padded, dim=1, out=output)

    assert output.tolist() == lengths
    assert padded.tolist() == expected_rows
    assert padded.is_contiguous()
    assert tuple(padded.shape) == (len(lengths), max(lengths, default=0))
    assert layout["storage_bytes"] == padded.untyped_storage().nbytes()
    assert layout["padded_elements"] == padded.numel()
    assert layout["padding_elements"] == padded.numel() - sum(lengths)
    assert layout["padding_fraction"] == (
        (padded.numel() - sum(lengths)) / padded.numel()
        if padded.numel()
        else 0.0
    )


@pytest.mark.parametrize(
    "name", [*_SYNTHETIC_DISTRIBUTIONS, "alternating-empty"]
)
def test_capped_rows_run_and_check_every_candidate(comparison, name):
    """Run every family, fixed to fused and looped, on capped lengths."""
    torch = pytest.importorskip("torch")

    row, measured = _row(comparison, torch, name)

    assert row["distribution"] == name
    assert row["seed"] == 7
    assert row["values"] == "quarters"
    assert row["segment_count"] == 96
    assert row["skipped"] == {}
    assert row["check"]["exact_segments"] == 96
    names = list(row["timings"])
    assert names[:6] == [
        "swage_warp",
        "swage_cta",
        "swage_mixed",
        "torch_segment_reduce",
        "torch_padded",
        "triton_fused",
    ]
    assert names[-1] == "triton_planned_looped_b1024_w8"
    assert sum(name.startswith("triton_looped_b") for name in names) == 15
    assert sum(name.startswith(_MATCHED) for name in names) == 4
    assert (
        sum(name.startswith("triton_planned_looped_b") for name in names) == 15
    )
    assert any(name.startswith("triton_b") for name in names)
    assert row["padded_layout"]["rows"] == 96
    assert row["useful_bytes"] == 4 * (row["statistics"]["total"] + 97 + 96)
    assert measured == [row["useful_bytes"]] * len(names)
    assert row["candidate_order"] == names
    assert row["excluded"] == []
    partition = row["matched_task_partition_triton"]
    assert partition["warp_tasks"] + partition["cta_tasks"] == 96
    assert row["triton_fused_contract"]["grid_programs"] == (
        (partition["warp_tasks"] + 3) // 4 + partition["cta_tasks"]
    )


def test_power_law_row_skips_the_baselines_that_cannot_cover_it(comparison):
    """Record why a baseline did not run instead of timing a wrong sum."""
    torch = pytest.importorskip("torch")

    row, _ = _row(
        comparison,
        torch,
        "power-law",
        segment_count=2048,
        free_bytes=lambda: 1000,
    )

    longest = row["statistics"]["max"]
    assert longest > 4096
    names = list(row["timings"])
    assert not any(name.startswith("triton_b") for name in names)
    assert not any(name.startswith(_MATCHED) for name in names)
    assert "triton_fused" not in names
    assert "torch_padded" not in names
    assert sum(name.startswith("triton_looped_b") for name in names) == 15
    # The looping matched comparator covers the row its one-block sibling
    # cannot: it packs the short tasks and loops over the long ones.
    assert (
        sum(name.startswith("triton_planned_looped_b") for name in names) == 15
    )
    assert set(row["skipped"]) == {
        "triton_fixed",
        _MATCHED,
        "triton_fused",
        "torch_padded",
    }
    assert str(longest) in row["skipped"]["triton_fixed"]
    assert "4096" in row["skipped"][_MATCHED]
    assert "4096" in row["skipped"]["triton_fused"]
    assert str(2048 * longest * 17) in row["skipped"]["torch_padded"]
    assert "1000" in row["skipped"]["torch_padded"]


def test_power_law_row_runs_the_padded_baseline_when_it_fits(comparison):
    """Pad to the longest segment when the device has room for it."""
    torch = pytest.importorskip("torch")

    row, _ = _row(comparison, torch, "power-law", segment_count=2048)

    assert "torch_padded" in row["timings"]
    assert set(row["skipped"]) == {"triton_fixed", _MATCHED, "triton_fused"}


def test_row_rejects_a_candidate_that_reads_past_a_segment(comparison):
    """Check every candidate before timing, on the timed values."""
    torch = pytest.importorskip("torch")

    with pytest.raises(AssertionError, match="triton_looped_b128_w1"):
        _row(comparison, torch, "bimodal", kernels=_Kernels(looped_extra=1))


def test_rows_differ_by_seed(comparison):
    """Draw new lengths for every seed of a distribution."""
    torch = pytest.importorskip("torch")

    first, _ = _row(comparison, torch, "bimodal", seed=7)
    second, _ = _row(comparison, torch, "bimodal", seed=8)

    assert first["statistics"] != second["statistics"]
    assert (first["seed"], second["seed"]) == (7, 8)


def test_timings_report_the_rate_beside_every_time(comparison, monkeypatch):
    """Add effective GB/s to each timing method that produced samples."""
    methods = {
        "call": {"summary_us": {"median": 50.0}},
        "batched_event": {"summary_us": {"median": 25.0}},
        "graph": {"available": False, "error": "capture failed"},
    }
    ticks = []
    for name, function in (
        ("call", "_interleaved_call_us"),
        ("batched_event", "_interleaved_batched_event_us"),
        ("graph", "_interleaved_graph_us"),
    ):
        monkeypatch.setattr(
            comparison,
            function,
            lambda torch, launches, warmups, samples, tick, name=name: (
                ticks.append((name, tick)) or {"a": dict(methods[name])}
            ),
        )

    timings, method = comparison._timings(
        None,
        {"a": None},
        1,
        2,
        ticks={"clock": 0.04, "event": 0.032},
        useful_bytes=1_000_000,
    )

    assert ticks == [
        ("call", 0.04),
        ("batched_event", 0.032),
        ("graph", 0.032),
    ]
    assert timings["a"]["call"]["effective_gb_per_s"] == pytest.approx(20.0)
    assert timings["a"]["batched_event"]["effective_gb_per_s"] == (
        pytest.approx(40.0)
    )
    assert "effective_gb_per_s" not in timings["a"]["graph"]
    assert method["base_candidate_order"] == ["a"]
    assert method["order_position_counts"] == {"a": [2]}


def test_family_drops_the_block_and_warp_suffix(comparison):
    """Name a whole sweep by its family in a candidate filter."""
    assert [
        comparison._family(name)
        for name in (
            "triton_b256_w8",
            "triton_planned_w4",
            "triton_looped_b128_w1",
            "triton_planned_looped_b1024_w8",
            "triton_matched_task_partition",
            "triton_matched_task_partition_w8",
            "triton_fused",
            "swage_mixed",
            "torch_padded",
        )
    ] == [
        "triton_fixed",
        "triton_planned",
        "triton_looped",
        "triton_planned_looped",
        _MATCHED,
        _MATCHED,
        "triton_fused",
        "swage_mixed",
        "torch_padded",
    ]


def test_selection_keeps_named_candidates_and_drops_excluded_ones(comparison):
    """Run only what is named, or everything but what is excluded."""
    names = [
        "swage_mixed",
        "torch",
        "triton_looped_b128_w1",
        "triton_looped_b256_w2",
        "triton_planned_looped_b128_w1",
        "torch_padded",
    ]

    assert comparison._select(names, None, []) == names
    assert comparison._select(names, ["torch", "triton_looped"], []) == [
        "torch",
        "triton_looped_b128_w1",
        "triton_looped_b256_w2",
    ]
    assert comparison._select(names, None, ["torch_padded", "torch"]) == [
        "swage_mixed",
        "triton_looped_b128_w1",
        "triton_looped_b256_w2",
        "triton_planned_looped_b128_w1",
    ]
    assert comparison._select(
        names, ["triton_looped"], ["triton_looped_b256_w2"]
    ) == ["triton_looped_b128_w1"]


def test_selectors_must_name_a_candidate_or_a_family(comparison):
    """Reject a misspelled selector instead of running without it."""
    names = comparison._segmented_candidates()

    comparison._check_selectors(
        ["torch_segment_reduce", "triton_planned_looped", _MATCHED], names
    )
    comparison._check_selectors(["triton_b256_w8"], names)
    with pytest.raises(ValueError, match="triton_loop.*triton_looped"):
        comparison._check_selectors(
            ["torch_segment_reduce", "triton_loop"], names
        )
    with pytest.raises(ValueError, match="torch"):
        comparison._check_selectors(["torch"], names)
    assert names[:4] == [
        "swage_warp",
        "swage_cta",
        "swage_mixed",
        "torch_segment_reduce",
    ]
    assert names[-1] == "triton_planned_looped_b1024_w8"
    assert len(names) == 3 + 1 + 1 + 1 + 26 + 4 + 15 + 15


def test_arguments_take_a_candidate_filter_for_the_segmented_suite(
    comparison,
):
    """Offer a filter and refuse one that names nothing or the wrong suite."""
    base = ["--output", "x.json", "--suite", "segmented-sum"]
    arguments = comparison._arguments(
        [
            *base,
            "--candidates",
            "swage_mixed",
            "triton_looped",
            "--exclude-candidates",
            "triton_looped_b128_w1",
        ]
    )

    assert arguments.candidates == ["swage_mixed", "triton_looped"]
    assert arguments.exclude_candidates == ["triton_looped_b128_w1"]
    default = comparison._arguments(base)
    assert default.candidates is None
    assert default.exclude_candidates == []
    for extra in (
        ["--candidates", "triton_loop"],
        ["--exclude-candidates", "pad_to_max"],
    ):
        with pytest.raises(SystemExit):
            comparison._arguments([*base, *extra])
    with pytest.raises(SystemExit):
        comparison._arguments(["--output", "x.json", "--candidates", "torch"])


def test_partition_matches_the_length_threshold(comparison):
    """Split the segment ids at 32 elements, as int32, in segment order."""
    torch = pytest.importorskip("torch")
    lengths = [0, 32, 33, 1, 4096, 5000, 7]

    warp_ids, cta_ids = comparison._partition_tasks(
        torch, _offsets(torch, lengths)
    )

    assert warp_ids.dtype == cta_ids.dtype == torch.int32
    assert warp_ids.tolist() == [0, 1, 3, 6]
    assert cta_ids.tolist() == [2, 4, 5]


def test_filtered_row_prepares_and_times_only_what_was_named(comparison):
    """Leave an unselected candidate out of setup, check, and timing."""
    torch = pytest.importorskip("torch")
    prepared = []

    row, measured = _row(
        comparison,
        torch,
        "bimodal",
        prepare=_reference_prepare(torch, prepared),
        only=["swage_mixed", "torch_segment_reduce", "triton_planned_looped"],
        exclude=["triton_planned_looped_b128_w1"],
    )

    names = list(row["timings"])
    assert names[:2] == ["swage_mixed", "torch_segment_reduce"]
    assert len(names) == 2 + 14
    assert all(name.startswith("triton_planned_looped_b") for name in names[2:])
    assert "triton_planned_looped_b128_w1" not in names
    assert row["candidate_order"] == names
    assert len(measured) == len(names)
    # One preparation, for the one Swage policy that is timed.
    assert len(prepared) == 1
    assert "swage_warp" in row["excluded"]
    assert "torch_padded" in row["excluded"]
    assert "triton_planned_looped_b128_w1" in row["excluded"]
    assert set(row["excluded"]).isdisjoint(names)
    assert row["skipped"] == {}


def test_excluding_the_padded_baseline_keeps_everything_else(comparison):
    """Drop one named candidate and nothing more."""
    torch = pytest.importorskip("torch")

    full, _ = _row(comparison, torch, "many-tiny")
    row, _ = _row(comparison, torch, "many-tiny", exclude=["torch_padded"])

    assert row["candidate_order"] == [
        name for name in full["candidate_order"] if name != "torch_padded"
    ]
    assert row["excluded"] == ["torch_padded"]


def test_a_filter_that_leaves_no_candidate_is_an_error(comparison):
    """Refuse a row that would time nothing."""
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="no candidate.*power-law"):
        _row(
            comparison,
            torch,
            "power-law",
            segment_count=2048,
            only=[_MATCHED, "triton_fused"],
        )


@pytest.mark.parametrize(
    ("kernels", "values_kind"),
    [
        (_Kernels(task_extra=1), "quarters"),
        (_Kernels(task_extra=1), "ones"),
        # On all-one values a shifted window of the right length sums to the
        # right value; the check on position-dependent values catches it.
        (_Kernels(task_shift=-1), "ones"),
    ],
)
def test_row_rejects_a_wrong_looping_task_kernel(
    comparison, kernels, values_kind
):
    """Check the looping matched comparator as strictly as the looped one."""
    torch = pytest.importorskip("torch")

    with pytest.raises(AssertionError, match="triton_planned_looped_b128_w1"):
        _row(
            comparison,
            torch,
            "bimodal",
            kernels=kernels,
            values_kind=values_kind,
        )


def test_a_family_left_out_by_option_is_not_reported_as_skipped(comparison):
    """Keep excluded, by option, apart from skipped, by inability."""
    torch = pytest.importorskip("torch")

    row, _ = _row(
        comparison,
        torch,
        "power-law",
        segment_count=2048,
        only=["torch_segment_reduce", "triton_looped_b256_w4", _MATCHED],
        free_bytes=lambda: 1000,
    )

    assert row["candidate_order"] == [
        "torch_segment_reduce",
        "triton_looped_b256_w4",
    ]
    # Asked for and unable to run: skipped. Not asked for: excluded,
    # whether or not the row could have run it.
    assert set(row["skipped"]) == {_MATCHED}
    assert {"torch_padded", "swage_mixed", "triton_fused"} <= set(
        row["excluded"]
    )
    assert not any(name.startswith(_MATCHED) for name in row["excluded"])


def test_position_dependent_check_covers_every_triton_family(comparison):
    """Catch a shifted fused window that all-one values cannot show."""
    torch = pytest.importorskip("torch")
    only = ["triton_fused", "torch_segment_reduce"]

    row, _ = _row(comparison, torch, "bimodal", values_kind="ones", only=only)
    assert row["candidate_order"] == ["torch_segment_reduce", "triton_fused"]
    # The timed all-one values pass the shifted window; the check on
    # position-dependent values, which B ran on the looped families only,
    # rejects it before timing.
    with pytest.raises(
        AssertionError, match="triton_fused on position-dependent values"
    ):
        _row(
            comparison,
            torch,
            "bimodal",
            kernels=_Kernels(fused_shift=-1),
            values_kind="ones",
            only=only,
        )


def test_segmented_cases_append_real_trace_with_provenance(comparison):
    """Keep seven synthetic cases unchanged, then append the real trace."""
    generated = []
    provenance = {"source": "sentinel"}
    real_lengths = [1, 1669]
    loaded = []

    def generate_lengths(name, count, seed):
        generated.append((name, count, seed))
        return [len(generated)]

    def load_real_trace(name):
        loaded.append(name)
        return real_lengths, provenance

    arguments = comparison._arguments(["--output", "x.json"])
    cases = list(
        comparison._segmented_cases(
            arguments.distributions,
            arguments.segment_count,
            arguments.seeds,
            generate_lengths,
            load_real_trace,
        )
    )

    assert [case["distribution"] for case in cases] == [
        *_SYNTHETIC_DISTRIBUTIONS,
        _TRACE,
    ]
    assert generated == [
        (name, comparison._SEGMENT_COUNT, comparison._SEED)
        for name in _SYNTHETIC_DISTRIBUTIONS
    ]
    assert all("trace_provenance" not in case for case in cases[:-1])
    assert loaded == [_TRACE]
    assert cases[-1]["lengths"] is real_lengths
    assert cases[-1]["trace_provenance"] is provenance


def test_segmented_cases_repeat_each_distribution_per_seed(comparison):
    """Run every seed of a distribution before the next distribution."""
    loaded = []
    cases = list(
        comparison._segmented_cases(
            ["uniform", _TRACE],
            8,
            [7, 11],
            lambda name, count, seed: [seed] * count,
            lambda name: loaded.append(name) or ([1, 2], {"trace": name}),
        )
    )

    assert [(case["distribution"], case["seed"]) for case in cases] == [
        ("uniform", 7),
        ("uniform", 11),
        (_TRACE, 7),
        (_TRACE, 11),
    ]
    assert cases[1]["lengths"] == [11] * 8
    assert loaded == [_TRACE]


def test_call_samples_use_balanced_rotating_order(comparison, monkeypatch):
    """Time one candidate per position before beginning the next round."""
    calls = []

    class FakeCuda:
        @staticmethod
        def synchronize():
            return None

    class FakeTorch:
        cuda = FakeCuda()

    clock = iter(range(1, 100))
    monkeypatch.setattr(
        comparison.time, "perf_counter_ns", lambda: next(clock) * 1_000
    )
    launches = {
        name: (lambda candidate=name: calls.append(candidate))
        for name in ("a", "b", "c")
    }

    result = comparison._interleaved_call_us(
        FakeTorch(), launches, warmups=0, samples=4, tick_us=0.5
    )

    assert calls == ["a", "b", "c", "b", "c", "a", "c", "a", "b"] + [
        "a",
        "b",
        "c",
    ]
    assert {name: len(entry["samples_us"]) for name, entry in result.items()}
    assert all(entry["launches_per_sample"] == 1 for entry in result.values())
    assert result["a"]["tick_fraction_of_sample"] == pytest.approx(0.5)
    orders = comparison._rotating_orders(launches, 8)
    counts = comparison._order_position_counts(orders)
    assert all(
        max(positions) - min(positions) <= 1 for positions in counts.values()
    )


def _fake_graph_torch(events, launch_ms=None):
    """Return a torch stand-in whose graphs and events record their use.

    Args:
        events: List that receives the captures and replays.
        launch_ms: Time of one captured launch, or None for one millisecond
            per replay whatever it holds.
    """
    replayed = {"launches": 0}

    class FakeGraph:
        def __init__(self, graph_id):
            self.graph_id = graph_id
            self.launches = 0

        def replay(self):
            replayed["launches"] = self.launches
            events.append(("replay", self.graph_id))

    class FakeEvent:
        def record(self):
            return None

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            del other
            if launch_ms is None:
                return 1.0
            return replayed["launches"] * launch_ms

    class FakeCuda:
        def __init__(self):
            self.graphs = []

        def synchronize(self):
            return None

        def CUDAGraph(self):
            graph = FakeGraph(len(self.graphs))
            self.graphs.append(graph)
            return graph

        def graph(self, graph):
            @contextlib.contextmanager
            def capture():
                events.append(("capture_start", graph.graph_id))
                begin = len(events)
                yield
                graph.launches = sum(
                    event[0] == "launch" for event in events[begin:]
                )
                events.append(("capture_end", graph.graph_id))

            return capture()

        def Event(self, enable_timing):
            assert enable_timing
            return FakeEvent()

    return SimpleNamespace(cuda=FakeCuda())


def test_all_graphs_are_captured_before_any_replay(comparison):
    """Prepare every candidate graph before interleaved replay begins."""
    events = []
    launches = {"a": lambda: None, "b": lambda: None}

    result = comparison._interleaved_graph_us(
        _fake_graph_torch(events), launches, warmups=0, samples=1
    )

    first_replay = next(
        index for index, event in enumerate(events) if event[0] == "replay"
    )
    capture_ends = [
        index for index, event in enumerate(events) if event[0] == "capture_end"
    ]
    assert len(capture_ends) == 2
    assert max(capture_ends) < first_replay
    assert result["a"]["available"] is True
    assert result["a"]["launches_per_sample"] == 32


def test_a_doubled_graph_batch_recaptures_before_replay(comparison):
    """Capture every graph of a new pass before the pass replays one."""
    events = []
    launches = {"a": lambda: events.append(("launch", "a"))}

    result = comparison._interleaved_graph_us(
        _fake_graph_torch(events, launch_ms=0.001),
        launches,
        warmups=0,
        samples=1,
        tick_us=0.5,
    )

    # A launch takes 1 us, so a replay of 32 is 64 ticks of 0.5 us, under
    # 100; the next pass captures 64 launches before it replays one.
    assert result["a"]["launches_per_sample"] == 64
    second = events.index(("capture_start", 1))
    assert events[second + 1 : second + 65] == [("launch", "a")] * 64
    assert events[second + 65] == ("capture_end", 1)
    assert events[-1] == ("replay", 1)


def test_pure_task_partition_boundary(comparison):
    """Keep the matched host partition boundary explicit."""
    assert comparison._partition_lengths([0, 1, 32, 33, 4096]) == (
        [0, 1, 2],
        [3, 4],
    )
    with pytest.raises(ValueError, match="up to 4096"):
        comparison._partition_lengths([4097])


@pytest.mark.parametrize(
    "invalid", ["missing", "file", "nonempty", "same", "alias"]
)
def test_compilation_rejects_unfresh_caches(
    comparison, monkeypatch, tmp_path, invalid
):
    """Missing, populated, or aliased cache locations cannot be called fresh."""
    swage_cache = tmp_path / "swage"
    triton_cache = tmp_path / "triton"
    swage_cache.mkdir()
    triton_cache.mkdir()
    monkeypatch.setenv("SWAGE_CACHE_DIR", str(swage_cache))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(triton_cache))
    if invalid == "missing":
        monkeypatch.delenv("TRITON_CACHE_DIR")
    elif invalid == "file":
        file = tmp_path / "cache-file"
        file.write_text("not a directory")
        monkeypatch.setenv("SWAGE_CACHE_DIR", str(file))
    elif invalid == "nonempty":
        (triton_cache / "previous-artifact").write_text("cached")
    elif invalid == "same":
        monkeypatch.setenv("TRITON_CACHE_DIR", str(swage_cache))
    else:
        alias = tmp_path / "alias"
        alias.symlink_to(swage_cache, target_is_directory=True)
        monkeypatch.setenv("TRITON_CACHE_DIR", str(alias))

    with pytest.raises(RuntimeError, match="empty directory|distinct"):
        comparison._require_empty_compilation_caches()


@pytest.fixture
def compilation_setup(comparison, monkeypatch, tmp_path):
    """Provide compiler clocks without CUDA execution or native builds."""
    for variable in ("SWAGE_CACHE_DIR", "TRITON_CACHE_DIR"):
        directory = tmp_path / variable
        directory.mkdir()
        monkeypatch.setenv(variable, str(directory))
    elapsed = [0]
    events = []
    calls = {
        name: []
        for name in ("fixed", "packed", "cta", "fused", "looped", "cta_looped")
    }
    monkeypatch.setattr(comparison.time, "perf_counter_ns", lambda: elapsed[0])

    def swage_steps(torch, cases):
        def compile_kernel():
            events.append("swage")
            elapsed[0] += 2_000

        return [
            (f"swage_{kernel}", compile_kernel)
            for kernel in ("warp", "cta", "mixed")
        ]

    monkeypatch.setattr(comparison, "_swage_compile_steps", swage_steps)

    class CompileOnlyKernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            pytest.fail("compile-only phase reached a launch")

        def warmup(self, *args, **kwargs):
            events.append(self.name)
            calls[self.name].append((args, kwargs))
            elapsed[0] += 1_000

    torch = SimpleNamespace(float32=object(), int32=object())
    kernels = SimpleNamespace(
        **{name: CompileOnlyKernel(name) for name in calls}
    )
    return comparison, torch, kernels, calls, events


def test_compilation_covers_scalar_specializations_without_launch(
    compilation_setup,
):
    """Compile every eligible signature, but skip empty partitions."""
    comparison, torch, kernels, calls, events = compilation_setup
    cases = [
        {"distribution": "tiny", "seed": 7, "lengths": [0, 1, 32]},
        {"distribution": "large", "seed": 7, "lengths": [33] * 16},
        {"distribution": "mixed", "seed": 7, "lengths": [1, 4096]},
        {"distribution": "tiny-repeat", "seed": 7, "lengths": [32, 1, 0]},
    ]
    result = comparison._measure_segmented_compilation(torch, cases, kernels)

    fixed_configs = [
        (block, warps)
        for block in (32, 64, 128, 256, 512, 1024, 2048, 4096)
        for warps in (1, 2, 4, 8)
        if warps <= block // 32
    ]
    expected_fixed = [
        (count, block, warps, (count,))
        for block, warps in fixed_configs
        for count, maximum in ((3, 32), (16, 33), (2, 4096), (3, 32))
        if block >= maximum
    ]
    assert [
        (args[3], options["BLOCK"], options["num_warps"], options["grid"])
        for args, options in calls["fixed"]
    ] == expected_fixed
    assert [
        (args[4], options["grid"]) for args, options in calls["packed"]
    ] == [(3, (1,)), (1, (1,)), (3, (1,))]
    assert [
        (args[4], options["num_warps"], options["grid"])
        for args, options in calls["cta"]
    ] == [
        (count, warps, (count,)) for warps in (1, 2, 4, 8) for count in (16, 1)
    ]
    assert [
        (args[4], args[6], options["WARP_PROGRAMS"], options["grid"])
        for args, options in calls["fused"]
    ] == [
        (3, 0, 1, (1,)),
        (0, 16, 0, (16,)),
        (1, 1, 1, (2,)),
        (3, 0, 1, (1,)),
    ]
    looped = comparison._triton_looped_configs()
    for name in ("looped", "cta_looped"):
        assert [
            (options["BLOCK"], options["num_warps"])
            for _, options in calls[name]
        ] == looped
    assert events == [
        *["swage"] * 3,
        *["fixed"] * len(expected_fixed),
        *["packed"] * 3,
        *["cta"] * 8,
        *["fused"] * 4,
        *["looped"] * 15,
        *["cta_looped"] * 15,
    ]
    assert result["timings"]["swage_total"]["samples_us"] == [6.0]
    assert result["timings"]["triton_total"]["samples_us"] == [
        float(len(expected_fixed) + 3 + 8 + 4 + 15 + 15)
    ]
    for component in result["component_order"]:
        assert result["timings"][component]["samples_us"][0] > 0

    # Validate the emitted phase against independently declared execution
    # geometry, without pretending the not-yet-run result rows are evidence.
    from benchmark_campaign import _compilation
    from benchmark_campaign_fixtures import methodology

    rows = [
        {
            "distribution": case["distribution"],
            "seed": 7,
            "statistics": {
                "total": sum(case["lengths"]),
                "max": max(case["lengths"]),
            },
            "segment_count": len(case["lengths"]),
            "matched_task_partition_triton": {
                "warp_tasks": sum(n <= 32 for n in case["lengths"]),
                "cta_tasks": sum(n > 32 for n in case["lengths"]),
            },
            "triton_fused_contract": {
                "warp_programs": (sum(n <= 32 for n in case["lengths"]) + 3)
                // 4
            },
        }
        for case in cases
    ]
    _compilation(result, rows, methodology(suite="segmented-sum"))


def test_compilation_follows_the_candidate_filter(compilation_setup):
    """Compile only what a timed candidate of some case launches."""
    comparison, torch, kernels, calls, events = compilation_setup
    cases = [
        {"distribution": "split", "seed": 7, "lengths": [1, 5000]},
        {"distribution": "short", "seed": 7, "lengths": [3, 40]},
    ]

    result = comparison._measure_segmented_compilation(
        torch,
        cases,
        kernels,
        only=[_MATCHED, "triton_looped_b256_w4"],
        exclude=["triton_matched_task_partition_w8"],
    )

    # The split case cannot run the matched partition, so only the short
    # case compiles it; no Swage candidate is timed, so nothing of Swage.
    assert "swage" not in events
    assert [args[4] for args, _ in calls["packed"]] == [1]
    assert [options["num_warps"] for _, options in calls["cta"]] == [1, 2, 4]
    assert [options["BLOCK"] for _, options in calls["looped"]] == [256]
    assert calls["fixed"] == calls["fused"] == calls["cta_looped"] == []
    assert result["derived_totals"] == {
        "triton_total": [
            "triton_matched_packed",
            "triton_matched_cta_w1",
            "triton_matched_cta_w2",
            "triton_matched_cta_w4",
            "triton_looped_b256_w4",
        ]
    }


def test_swage_compile_steps_follow_the_prepared_sum(comparison, monkeypatch):
    """Compile the kernels a preparation compiles, with its options."""
    from swage import _segmented_runtime

    compiled = []

    class Native:
        def _classify_segments(self, offsets, **counts):
            lengths = offsets[1:] - offsets[:-1]
            split = int((lengths > 4096).sum())
            direct = len(lengths) - split
            return None, direct, 0, split, split

        def __getattr__(self, name):
            return name

    blocks = SimpleNamespace(
        subgroup_width=32,
        cta_block_threads=128,
        default_warp_max_elements=32,
        default_cta_chunk_elements=4096,
    )
    monkeypatch.setattr(_segmented_runtime, "_native_swage", Native)
    monkeypatch.setattr(
        _segmented_runtime, "_target_description", lambda: blocks
    )
    monkeypatch.setattr(_segmented_runtime, "_target", lambda torch, i: "sm_86")
    monkeypatch.setattr(
        _segmented_runtime,
        "_compile_once",
        lambda compiler, text, **options: compiled.append((compiler, options)),
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(current_device=lambda: 0))

    short = comparison._swage_compile_steps(torch, [{"lengths": [1, 4096]}])
    split = comparison._swage_compile_steps(
        torch, [{"lengths": [1, 4096]}, {"lengths": [5000]}]
    )
    for _, compile_kernel in split:
        compile_kernel()

    assert [name for name, _ in short] == [
        "swage_warp",
        "swage_cta",
        "swage_mixed",
    ]
    assert [name for name, _ in split] == [
        "swage_warp",
        "swage_cta",
        "swage_mixed",
        "swage_partial",
        "swage_merge",
    ]
    common = {"kernel_name": "segmented_sum", "target": "sm_86"}
    assert compiled == [
        (
            "_compile_segmented_reduction_ptx",
            {**common, "block_size": 32, "use_task_ids": True},
        ),
        (
            "_compile_segmented_reduction_ptx",
            {**common, "block_size": 128, "use_task_ids": True},
        ),
        ("_compile_fused_segmented_reduction_ptx", common),
        ("_compile_split_partial_reduction_ptx", common),
        ("_compile_split_merge_reduction_ptx", common),
    ]


def test_swage_planning_classifies_and_uploads_without_execution(
    comparison, monkeypatch
):
    """Plan changing geometry with compilation, binding, and load forbidden."""
    torch = pytest.importorskip("torch")
    from swage import (
        _segmented_plan,
        _segmented_qualification,
        _segmented_runtime,
        _segmented_validation,
    )

    validate_shapes = _segmented_validation._validate_shapes
    monkeypatch.setattr(
        _segmented_validation,
        "_validate_shapes",
        lambda values, offsets, output, validator, **options: validate_shapes(
            values, offsets, output, validator, require_cuda=False
        ),
    )
    monkeypatch.setattr(
        _segmented_plan, "_planning_limits", lambda warp, cta: (warp, 4096)
    )
    admitted = []
    monkeypatch.setattr(
        _segmented_plan,
        "_admit_program",
        lambda *arguments: admitted.append(arguments[1:]),
    )

    def classifying_validator(warp_max, cta_chunk):
        found = []

        def validate(offsets, value_count, output_count):
            count = _segmented_validation._validate_offsets(
                offsets, value_count, output_count
            )
            lengths = offsets[1:] - offsets[:-1]
            warp = [i for i, n in enumerate(lengths) if n <= warp_max]
            cta = [i for i, n in enumerate(lengths) if n > warp_max]
            found.append((warp + cta, len(warp), len(cta), 0, 0))
            return count

        return validate, found

    monkeypatch.setattr(
        _segmented_plan, "_classifying_validator", classifying_validator
    )
    synchronized = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: synchronized.append(True)
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "planning reached compilation, binding, loading, launch, "
            "or output allocation"
        )

    for name in ("_compile_once", "_lease", "_bind", "_enqueue"):
        monkeypatch.setattr(_segmented_runtime, name, forbidden)
    monkeypatch.setattr(
        _segmented_qualification, "_prepare_planned_sum", forbidden
    )
    values = torch.ones(66, dtype=torch.float32)
    output = torch.full((4,), -1.0, dtype=torch.float32)
    offsets = torch.tensor([0, 0, 32, 65, 66], dtype=torch.int32)
    changed_offsets = torch.tensor([0, 33, 65, 65, 66], dtype=torch.int32)
    invalid_offsets = torch.tensor([0, 33, 32, 65, 66], dtype=torch.int32)
    monkeypatch.setattr(torch, "empty", forbidden)

    first = comparison._plan_swage_sum(torch, values, offsets, output)
    changed = comparison._plan_swage_sum(torch, values, changed_offsets, output)

    assert first.records.tolist() == [0, 1, 3, 2]
    assert changed.records.tolist() == [1, 2, 3, 0]
    assert first.counts == (3, 1, 0, 0)
    assert first.scratch is None
    assert output.tolist() == [-1.0] * 4
    assert synchronized == [True, True]
    assert admitted == [("segmented_sum", 32, 4096)] * 2
    with pytest.raises(ValueError, match="nondecreasing"):
        comparison._plan_swage_sum(torch, values, invalid_offsets, output)


def test_orchestration_is_not_run_without_its_candidates(comparison):
    """Mark planning and end to end as not run when neither is timed."""
    phases = comparison._orchestration_measurements(
        None, None, None, {"torch_segment_reduce": None}, [1], None, None, 1, 1
    )

    assert phases["planning"]["status"] == "not-run"
    assert "swage_mixed" in phases["planning"]["reason"]
    assert phases["end_to_end"] == phases["planning"]


def test_comparison_emits_schema_valid_evidence(
    comparison, monkeypatch, tmp_path
):
    """Framework string subclasses must produce schema-valid CLI evidence."""
    from benchmark_campaign import load_unique_json, validate_child
    from swage import _cuda_backend, env

    expected = make_child()

    class FrameworkVersion(str):
        pass

    cuda = SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda _: SimpleNamespace(
            multi_processor_count=84, total_memory=50_887_852_032
        ),
        get_device_capability=lambda _: (8, 6),
        get_device_name=lambda _: "NVIDIA RTX A6000",
    )
    monkeypatch.setitem(
        comparison.sys.modules,
        "torch",
        SimpleNamespace(
            __version__=FrameworkVersion("2.12.0+cu130"),
            version=SimpleNamespace(cuda="13.0"),
            cuda=cuda,
        ),
    )
    monkeypatch.setitem(
        comparison.sys.modules, "triton", SimpleNamespace(__version__="3.7.0")
    )
    monkeypatch.setattr(
        env,
        "report",
        lambda: {
            "native": {**expected["environment"]["compiler"], "available": True}
        },
    )
    monkeypatch.setattr(_cuda_backend, "driver_version", lambda: "13.0")
    monkeypatch.setattr(comparison, "_start_provenance", lambda _: provenance())
    monkeypatch.setattr(
        comparison.benchmark_provenance, "finish", lambda block: block
    )
    monkeypatch.setattr(
        comparison.benchmark_provenance,
        "clock_tick_us",
        lambda: TICKS["clock"],
    )
    monkeypatch.setattr(
        comparison, "_event_tick_us", lambda torch, device: TICKS["event"]
    )
    monkeypatch.setattr(
        comparison, "_git_metadata", lambda _: expected["source"]
    )
    monkeypatch.setattr(comparison, "_run_vadd", lambda *_: expected["results"])
    output = tmp_path / "child.json"

    comparison.main(
        [
            "--output",
            str(output),
            "--suite",
            "vadd",
            "--samples",
            "2",
            "--warmups",
            "1",
        ]
    )

    child = load_unique_json(output)
    validate_child(child)
    assert child["environment"]["pytorch"] == "2.12.0+cu130"
    assert child["schema_version"] == 2
    assert child["methodology"] == expected["methodology"]
