# tests/python/test_benchmark_triton_comparison.py
"""Tests for the options and methods of the comparison harness."""

import importlib
import itertools
import math
import pathlib
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DEFAULT_DISTRIBUTIONS = [
    "many-tiny",
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "few-huge",
    "one-outlier",
]


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
    """

    def __init__(self, looped_extra=0):
        self._looped_extra = looped_extra

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

        def body(programs, values, offsets, output, ids, count, *, TASKS,
                 WARP, num_warps):
            for sid in ids.tolist():
                begin, end = int(offsets[sid]), int(offsets[sid + 1])
                output[sid] = values[begin : min(end, begin + WARP)].sum()

        return self._launcher(body)

    @property
    def cta(self):
        """CTA tasks: one masked block per task, as the fixed kernel."""

        def body(programs, values, offsets, output, ids, count, *, BLOCK,
                 num_warps):
            for sid in ids.tolist():
                begin, end = int(offsets[sid]), int(offsets[sid + 1])
                output[sid] = values[begin : min(end, begin + BLOCK)].sum()

        return self._launcher(body)


def _reference_prepare(torch):
    """Return a stand-in for the planned preparation that sums on the CPU."""

    def prepare(values, offsets, output, *, warp_max_elements):
        def launch():
            output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))

        return types.SimpleNamespace(warp=launch, cta=launch, mixed=launch)

    return prepare


def _row(comparison, torch, name, **changes):
    """Run one segmented row on the CPU with stand-in kernels."""
    options = {
        "segment_count": 96,
        "seed": 7,
        "values_kind": "quarters",
        "device": "cpu",
        "memory_budget": 1 << 30,
        "synchronize": lambda: None,
    }
    options.update(changes)
    kernels = options.pop("kernels", _Kernels())
    measured = []

    def measure(launch, useful_bytes):
        measured.append(useful_bytes)
        launch()
        return {"call": {"summary_us": {"median": 1.0}}}

    row = comparison._segmented_row(
        torch, kernels, _reference_prepare(torch), measure, name, **options
    )
    return row, measured


def test_default_arguments_keep_the_recorded_configuration(comparison):
    """Run the recorded campaign's rows unless an option says otherwise."""
    arguments = comparison._arguments(["--output", "x.json"])

    assert arguments.suite == "all"
    assert arguments.distributions == _DEFAULT_DISTRIBUTIONS
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


def test_launches_double_until_the_tick_is_below_one_percent(comparison):
    """Batch enough launches that one timer tick is under 1% of a sample."""
    pilots = []

    def elapsed_us(launches):
        pilots.append(launches)
        return 2.5 * launches

    # 32 launches take 80 us, 78 ticks of 1.024 us; 64 take 160 us.
    assert comparison._launches_per_sample(elapsed_us, 1.024) == 64
    assert set(pilots) == {32, 64}
    assert comparison._launches_per_sample(elapsed_us, 0.032) == 32
    assert comparison._launches_per_sample(elapsed_us, None) == 32
    # Exactly one percent is not below one percent.
    assert comparison._launches_per_sample(lambda n: 100.0 * n / 32, 1.0) == 64


def test_launches_stop_at_a_timer_that_never_resolves(comparison):
    """Fail instead of looping when samples never outgrow the tick."""
    with pytest.raises(RuntimeError, match="below one percent"):
        comparison._launches_per_sample(lambda launches: 0.0, 1.0)


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

    timing = comparison._batched_event_us(torch, launch, 2, 4, tick_us=1.024)

    assert timing["launches_per_sample"] == 64
    assert timing["samples_us"] == [2.5] * 4
    assert timing["timer_tick_us"] == 1.024
    assert timing["tick_fraction_of_sample"] == pytest.approx(1.024 / 160)
    assert timing["tick_fraction_of_sample"] < 0.01

    untouched = comparison._batched_event_us(torch, launch, 2, 4)
    assert untouched["launches_per_sample"] == 32
    assert untouched["timer_tick_us"] is None
    assert untouched["tick_fraction_of_sample"] is None


def test_event_tick_is_measured_from_back_to_back_events(comparison):
    """Estimate the event timer tick on the device, not from a constant."""
    elapsed = itertools.cycle([3.072, 3.104, 3.2, 3.104])

    class Tensor:
        def add_(self, value):
            _Event.clock[0] += next(elapsed)

    torch = _event_torch()
    torch.zeros = lambda count, device: Tensor()

    assert comparison._event_tick_us(torch, "cuda", pairs=8) == (
        pytest.approx(0.032)
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


@pytest.mark.parametrize("name", [*_DEFAULT_DISTRIBUTIONS, "alternating-empty"])
def test_capped_rows_run_and_check_every_candidate(comparison, name):
    """Run fixed, planned, looped, padded, and PyTorch on capped lengths."""
    torch = pytest.importorskip("torch")

    row, measured = _row(comparison, torch, name)

    assert row["distribution"] == name
    assert row["seed"] == 7
    assert row["values"] == "quarters"
    assert row["segment_count"] == 96
    assert row["skipped"] == {}
    assert row["check"]["exact_segments"] == 96
    names = list(row["timings"])
    assert names[:4] == ["swage_warp", "swage_cta", "swage_mixed", "torch"]
    assert names[-1] == "torch_padded"
    assert sum(name.startswith("triton_looped_b") for name in names) == 15
    assert sum(name.startswith("triton_planned_w") for name in names) == 4
    assert any(name.startswith("triton_b") for name in names)
    assert row["useful_bytes"] == 4 * (row["statistics"]["total"] + 97 + 96)
    assert measured == [row["useful_bytes"]] * len(names)


def test_power_law_row_skips_the_baselines_that_cannot_cover_it(comparison):
    """Record why a baseline did not run instead of timing a wrong sum."""
    torch = pytest.importorskip("torch")

    row, _ = _row(
        comparison, torch, "power-law", segment_count=2048, memory_budget=1000
    )

    longest = row["statistics"]["max"]
    assert longest > 4096
    names = list(row["timings"])
    assert not any(name.startswith("triton_b") for name in names)
    assert not any(name.startswith("triton_planned") for name in names)
    assert "torch_padded" not in names
    assert sum(name.startswith("triton_looped_b") for name in names) == 15
    assert set(row["skipped"]) == {
        "triton_fixed",
        "triton_planned",
        "torch_padded",
    }
    assert str(longest) in row["skipped"]["triton_fixed"]
    assert "4096" in row["skipped"]["triton_planned"]
    assert str(2048 * longest * 17) in row["skipped"]["torch_padded"]
    assert "1000" in row["skipped"]["torch_padded"]


def test_power_law_row_runs_the_padded_baseline_when_it_fits(comparison):
    """Pad to the longest segment when the device has room for it."""
    torch = pytest.importorskip("torch")

    row, _ = _row(comparison, torch, "power-law", segment_count=2048)

    assert "torch_padded" in row["timings"]
    assert set(row["skipped"]) == {"triton_fixed", "triton_planned"}


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
    monkeypatch.setattr(
        comparison, "_call_us", lambda *a, **k: dict(methods["call"])
    )
    monkeypatch.setattr(
        comparison,
        "_batched_event_us",
        lambda *a, **k: dict(methods["batched_event"]),
    )
    monkeypatch.setattr(
        comparison, "_graph_us", lambda *a, **k: dict(methods["graph"])
    )

    timings = comparison._timings(
        None,
        None,
        1,
        2,
        ticks={"clock": 0.04, "event": 0.032},
        useful_bytes=1_000_000,
    )

    assert timings["call"]["effective_gb_per_s"] == pytest.approx(20.0)
    assert timings["batched_event"]["effective_gb_per_s"] == (
        pytest.approx(40.0)
    )
    assert "effective_gb_per_s" not in timings["graph"]
