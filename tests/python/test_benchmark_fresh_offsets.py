# tests/python/test_benchmark_fresh_offsets.py
"""Tests for the fresh-offsets benchmark and the looped Triton baseline."""

import collections
import importlib
import itertools
import pathlib
import subprocess
import sys
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_LOOPED_BLOCKS = (128, 256, 512, 1024)


@pytest.fixture
def fresh_offsets(monkeypatch):
    """Import the standalone benchmark as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_fresh_offsets")


@pytest.fixture
def triton_comparison(monkeypatch):
    """Import the comparison harness as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_triton_comparison")


def _stub_torch():
    """Return the device-identity surface the environment record reads."""
    properties = types.SimpleNamespace(
        multi_processor_count=84, total_memory=51_527_024_640
    )
    return types.SimpleNamespace(
        __version__="2.12.0+cu130",
        version=types.SimpleNamespace(cuda="13.0"),
        cuda=types.SimpleNamespace(
            current_device=lambda: 0,
            get_device_name=lambda device: "NVIDIA RTX A6000",
            get_device_capability=lambda device: (8, 6),
            get_device_properties=lambda device: properties,
        ),
    )


def _environment(fresh_offsets, triton_version="3.7.0"):
    """Build one complete environment record from the stub device."""
    return fresh_offsets._environment(
        _stub_torch(),
        cuda_driver="13.0",
        nvidia_driver="580.178.04",
        triton_version=triton_version,
    )


_IDENTITY = {
    "revision": "abc123",
    "clean": True,
    "llvm": "llvmorg-22.1.8",
    "frontend": "f" * 64,
    "native": [
        ["_swageDialectsNanobind.cpython-313-x86_64-linux-gnu.so", 10, 20],
        ["libSwagePythonCAPI.so.22.1", 30, 40],
    ],
}


def _imported_code(fresh_offsets, **changes):
    """Build the imported-code block of a run from one consistent checkout."""
    root = pathlib.Path("/work/swage")
    libraries = root / "build/python_packages/mlir_swage/_mlir_libs"
    fields = {
        "root": root,
        "package": root / "python/swage",
        "same_checkout": True,
        "identity": dict(_IDENTITY),
        "native_extension": libraries / _IDENTITY["native"][0][0],
        "native_library_paths": [
            str(libraries / name) for name, *_ in _IDENTITY["native"]
        ],
        "mlir_swage_locations": [str(libraries.parent)],
        "llvm_linked": "22.1.8",
    }
    fields.update(changes)
    return fresh_offsets._imported_code(**fields)


def _provenance(**changes):
    """Build the provenance block of a run that loaded two kernels."""
    block = {
        "gpu": "NVIDIA RTX A6000",
        "cpu_model": "Test CPU",
        "native_sha256": {"/work/swage/build/lib.so": "a" * 64},
        "loaded_ptx": [
            {"kernel": "segmented_sum", "sha256": "b" * 64, "bytes": 2},
            {"kernel": "segmented_sum", "sha256": "c" * 64, "bytes": 3},
        ],
        "other_compute_process_seen": False,
    }
    block.update(changes)
    return block


def _git_results(status, calls):
    """Return a fake git that logs each command and answers in order."""
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "abc123\n"),
            subprocess.CompletedProcess([], 0, status),
        ]
    )

    def run(command, **options):
        calls.append((command, options))
        return next(results)

    return run


def _assert_git_commands(calls, root):
    """Require the exact revision and status commands, untracked included."""
    options = {
        "cwd": root,
        "check": True,
        "capture_output": True,
        "text": True,
    }
    assert calls == [
        (["git", "rev-parse", "HEAD"], options),
        (["git", "status", "--porcelain"], options),
    ]


class _EmulatedLoopedKernel:
    """Walk each segment in fixed blocks on the CPU, as the kernel does.

    Args:
        extra: Elements admitted past the segment end by the block mask.
            Zero is the real kernel; one is a mask of ``index < end + 1``.
        shift: Offset of the window that is read for every segment that
            leaves room for it. Zero is the real kernel; minus one reads an
            in-bounds window of the right length one element early.
    """

    def __init__(self, extra=0, shift=0):
        self._extra = extra
        self._shift = shift

    def __getitem__(self, grid):
        (programs,) = grid

        def launch(values, offsets, output, *, BLOCK, num_warps):
            data = values.tolist()
            bounds = offsets.tolist()
            for sid in range(programs):
                begin, end = bounds[sid], bounds[sid + 1]
                if 0 <= begin + self._shift and end + self._shift <= len(data):
                    begin, end = begin + self._shift, end + self._shift
                limit = min(end + self._extra, len(data))
                output[sid] = sum(
                    sum(data[start : min(start + BLOCK, limit)])
                    for start in range(begin, end, BLOCK)
                )

        return launch


class _EmulatedTaskKernel:
    """Sum each listed segment on the CPU, as a Triton task kernel does.

    Args:
        one_block: Whether a task reads one block of ``BLOCK`` or ``WARP``
            elements, as the packed and the one-block kernels do, or loops
            over its whole segment.
        extra: Elements admitted past the segment end.
        shift: Offset of the window that is read for every segment that
            leaves room for it.
    """

    def __init__(self, one_block, extra=0, shift=0):
        self._one_block = one_block
        self._extra = extra
        self._shift = shift

    def __getitem__(self, grid):
        def launch(values, offsets, output, ids, *counts, num_warps,
                   BLOCK=None, WARP=None, TASKS=None):
            data = values.tolist()
            bounds = offsets.tolist()
            for sid in ids.tolist():
                begin, end = bounds[sid], bounds[sid + 1]
                if 0 <= begin + self._shift and end + self._shift <= len(data):
                    begin, end = begin + self._shift, end + self._shift
                stop = min(end + self._extra, len(data))
                if self._one_block:
                    stop = min(stop, begin + (BLOCK or WARP))
                output[sid] = sum(data[begin:stop])

        return launch


def _kernels(**changes):
    """Return CPU stand-ins for every Triton kernel the harness launches."""
    kernels = {
        "looped": _EmulatedLoopedKernel(),
        "packed": _EmulatedTaskKernel(one_block=True),
        "cta": _EmulatedTaskKernel(one_block=True),
        "cta_looped": _EmulatedTaskKernel(one_block=False),
    }
    kernels.update(changes)
    return types.SimpleNamespace(**kernels)


def _reference_prepare(torch):
    """Return a stand-in for the planned preparation that sums on the CPU."""

    def prepare(values, offsets, output, *, warp_max_elements):
        def mixed():
            output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))

        return types.SimpleNamespace(mixed=mixed)

    return prepare


def _swage(torch, **changes):
    """Return stand-ins for the two private Swage entry points."""

    def launch(values, offsets, output, kind):
        assert kind == "sum"
        output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))

    entries = {"prepare": _reference_prepare(torch), "launch": launch}
    entries.update(changes)
    return types.SimpleNamespace(**entries)


def test_harnesses_import_without_torch_or_triton():
    """Keep Triton and PyTorch lazy so the harnesses import anywhere."""
    script = (
        "import sys\n"
        "import benchmark_fresh_offsets\n"
        "import benchmark_processes\n"
        "import benchmark_provenance\n"
        "import benchmark_triton_comparison\n"
        "loaded = {'triton', 'torch'} & set(sys.modules)\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT / "benchmarks",
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_missing_triton_drops_only_the_looped_candidates(
    fresh_offsets, monkeypatch
):
    """Run without Triton and say so, instead of failing or substituting."""
    monkeypatch.setitem(sys.modules, "triton", None)

    assert fresh_offsets._optional_triton() is None
    assert fresh_offsets._candidate_names(triton_available=False) == (
        "swage_mixed",
        "swage_cta_call",
        "torch",
        "torch_pad_to_max",
    )
    configuration = fresh_offsets._configuration(
        segment_count=2048, warmups=1, samples=3, triton_available=False
    )
    assert configuration["candidates"] == [
        "swage_mixed",
        "swage_cta_call",
        "torch",
        "torch_pad_to_max",
    ]
    assert configuration["triton_looped"] == (
        "skipped: Triton is not installed"
    )


def test_looped_candidates_cover_the_declared_sweep(fresh_offsets):
    """Time every looped configuration instead of one chosen afterwards."""
    names = fresh_offsets._candidate_names(triton_available=True)

    assert names[:4] == (
        "swage_mixed",
        "swage_cta_call",
        "torch",
        "torch_pad_to_max",
    )
    assert len(names) == 4 + 15 + 4 + 15
    looped = names[4:19]
    assert {name.split("_")[2] for name in looped} == {
        f"b{block}" for block in _LOOPED_BLOCKS
    }
    assert all(name.startswith("triton_looped_b") for name in looped)
    assert names[19:23] == tuple(
        f"triton_planned_w{warps}" for warps in (1, 2, 4, 8)
    )
    # The looping matched scheduler sweeps the blocks and warps of looped.
    assert names[23:] == tuple(
        name.replace("triton_looped", "triton_planned_looped")
        for name in looped
    )


def test_looped_sweep_has_no_longest_segment_floor(triton_comparison):
    """Sweep small blocks whatever the longest segment is."""
    configs = triton_comparison._triton_looped_configs()

    assert configs == [
        *((128, warps) for warps in (1, 2, 4)),
        *((256, warps) for warps in (1, 2, 4, 8)),
        *((512, warps) for warps in (1, 2, 4, 8)),
        *((1024, warps) for warps in (1, 2, 4, 8)),
    ]


def test_fixed_sweep_still_covers_the_longest_segment(triton_comparison):
    """Keep the fixed-shape baseline exactly as the frozen record ran it."""
    assert triton_comparison._triton_sum_configs(4096) == [
        (4096, 1),
        (4096, 2),
        (4096, 4),
        (4096, 8),
    ]
    assert len(triton_comparison._triton_sum_configs(32)) == 26
    assert triton_comparison._triton_sum_configs(4097) == []


def test_layout_pool_yields_distinct_layouts(fresh_offsets):
    """Give every iteration an offsets layout no other iteration sees."""
    from distributions import generate_lengths

    pool = fresh_offsets._layout_pool("power-law", 2048, 6, 7)

    assert [layout.seed for layout in pool] == [7, 8, 9, 10, 11, 12]
    assert len({tuple(layout.offsets) for layout in pool}) == 6
    for layout in pool:
        # Four bytes per segment, so a pool of 10^6-segment layouts fits.
        assert layout.lengths.itemsize == layout.offsets.itemsize == 4
        assert list(layout.lengths) == generate_lengths(
            "power-law", 2048, layout.seed
        )
        assert len(layout.offsets) == 2049
        assert layout.offsets[0] == 0
        assert layout.offsets[-1] == sum(layout.lengths)
        assert all(
            end - begin == length
            for begin, end, length in zip(
                layout.offsets, layout.offsets[1:], layout.lengths
            )
        )
        assert max(layout.lengths) > 4096


def test_layout_pool_rejects_a_repeated_layout(fresh_offsets):
    """Fail loudly when two iterations would share one layout."""
    with pytest.raises(ValueError, match="distinct"):
        fresh_offsets._layout_pool("one-outlier", 1, 2, 7)


def test_candidate_order_is_a_seeded_permutation(fresh_offsets):
    """Derive each iteration's order from the seed, row, and iteration."""
    names = fresh_offsets._candidate_names(triton_available=True)

    seed, order = fresh_offsets._candidate_order(names, 7, "bimodal", 3)

    assert seed == "7:bimodal:3"
    assert sorted(order) == sorted(names)
    assert (seed, order) == fresh_offsets._candidate_order(
        names, 7, "bimodal", 3
    )
    assert order != fresh_offsets._candidate_order(names, 7, "bimodal", 4)[1]
    assert order != fresh_offsets._candidate_order(names, 7, "uniform", 3)[1]
    assert order != fresh_offsets._candidate_order(names, 8, "bimodal", 3)[1]


def test_no_candidate_keeps_a_fixed_neighbour(fresh_offsets):
    """Spread who runs right after the slow candidate across iterations."""
    names = fresh_offsets._select(
        fresh_offsets._candidate_names(triton_available=True),
        ["swage_mixed", "torch", "torch_pad_to_max", "triton_looped"],
        (),
    )
    assert len(names) == 18
    followers = collections.Counter()
    positions = collections.Counter()
    for iteration in range(105):
        _, order = fresh_offsets._candidate_order(
            names, 7, "uniform", iteration
        )
        position = order.index("swage_mixed")
        positions[position] += 1
        if position + 1 < len(order):
            followers[order[position + 1]] += 1

    assert set(followers) == set(names) - {"swage_mixed"}
    assert max(followers.values()) <= 15
    assert len(positions) == len(names)


def test_measure_follows_the_given_order_on_fresh_layouts(fresh_offsets):
    """Run every candidate once per layout and keep every sample."""
    calls = []
    checked = []
    ticks = iter(range(0, 1_000_000, 2_000))
    candidates = {
        name: (lambda layout, name=name: calls.append((name, layout)) or name)
        for name in ("a", "b", "c")
    }
    orders = [["c", "a", "b"], ["a", "b", "c"], ["b", "a", "c"]]

    iterations = fresh_offsets._measure(
        ["l0", "l1", "l2"],
        orders,
        candidates,
        lambda name, result, layout: checked.append((name, result, layout)),
        warmups=1,
        synchronize=lambda: None,
        clock=lambda: next(ticks),
    )

    assert calls == [
        (name, layout)
        for layout, order in zip(["l0", "l1", "l2"], orders)
        for name in order
    ]
    assert checked == [(name, name, layout) for name, layout in calls]
    assert iterations == [
        {
            "timed": timed,
            "candidate_order": order,
            "samples_us": {"a": 2.0, "b": 2.0, "c": 2.0},
        }
        for timed, order in zip([False, True, True], orders)
    ]
    assert [list(iteration["samples_us"]) for iteration in iterations] == (
        orders
    )


def test_measure_requires_one_order_per_layout(fresh_offsets):
    """Refuse a pool and an order list that do not line up."""
    with pytest.raises(ValueError):
        fresh_offsets._measure(
            ["l0", "l1"],
            [["a"]],
            {"a": lambda layout: None},
            lambda name, result, layout: None,
            warmups=0,
            synchronize=lambda: None,
        )


def test_timed_samples_are_grouped_by_candidate(fresh_offsets):
    """Keep the per-candidate series the summaries are computed from."""
    iterations = [
        {"timed": False, "samples_us": {"a": 9.0, "b": 8.0}},
        {"timed": True, "samples_us": {"b": 2.0, "a": 1.0}},
        {"timed": True, "samples_us": {"a": 3.0, "b": 4.0}},
    ]

    assert fresh_offsets._timed_samples(iterations, ("a", "b")) == {
        "a": [1.0, 3.0],
        "b": [2.0, 4.0],
    }


def test_measure_times_the_candidate_and_its_device_work_only(fresh_offsets):
    """Pin the timed region: synchronize, start, run, synchronize, stop."""
    log = []
    now = [0]

    def clock():
        log.append("clock")
        return now[0]

    def synchronize():
        log.append("synchronize")
        now[0] += 3_000

    def candidate(layout):
        log.append("candidate")
        now[0] += 5_000
        return f"result of {layout}"

    def check(name, result, layout):
        log.append("check")
        now[0] += 100_000
        assert (name, result) == ("only", f"result of {layout}")

    iterations = fresh_offsets._measure(
        ["l0", "l1", "l2"],
        [["only"]] * 3,
        {"only": candidate},
        check,
        warmups=1,
        synchronize=synchronize,
        clock=clock,
    )

    assert log == [
        "synchronize",
        "clock",
        "candidate",
        "synchronize",
        "clock",
        "check",
    ] * 3
    # The candidate and the device work it left behind: 5 us plus 3 us. The
    # wait before the timer and the correctness check are outside.
    assert [iteration["samples_us"] for iteration in iterations] == [
        {"only": 8.0}
    ] * 3


def test_warm_step_runs_before_the_timer_and_outside_it(fresh_offsets):
    """Warm, synchronize, start, run, synchronize, stop: per candidate."""
    log = []
    now = [0]

    def clock():
        log.append("clock")
        return now[0]

    def synchronize():
        log.append("synchronize")
        now[0] += 3_000

    def candidate(name):
        def run(layout):
            log.append(f"{name} on {layout}")
            now[0] += 5_000
            return name

        return run

    def warm(name):
        log.append(f"warm {name}")
        now[0] += 700_000

    iterations = fresh_offsets._measure(
        ["l0", "l1"],
        [["a", "b"], ["b", "a"]],
        {"a": candidate("a"), "b": candidate("b")},
        lambda name, result, layout: log.append(f"check {name}"),
        warmups=1,
        synchronize=synchronize,
        clock=clock,
        warm=warm,
    )

    assert log == [
        step
        for layout, order in (("l0", "ab"), ("l1", "ba"))
        for name in order
        for step in (
            f"warm {name}",
            "synchronize",
            "clock",
            f"{name} on {layout}",
            "synchronize",
            "clock",
            f"check {name}",
        )
    ]
    # Still the candidate and the device work it left behind, 5 us plus
    # 3 us, whatever the warm step cost.
    assert [iteration["samples_us"] for iteration in iterations] == [
        {"a": 8.0, "b": 8.0},
        {"b": 8.0, "a": 8.0},
    ]


def test_exact_values_make_any_extra_or_missing_element_visible(
    triton_comparison,
):
    """Use nonzero quarter multiples so every boundary error changes a sum."""
    torch = pytest.importorskip("torch")

    values = triton_comparison._exact_values(torch, 100_000)

    assert values.dtype == torch.float32
    assert torch.equal(values, triton_comparison._exact_values(torch, 100_000))
    assert torch.equal(values * 4, (values * 4).round())
    assert values.min() == 0.25
    assert values.max() == 1.75
    assert len(values.unique()) == 7


def test_looped_check_rejects_what_all_ones_values_accept(triton_comparison):
    """Check the looped baseline on values that expose a shifted window."""
    torch = pytest.importorskip("torch")
    lengths = [3, 0, 130, 7, 300, 1, 1024, 5]
    offsets = torch.tensor(
        [0, *itertools.accumulate(lengths)], dtype=torch.int32
    )
    configs = triton_comparison._triton_looped_configs()
    shifted = _EmulatedLoopedKernel(shift=-1)

    ones = torch.ones(sum(lengths))
    output = torch.empty(len(lengths))
    triton_comparison._launch_triton_looped(
        shifted, ones, offsets, output, len(lengths), 128, 4
    )
    assert output.tolist() == lengths

    triton_comparison._check_triton_looped(
        torch, _EmulatedLoopedKernel(), configs, offsets, len(lengths)
    )
    for broken in (shifted, _EmulatedLoopedKernel(extra=1)):
        with pytest.raises(AssertionError, match="triton_looped_b128_w1"):
            triton_comparison._check_triton_looped(
                torch, broken, configs, offsets, len(lengths)
            )


_DISTRIBUTIONS = (
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "many-tiny",
    "few-huge",
    "one-outlier",
    "alternating-empty",
    "power-law",
)


@pytest.mark.parametrize("name", _DISTRIBUTIONS)
def test_run_distribution_checks_and_records_every_candidate(
    fresh_offsets, name
):
    """Run the whole measurement loop on the CPU with stand-in candidates."""
    torch = pytest.importorskip("torch")

    row = fresh_offsets._run_distribution(
        torch,
        _swage(torch),
        _kernels(),
        name,
        64,
        1,
        2,
        device="cpu",
        synchronize=lambda: None,
        free_bytes=lambda: 1 << 30,
        clock_tick_us=0.04,
    )

    candidates = fresh_offsets._candidate_names(triton_available=True)
    assert row["distribution"] == name
    assert row["seed"] == 7
    assert row["values"] == "quarters"
    assert row["skipped"] == {}
    assert row["excluded"] == []
    assert row["candidates"] == list(candidates)
    assert row["warm_layout_seed"] == 7 + 3
    planned = [name for name in candidates if "planned" in name]
    assert len(planned) == 4 + 15
    assert set(row["triton_planned_partition_samples_us"]) == set(planned)
    assert all(
        len(series) == 2
        for series in row["triton_planned_partition_samples_us"].values()
    )
    assert row["check"] == {
        "exact_segments": 3 * 64,
        "bounded_segments": 0,
        "unchecked_segments": 0,
    }
    assert row["correctness_passed"] is True
    assert set(row["effective_gb_per_s"]) == set(candidates)
    assert set(row["tick_fraction_of_sample"]) == set(candidates)
    for candidate in candidates:
        median = row["summary_us"][candidate]["median"]
        assert row["tick_fraction_of_sample"][candidate] == pytest.approx(
            0.04 / median
        )
        rates = sorted(
            entry["useful_bytes"] / (entry["samples_us"][candidate] * 1_000)
            for entry in row["iterations"]
            if entry["timed"]
        )
        assert row["effective_gb_per_s"][candidate]["median"] == (
            pytest.approx(sum(rates) / 2)
        )
    assert set(row["raw_samples_us"]) == set(candidates)
    assert all(len(row["raw_samples_us"][c]) == 2 for c in candidates)
    assert set(row["summary_us"]) == set(candidates)
    assert len(row["swage_mixed_prepare_samples_us"]) == 2
    assert [entry["timed"] for entry in row["iterations"]] == [
        False,
        True,
        True,
    ]
    for index, entry in enumerate(row["iterations"]):
        assert entry["layout_seed"] == 7 + index
        assert entry["layout_statistics"]["count"] == 64
        assert entry["useful_bytes"] == 4 * (
            entry["layout_statistics"]["total"] + 65 + 64
        )
        assert (
            entry["order_seed"],
            entry["candidate_order"],
        ) == fresh_offsets._candidate_order(candidates, 7, name, index)
        assert list(entry["samples_us"]) == entry["candidate_order"]
    assert row["raw_samples_us"] == fresh_offsets._timed_samples(
        row["iterations"], candidates
    )


@pytest.mark.parametrize("name", _DISTRIBUTIONS)
def test_reading_one_element_past_a_segment_end_is_rejected(
    fresh_offsets, name
):
    """Catch a looped block mask of ``index < end + 1`` on every input."""
    torch = pytest.importorskip("torch")

    with pytest.raises(AssertionError, match=f"triton_looped.* on {name}"):
        fresh_offsets._run_distribution(
            torch,
            _swage(torch),
            _kernels(looped=_EmulatedLoopedKernel(extra=1)),
            name,
            64,
            1,
            2,
            device="cpu",
            synchronize=lambda: None,
            free_bytes=lambda: 1 << 30,
            clock_tick_us=0.04,
        )


def _run(fresh_offsets, torch, name, **changes):
    """Run one row on the CPU with stand-in candidates."""
    options = {
        "device": "cpu",
        "synchronize": lambda: None,
        "free_bytes": lambda: 1 << 30,
        "clock_tick_us": 0.04,
    }
    options.update(changes)
    kernels = options.pop("kernels", _kernels())
    swage = options.pop("swage", None) or _swage(
        torch, prepare=options.pop("prepare", _reference_prepare(torch))
    )
    segment_count = options.pop("segment_count", 64)
    return fresh_offsets._run_distribution(
        torch, swage, kernels, name, segment_count, 1, 2, **options
    )


def test_pad_to_max_is_skipped_with_its_bytes_when_it_does_not_fit(
    fresh_offsets,
):
    """Record what padding would need instead of timing or hiding it."""
    torch = pytest.importorskip("torch")

    row = _run(fresh_offsets, torch, "power-law", free_bytes=lambda: 1000)

    # Three timed layouts and the warm layout, which is padded as well.
    longest = max(
        max(layout.lengths)
        for layout in fresh_offsets._layout_pool("power-law", 64, 4, 7)
    )
    candidates = fresh_offsets._select(
        fresh_offsets._candidate_names(triton_available=True),
        None,
        ["torch_pad_to_max"],
    )
    assert "torch_pad_to_max" not in candidates
    assert set(row["raw_samples_us"]) == set(candidates)
    assert all(
        sorted(entry["candidate_order"]) == sorted(candidates)
        for entry in row["iterations"]
    )
    assert row["pad_to_max"] == {
        "longest_segment": longest,
        "padded_bytes": 64 * longest * 17,
        "free_bytes": 1000,
        "timed": False,
    }
    assert set(row["skipped"]) == {"torch_pad_to_max"}
    assert str(64 * longest * 17) in row["skipped"]["torch_pad_to_max"]


def test_pad_to_max_is_timed_when_it_fits(fresh_offsets):
    """Time padding and the masked sum from offsets in to result out."""
    torch = pytest.importorskip("torch")

    row = _run(fresh_offsets, torch, "bimodal")

    assert row["pad_to_max"]["timed"] is True
    assert row["pad_to_max"]["padded_bytes"] == (
        64 * row["pad_to_max"]["longest_segment"] * 17
    )
    assert len(row["raw_samples_us"]["torch_pad_to_max"]) == 2


def test_rows_follow_the_seed_and_the_value_kind(fresh_offsets):
    """Draw other layouts for another seed and bound random values."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets, torch, "bimodal", seed=100, values_kind="normal"
    )

    assert row["seed"] == 100
    assert row["values"] == "normal"
    assert [entry["layout_seed"] for entry in row["iterations"]] == [
        100,
        101,
        102,
    ]
    assert row["iterations"][0]["order_seed"] == "100:bimodal:0"
    assert row["check"]["bounded_segments"] > 0
    assert row["check"]["unchecked_segments"] == 0


def test_random_values_still_reject_a_wrong_candidate(fresh_offsets):
    """Keep the check meaningful when no exact comparison is possible."""
    torch = pytest.importorskip("torch")

    def prepare(values, offsets, output, *, warp_max_elements):
        def mixed():
            output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))
            output[0] += 1.0

        return types.SimpleNamespace(mixed=mixed)

    with pytest.raises(AssertionError, match="swage_mixed on many-tiny"):
        _run(
            fresh_offsets,
            torch,
            "many-tiny",
            values_kind="normal",
            prepare=prepare,
        )


def test_headline_names_the_fastest_looped_configuration(fresh_offsets):
    """Print a short summary without dropping anything from the record."""
    summary = {
        "swage_mixed": {"median": 20_000.04},
        "torch": {"median": 55.26},
        "triton_looped_b128_w1": {"median": 54.1},
        "triton_looped_b512_w4": {"median": 18.3},
    }

    assert fresh_offsets._headline(summary) == {
        "swage_mixed": 20_000.0,
        "torch": 55.3,
        "fastest triton_looped_b512_w4": 18.3,
    }
    del summary["triton_looped_b128_w1"], summary["triton_looped_b512_w4"]
    summary["torch_pad_to_max"] = {"median": 900.04}
    assert fresh_offsets._headline(summary) == {
        "swage_mixed": 20_000.0,
        "torch": 55.3,
        "torch_pad_to_max": 900.0,
    }
    # A filtered run prints what it timed, one line per Triton family.
    filtered = {
        "swage_cta_call": {"median": 61.04},
        "triton_planned_w1": {"median": 44.0},
        "triton_planned_w4": {"median": 41.0},
        "triton_planned_looped_b128_w1": {"median": 30.0},
        "triton_planned_looped_b256_w2": {"median": 33.0},
    }
    assert fresh_offsets._headline(filtered) == {
        "swage_cta_call": 61.0,
        "fastest triton_planned_w4": 41.0,
        "fastest triton_planned_looped_b128_w1": 30.0,
    }


def test_measure_stops_at_the_first_wrong_result(fresh_offsets):
    """Never record a timing sample for an incorrect result."""

    def check(name, result, layout):
        raise AssertionError(f"{name} is wrong on {layout}")

    with pytest.raises(AssertionError, match="torch is wrong on l0"):
        fresh_offsets._measure(
            ["l0", "l1"],
            [["torch"], ["torch"]],
            {"torch": lambda layout: None},
            check,
            warmups=0,
            synchronize=lambda: None,
        )


def test_git_metadata_refuses_a_dirty_worktree(fresh_offsets, monkeypatch):
    """Do not label measurements from modified or untracked sources."""
    calls = []
    monkeypatch.setattr(
        fresh_offsets.subprocess,
        "run",
        _git_results("?? scratch.txt\n", calls),
    )

    with pytest.raises(RuntimeError, match="clean source worktree"):
        fresh_offsets._git_metadata(pathlib.Path("."), allow_dirty=False)
    _assert_git_commands(calls, pathlib.Path("."))


def test_git_metadata_records_a_clean_revision(fresh_offsets, monkeypatch):
    """Carry the exact revision and the dirty flag in every record."""
    calls = []
    monkeypatch.setattr(
        fresh_offsets.subprocess, "run", _git_results("", calls)
    )

    assert fresh_offsets._git_metadata(
        pathlib.Path("checkout"), allow_dirty=False
    ) == {"revision": "abc123", "worktree_clean": True, "dirty": []}
    _assert_git_commands(calls, pathlib.Path("checkout"))


def test_smoke_record_states_that_the_worktree_was_dirty(
    fresh_offsets, monkeypatch
):
    """Allow a smoke run on a dirty tree only with the flag recorded."""
    calls = []
    monkeypatch.setattr(
        fresh_offsets.subprocess,
        "run",
        _git_results("?? scratch.txt\n", calls),
    )

    assert fresh_offsets._git_metadata(
        pathlib.Path("."), allow_dirty=True
    ) == {
        "revision": "abc123",
        "worktree_clean": False,
        "dirty": ["?? scratch.txt"],
    }
    _assert_git_commands(calls, pathlib.Path("."))


def test_smoke_output_stays_out_of_the_evidence_directory(
    fresh_offsets, tmp_path
):
    """Keep smoke records away from committed benchmark evidence."""
    evidence = tmp_path / "benchmarks" / "results" / "smoke.json"

    with pytest.raises(ValueError, match="benchmarks/results"):
        fresh_offsets._check_output(tmp_path, evidence, smoke=True)
    fresh_offsets._check_output(tmp_path, evidence, smoke=False)
    fresh_offsets._check_output(tmp_path, tmp_path / "smoke.json", smoke=True)


def test_smoke_selects_tiny_sizes(fresh_offsets):
    """Run a few small layouts for smoke and the full sizes otherwise."""
    smoke = fresh_offsets._arguments(["--output", "x.json", "--smoke"])
    full = fresh_offsets._arguments(["--output", "x.json"])

    assert fresh_offsets._sizes(smoke) == (2048, 1, 3)
    assert fresh_offsets._sizes(full) == (32_768, 5, 100)


def test_arguments_select_scale_seed_values_and_distributions(fresh_offsets):
    """Offer 10^6 segments, random values, a seed, and chosen rows."""
    default = fresh_offsets._arguments(["--output", "x.json"])
    chosen = fresh_offsets._arguments(
        [
            "--output",
            "x.json",
            "--distributions",
            "power-law",
            "bimodal",
            "--segment-count",
            "1000000",
            "--seed",
            "1000",
            "--values",
            "normal",
            "--samples",
            "20",
            "--warmups",
            "2",
        ]
    )

    assert default.distributions == list(_DISTRIBUTIONS)
    assert (default.seed, default.values) == (7, "quarters")
    assert chosen.distributions == ["power-law", "bimodal"]
    assert (chosen.seed, chosen.values) == (1000, "normal")
    assert fresh_offsets._sizes(chosen) == (1_000_000, 2, 20)
    smoke = fresh_offsets._arguments(
        ["--output", "x.json", "--smoke", "--segment-count", "4096"]
    )
    assert fresh_offsets._sizes(smoke) == (4096, 1, 3)


@pytest.mark.parametrize(
    "extra",
    [
        ["--distributions", "normal"],
        ["--distributions", "uniform", "--segment-count", "1000000"],
        ["--segment-count", "0"],
        ["--smoke", "--segment-count", "0"],
        ["--values", "ones"],
    ],
)
def test_arguments_reject_what_cannot_be_measured(fresh_offsets, extra):
    """Refuse a configuration before any device work starts."""
    with pytest.raises(SystemExit):
        fresh_offsets._arguments(["--output", "x.json", *extra])


@pytest.mark.parametrize(
    "extra", [["--samples", "1"], ["--warmups", "0"], ["--samples", "x"]]
)
def test_rejects_sizes_that_cannot_be_summarized(fresh_offsets, extra):
    """Require a warmup for lazy compilation and two samples for quartiles."""
    with pytest.raises(SystemExit):
        fresh_offsets._sizes(
            fresh_offsets._arguments(["--output", "x.json", *extra])
        )


def test_environment_reads_the_device_identity(fresh_offsets):
    """Record the GPU model, capability, driver, and library versions."""
    assert _environment(fresh_offsets) == {
        "platform": fresh_offsets.platform.platform(),
        "python": sys.version,
        "pytorch": "2.12.0+cu130",
        "pytorch_cuda": "13.0",
        "triton": "3.7.0",
        "cuda_driver": "13.0",
        "nvidia_driver": "580.178.04",
        "gpu": "NVIDIA RTX A6000",
        "compute_capability": "sm_86",
        "multiprocessors": 84,
        "total_memory_bytes": 51_527_024_640,
    }


def test_record_carries_provenance_and_the_smoke_label(fresh_offsets):
    """Write the fields a reader needs to place and trust a measurement."""
    source = {"revision": "abc123", "worktree_clean": False, "dirty": ["?? x"]}
    configuration = fresh_offsets._configuration(
        segment_count=2048, warmups=1, samples=3, triton_available=True
    )

    record = fresh_offsets._record(
        source=source,
        environment=_environment(fresh_offsets),
        configuration=configuration,
        results=[],
        imported_code=_imported_code(fresh_offsets),
        provenance=_provenance(),
        smoke=True,
    )

    assert record["benchmark"] == "fresh-offsets-segmented-sum"
    assert record["smoke"] is True
    assert record["status"] == "smoke run; not evidence"
    assert record["source"] == source
    assert record["environment"]["gpu"] == "NVIDIA RTX A6000"
    assert record["environment"]["compute_capability"] == "sm_86"
    assert record["environment"]["cuda_driver"] == "13.0"
    assert record["environment"]["pytorch"] == "2.12.0+cu130"
    assert record["environment"]["triton"] == "3.7.0"
    assert record["configuration"]["layouts_per_distribution"] == 4
    assert record["configuration"]["distributions"][-1] == "power-law"
    assert "classification" in record["configuration"]["timed_region"][
        "swage_mixed"
    ]
    assert record["results"] == []
    assert record["recorded_at"]


def test_full_record_is_not_labelled_smoke(fresh_offsets):
    """Distinguish a full run from a smoke run in the record itself."""
    record = fresh_offsets._record(
        source={"revision": "abc123", "worktree_clean": True, "dirty": []},
        environment=_environment(fresh_offsets, triton_version=None),
        configuration={},
        results=[],
        imported_code=_imported_code(fresh_offsets),
        provenance=_provenance(),
        smoke=False,
    )

    assert record["smoke"] is False
    assert record["status"] == "research run; no performance gate"
    assert record["environment"]["triton"] is None


@pytest.mark.parametrize(
    "field", ["gpu", "compute_capability", "cuda_driver", "pytorch"]
)
def test_record_rejects_missing_environment_provenance(fresh_offsets, field):
    """Refuse to write a record that cannot be attributed to a machine."""
    environment = _environment(fresh_offsets)
    environment[field] = None

    with pytest.raises(ValueError, match=field):
        fresh_offsets._record(
            source={"revision": "abc123", "worktree_clean": True, "dirty": []},
            environment=environment,
            configuration={},
            results=[],
            imported_code=_imported_code(fresh_offsets),
            provenance=_provenance(),
            smoke=False,
        )


def test_package_from_this_checkout_is_accepted(fresh_offsets, tmp_path):
    """Measure the swage package that sits beside the benchmark script."""
    package = tmp_path / "python" / "swage"
    package.mkdir(parents=True)
    linked = tmp_path.parent / f"{tmp_path.name}-link"
    linked.symlink_to(tmp_path)

    assert fresh_offsets._same_checkout(tmp_path, package, smoke=False)
    assert fresh_offsets._same_checkout(
        tmp_path, linked / "python" / "swage", smoke=False
    )


def test_full_run_refuses_a_package_from_another_checkout(
    fresh_offsets, tmp_path
):
    """Do not attribute another checkout's code to this revision."""
    root = tmp_path / "swage-review"
    package = tmp_path / "swage" / "python" / "swage"
    root.mkdir()
    package.mkdir(parents=True)

    with pytest.raises(RuntimeError) as refusal:
        fresh_offsets._same_checkout(root, package, smoke=False)
    assert str(package) in str(refusal.value)
    assert str(root) in str(refusal.value)
    # A sibling whose name only extends the checkout name is still outside.
    assert not fresh_offsets._same_checkout(root, package, smoke=True)
    assert not fresh_offsets._same_checkout(
        tmp_path / "swage", tmp_path / "swage-review", smoke=True
    )


def test_native_library_paths_follow_the_runtime_identity(
    fresh_offsets, tmp_path
):
    """Resolve exactly the files the runtime's native identity names."""
    build = tmp_path / "build" / "_mlir_libs"
    other = tmp_path / "source" / "_mlir_libs"
    build.mkdir(parents=True)
    other.mkdir(parents=True)
    for name, *_ in _IDENTITY["native"]:
        (build / name).write_bytes(b"")
    (build / "unrelated.so").write_bytes(b"")
    (build / "libSwagePythonCAPI.so").symlink_to("libSwagePythonCAPI.so.22.1")
    native = [*_IDENTITY["native"], ["libSwagePythonCAPI.so", 30, 40]]
    linked = tmp_path / "linked-build"
    linked.symlink_to(tmp_path / "build")

    # One path per identity entry, with the directory resolved so a build
    # reached through a symlink is reported where it really lives.
    assert fresh_offsets._native_library_paths(
        native, [str(other), str(linked / "_mlir_libs")]
    ) == sorted(str(build / name) for name, *_ in native)
    assert fresh_offsets._native_library_paths(None, [str(build)]) == []


def test_imported_code_identifies_the_package_and_the_native_build(
    fresh_offsets,
):
    """Record which package and which native build produced the numbers."""
    libraries = "/work/swage/build/python_packages/mlir_swage/_mlir_libs"

    assert _imported_code(fresh_offsets) == {
        "benchmark_checkout": "/work/swage",
        "swage_package": "/work/swage/python/swage",
        "swage_package_in_checkout": True,
        "compiler_identity": _IDENTITY,
        "native_extension": f"{libraries}/{_IDENTITY['native'][0][0]}",
        "native_library_paths": [
            f"{libraries}/{name}" for name, *_ in _IDENTITY["native"]
        ],
        "native_libraries_in_checkout": True,
        "mlir_swage_locations": [
            "/work/swage/build/python_packages/mlir_swage"
        ],
        "llvm_linked": "22.1.8",
    }
    outside = _imported_code(
        fresh_offsets,
        native_library_paths=["/work/other/build/libSwagePythonCAPI.so"],
    )
    assert outside["native_libraries_in_checkout"] is False


def test_imported_code_reads_the_runtime_compiler_identity(fresh_offsets):
    """Store the identity the runtime itself uses for its cache key."""
    from swage import _runtime

    identity = _runtime._compiler_identity()
    block = _imported_code(fresh_offsets, identity=identity)

    assert set(identity) >= {"frontend", "native"}
    assert block["compiler_identity"] == identity
    assert block["compiler_identity"]["frontend"] == identity["frontend"]


def test_record_carries_the_imported_code(fresh_offsets):
    """Keep the package path, compiler identity, and LLVM in every record."""
    record = fresh_offsets._record(
        source={"revision": "abc123", "worktree_clean": True, "dirty": []},
        environment=_environment(fresh_offsets),
        configuration={},
        results=[],
        imported_code=_imported_code(fresh_offsets),
        provenance=_provenance(),
        smoke=False,
    )

    assert record["imported_code"] == _imported_code(fresh_offsets)
    assert record["imported_code"]["compiler_identity"]["frontend"] == "f" * 64
    assert record["imported_code"]["llvm_linked"] == "22.1.8"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"same_checkout": False}, "/work/swage/python/swage"),
        ({"identity": {**_IDENTITY, "frontend": None}}, "frontend"),
        ({"identity": {**_IDENTITY, "native": None}}, "native"),
        ({"native_library_paths": []}, "native_library_paths"),
        ({"llvm_linked": None}, "llvm_linked"),
    ],
)
def test_full_record_requires_identified_code(
    fresh_offsets, changes, message
):
    """Refuse a full record whose measured code cannot be identified."""
    imported_code = _imported_code(fresh_offsets, **changes)
    arguments = {
        "source": {"revision": "abc123", "worktree_clean": True, "dirty": []},
        "environment": _environment(fresh_offsets),
        "configuration": {},
        "results": [],
        "imported_code": imported_code,
        "provenance": _provenance(),
    }

    with pytest.raises(ValueError, match=message):
        fresh_offsets._record(**arguments, smoke=False)
    record = fresh_offsets._record(**arguments, smoke=True)
    assert record["imported_code"] == imported_code


def test_record_rejects_a_dirty_tree_outside_smoke(fresh_offsets):
    """Keep the dirty-tree guard at the point where the record is built."""
    with pytest.raises(ValueError, match="worktree"):
        fresh_offsets._record(
            source={"revision": "abc123", "worktree_clean": False},
            environment=_environment(fresh_offsets),
            configuration={},
            results=[],
            imported_code=_imported_code(fresh_offsets),
            provenance=_provenance(),
            smoke=False,
        )


def test_record_carries_the_provenance_block(fresh_offsets):
    """Keep the binary hashes and the machine state beside the revision."""
    record = fresh_offsets._record(
        source={"revision": "abc123", "worktree_clean": True, "dirty": []},
        environment=_environment(fresh_offsets),
        configuration={},
        results=[],
        imported_code=_imported_code(fresh_offsets),
        provenance=_provenance(),
        smoke=False,
    )

    assert record["provenance"] == _provenance()


@pytest.mark.parametrize("field", ["native_sha256", "loaded_ptx"])
def test_full_record_requires_the_binary_hashes(fresh_offsets, field):
    """Refuse a full record that cannot name the binary it measured."""
    arguments = {
        "source": {"revision": "abc123", "worktree_clean": True, "dirty": []},
        "environment": _environment(fresh_offsets),
        "configuration": {},
        "results": [],
        "imported_code": _imported_code(fresh_offsets),
        "provenance": _provenance(**{field: type(_provenance()[field])()}),
    }

    with pytest.raises(ValueError, match=field):
        fresh_offsets._record(**arguments, smoke=False)
    assert fresh_offsets._record(**arguments, smoke=True)["smoke"] is True


def test_configuration_states_the_methods(fresh_offsets):
    """Say what is timed, how it is checked, and what the rate counts."""
    configuration = fresh_offsets._configuration(
        segment_count=4096,
        warmups=2,
        samples=20,
        triton_available=True,
        distributions=["power-law"],
        seed=1000,
        values="normal",
        clock_tick_us=0.04,
    )

    assert configuration["distributions"] == ["power-law"]
    assert configuration["seed"] == 1000
    assert configuration["values_kind"] == "normal"
    assert configuration["timer"]["tick_us"] == 0.04
    assert "padding" in configuration["timed_region"]["torch_pad_to_max"]
    assert "offsets" in configuration["effective_gb_per_s"]
    assert "float64" in configuration["correctness"]


def _recording_swage(torch, calls):
    """Return Swage stand-ins that log every call with its offsets total."""

    def prepare(values, offsets, output, *, warp_max_elements):
        calls.append(("prepare", int(offsets[-1]), output))

        def mixed():
            output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))

        return types.SimpleNamespace(mixed=mixed)

    def launch(values, offsets, output, kind):
        calls.append(("launch", int(offsets[-1]), output))
        output.copy_(torch.segment_reduce(values, "sum", offsets=offsets))

    return types.SimpleNamespace(prepare=prepare, launch=launch)


def test_warm_calls_repeat_the_candidate_on_one_layout_that_is_never_timed(
    fresh_offsets,
):
    """Precede every sample by calls of the same candidate elsewhere."""
    torch = pytest.importorskip("torch")
    calls = []

    row = _run(
        fresh_offsets,
        torch,
        "bimodal",
        swage=_recording_swage(torch, calls),
        only=["swage_mixed", "swage_cta_call"],
        warm_calls=2,
    )

    timed_totals = [
        entry["layout_statistics"]["total"] for entry in row["iterations"]
    ]
    (warm_total,) = {total for _, total, _ in calls} - set(timed_totals)
    assert row["warm_layout_seed"] == 7 + 3
    assert warm_total == sum(
        fresh_offsets._layout_pool("bimodal", 64, 4, 7)[-1].lengths
    )
    for kind in ("prepare", "launch"):
        totals = [total for call, total, _ in calls if call == kind]
        # Two warm calls, then the timed call, for each of three layouts.
        assert totals == [
            total
            for timed in timed_totals
            for total in (warm_total, warm_total, timed)
        ]
        outputs = [output for call, _, output in calls if call == kind]
        warm_output, timed_output = outputs[0], outputs[2]
        assert warm_output is not timed_output
        assert all(
            output is (timed_output if index % 3 == 2 else warm_output)
            for index, output in enumerate(outputs)
        )
    # Only the timed preparations are in the preparation series.
    assert len(row["swage_mixed_prepare_samples_us"]) == 2
    assert row["check"]["exact_segments"] == 3 * 64


def test_without_warm_calls_every_call_is_a_timed_call(fresh_offsets):
    """Reproduce the earlier method when the warm step is switched off."""
    torch = pytest.importorskip("torch")
    calls = []

    row = _run(
        fresh_offsets,
        torch,
        "bimodal",
        swage=_recording_swage(torch, calls),
        only=["swage_mixed"],
        warm_calls=0,
    )

    assert row["warm_layout_seed"] is None
    assert [total for _, total, _ in calls] == [
        entry["layout_statistics"]["total"] for entry in row["iterations"]
    ]


def test_a_warm_call_cannot_stand_in_for_an_unwritten_timed_result(
    fresh_offsets,
):
    """Keep the unwritten-output check when a warm call wrote a result."""
    torch = pytest.importorskip("torch")
    seen = []

    def prepare(values, offsets, output, *, warp_max_elements):
        def mixed():
            # Writes on the warm layout only, which is prepared first.
            if not seen or int(offsets[-1]) == seen[0]:
                seen.append(int(offsets[-1]))
                output.copy_(
                    torch.segment_reduce(values, "sum", offsets=offsets)
                )

        return types.SimpleNamespace(mixed=mixed)

    with pytest.raises(AssertionError, match="swage_mixed on bimodal"):
        _run(
            fresh_offsets,
            torch,
            "bimodal",
            prepare=prepare,
            only=["swage_mixed"],
            warm_calls=1,
        )


def test_filtered_row_times_only_the_selected_candidates(fresh_offsets):
    """Leave pad-to-max out by option and record that it was."""
    torch = pytest.importorskip("torch")
    budget_reads = []

    def free_bytes():
        budget_reads.append(1)
        return 1 << 30

    row = _run(
        fresh_offsets,
        torch,
        "bimodal",
        exclude=["torch_pad_to_max", "triton_planned_looped"],
        free_bytes=free_bytes,
    )

    names = fresh_offsets._candidate_names(triton_available=True)
    kept = [
        name
        for name in names
        if name != "torch_pad_to_max"
        and not name.startswith("triton_planned_looped")
    ]
    assert row["candidates"] == kept
    assert set(row["raw_samples_us"]) == set(kept)
    assert row["excluded"] == [name for name in names if name not in kept]
    assert row["skipped"] == {}
    assert row["pad_to_max"]["timed"] is False
    assert all(
        sorted(entry["candidate_order"]) == sorted(kept)
        for entry in row["iterations"]
    )
    assert (
        row["iterations"][0]["order_seed"],
        row["iterations"][0]["candidate_order"],
    ) == fresh_offsets._candidate_order(kept, 7, "bimodal", 0)


def test_named_candidates_run_alone(fresh_offsets):
    """Time a short list, by candidate name or by family."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets,
        torch,
        "many-tiny",
        only=["swage_mixed", "torch", "triton_planned"],
    )

    assert row["candidates"] == [
        "swage_mixed",
        "torch",
        *(f"triton_planned_w{warps}" for warps in (1, 2, 4, 8)),
    ]
    assert "torch_pad_to_max" in row["excluded"]


def test_a_filter_that_leaves_no_candidate_is_an_error(fresh_offsets):
    """Refuse a row on which nothing that was named can run."""
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="no candidate.*power-law"):
        _run(
            fresh_offsets,
            torch,
            "power-law",
            segment_count=2048,
            only=["triton_planned"],
            warm_calls=0,
        )


def test_one_block_planned_triton_is_skipped_by_the_longest_pool_segment(
    fresh_offsets,
):
    """Decide the skip on the whole row, not layout by layout."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets,
        torch,
        "power-law",
        segment_count=2048,
        only=["torch", "triton_planned", "triton_planned_looped_b256_w4"],
        warm_calls=1,
    )

    assert row["candidates"] == ["torch", "triton_planned_looped_b256_w4"]
    assert set(row["skipped"]) == {"triton_planned"}
    assert "4096" in row["skipped"]["triton_planned"]
    assert str(row["pad_to_max"]["longest_segment"]) in (
        row["skipped"]["triton_planned"]
    )
    assert row["pad_to_max"]["longest_segment"] > 4096
    assert list(row["triton_planned_partition_samples_us"]) == [
        "triton_planned_looped_b256_w4"
    ]


@pytest.mark.parametrize(
    ("kernels", "family"),
    [
        (
            {"cta_looped": _EmulatedTaskKernel(one_block=False, extra=1)},
            "triton_planned_looped",
        ),
        (
            {"cta_looped": _EmulatedTaskKernel(one_block=False, shift=-1)},
            "triton_planned_looped",
        ),
        (
            {"cta": _EmulatedTaskKernel(one_block=True, shift=-1)},
            "triton_planned",
        ),
        (
            {"packed": _EmulatedTaskKernel(one_block=True, extra=1)},
            "triton_planned",
        ),
    ],
)
def test_a_wrong_planned_kernel_is_rejected(fresh_offsets, kernels, family):
    """Check the planned candidates as strictly as the looped ones."""
    torch = pytest.importorskip("torch")

    with pytest.raises(AssertionError, match=f"{family}_.* on bimodal"):
        _run(
            fresh_offsets,
            torch,
            "bimodal",
            kernels=_kernels(**kernels),
            only=[family],
            warm_calls=0,
        )


def test_planned_triton_partitions_inside_the_timed_call(
    fresh_offsets, monkeypatch
):
    """Plan every fresh layout again; never reuse a task list."""
    torch = pytest.importorskip("torch")
    partitioned = []
    partition = fresh_offsets._partition_tasks

    def recording(torch, offsets):
        partitioned.append(int(offsets[-1]))
        return partition(torch, offsets)

    monkeypatch.setattr(fresh_offsets, "_partition_tasks", recording)

    row = _run(
        fresh_offsets,
        torch,
        "bimodal",
        only=["triton_planned_w4", "triton_planned_looped_b128_w2"],
        warm_calls=0,
    )

    totals = [
        entry["layout_statistics"]["total"] for entry in row["iterations"]
    ]
    assert sorted(partitioned) == sorted(totals * 2)
    assert set(row["triton_planned_partition_samples_us"]) == {
        "triton_planned_w4",
        "triton_planned_looped_b128_w2",
    }
    for name, series in row["triton_planned_partition_samples_us"].items():
        assert len(series) == 2
        assert all(
            0 < part <= whole
            for part, whole in zip(series, row["raw_samples_us"][name])
        )


def test_arguments_take_a_candidate_filter_and_warm_calls(fresh_offsets):
    """Offer the filter and the warm step; reject what names nothing."""
    default = fresh_offsets._arguments(["--output", "x.json"])
    chosen = fresh_offsets._arguments(
        [
            "--output",
            "x.json",
            "--candidates",
            "swage_mixed",
            "torch",
            "triton_planned_looped",
            "--exclude-candidates",
            "triton_planned_looped_b128_w1",
            "--warm-calls",
            "0",
        ]
    )

    assert default.candidates is None
    assert default.exclude_candidates == []
    assert default.warm_calls == 2
    assert chosen.candidates == [
        "swage_mixed",
        "torch",
        "triton_planned_looped",
    ]
    assert chosen.exclude_candidates == ["triton_planned_looped_b128_w1"]
    assert chosen.warm_calls == 0
    for extra in (
        ["--candidates", "pad_to_max"],
        ["--exclude-candidates", "triton_fixed"],
        ["--warm-calls", "-1"],
    ):
        with pytest.raises(SystemExit):
            fresh_offsets._arguments(["--output", "x.json", *extra])


def test_a_named_triton_candidate_requires_triton(fresh_offsets):
    """Fail instead of dropping a candidate that was asked for by name."""
    fresh_offsets._require_available(["torch", "triton_looped"], True)
    fresh_offsets._require_available(None, False)
    fresh_offsets._require_available(["torch", "swage_mixed"], False)
    with pytest.raises(RuntimeError, match="triton_looped.*not installed"):
        fresh_offsets._require_available(["torch", "triton_looped"], False)


def test_configuration_states_the_filter_and_the_warm_step(fresh_offsets):
    """Say which candidates ran and what preceded every sample."""
    configuration = fresh_offsets._configuration(
        segment_count=2048,
        warmups=1,
        samples=3,
        triton_available=True,
        candidates=["swage_mixed", "swage_cta_call", "triton_planned"],
        exclude_candidates=["triton_planned_w8"],
        warm_calls=2,
    )

    assert configuration["candidates"] == [
        "swage_mixed",
        "swage_cta_call",
        "triton_planned_w1",
        "triton_planned_w2",
        "triton_planned_w4",
    ]
    assert configuration["candidate_filter"]["candidates"] == [
        "swage_mixed",
        "swage_cta_call",
        "triton_planned",
    ]
    assert configuration["candidate_filter"]["exclude_candidates"] == [
        "triton_planned_w8"
    ]
    assert configuration["warm_step"]["calls"] == 2
    assert "same candidate" in configuration["warm_step"]["step"]
    assert "same candidate" in configuration["timer"]["batching"]
    assert "no mixed-only preparation" in configuration["swage_policy"]
    assert "launch_gpu" in configuration["timed_region"]["swage_cta_call"]
    assert "nonzero" in configuration["timed_region"]["triton_planned"]
    assert "nonzero" in configuration["timed_region"]["triton_planned_looped"]
    off = fresh_offsets._configuration(
        segment_count=2048,
        warmups=1,
        samples=3,
        triton_available=True,
        warm_calls=0,
    )
    assert off["warm_step"]["calls"] == 0
    assert off["warm_step"]["step"].startswith("none")


def test_a_family_left_out_by_option_is_not_reported_as_skipped(
    fresh_offsets,
):
    """Keep excluded, by option, apart from skipped, by inability."""
    torch = pytest.importorskip("torch")

    row = _run(
        fresh_offsets,
        torch,
        "power-law",
        segment_count=2048,
        only=["torch", "triton_looped_b256_w4"],
        free_bytes=lambda: 1000,
        warm_calls=0,
    )

    # Neither pad-to-max nor the one-block planned baseline can run on
    # this row, but neither was asked for.
    assert row["skipped"] == {}
    assert row["candidates"] == ["torch", "triton_looped_b256_w4"]
    assert {"torch_pad_to_max", "triton_planned_w1", "swage_mixed"} <= set(
        row["excluded"]
    )
    names = fresh_offsets._candidate_names(triton_available=True)
    assert sorted(row["candidates"] + row["excluded"]) == sorted(names)
