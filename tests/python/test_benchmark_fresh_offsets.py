# tests/python/test_benchmark_fresh_offsets.py
"""Tests for the fresh-offsets benchmark and the looped Triton baseline."""

import importlib
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


def _git_results(status):
    """Return fake git output for one revision and one porcelain status."""
    results = iter(
        [
            subprocess.CompletedProcess([], 0, "abc123\n"),
            subprocess.CompletedProcess([], 0, status),
        ]
    )
    return lambda *args, **kwargs: next(results)


def test_harnesses_import_without_torch_or_triton():
    """Keep Triton and PyTorch lazy so the harnesses import anywhere."""
    script = (
        "import sys\n"
        "import benchmark_fresh_offsets\n"
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
        "torch",
    )
    configuration = fresh_offsets._configuration(
        segment_count=2048, warmups=1, samples=3, triton_available=False
    )
    assert configuration["candidates"] == ["swage_mixed", "torch"]
    assert configuration["triton_looped"] == (
        "skipped: Triton is not installed"
    )


def test_looped_candidates_cover_the_declared_sweep(fresh_offsets):
    """Time every looped configuration instead of one chosen afterwards."""
    names = fresh_offsets._candidate_names(triton_available=True)

    assert names[:2] == ("swage_mixed", "torch")
    assert len(names) == 2 + 15
    assert {name.split("_")[2] for name in names[2:]} == {
        f"b{block}" for block in _LOOPED_BLOCKS
    }
    assert all(name.startswith("triton_looped_b") for name in names[2:])


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
        assert layout.lengths == generate_lengths(
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


def test_measure_times_each_candidate_once_per_fresh_layout(fresh_offsets):
    """Rotate the order, check every result, and keep every timed sample."""
    calls = []
    checked = []
    ticks = iter(range(0, 1_000_000, 2_000))
    candidates = {
        name: (lambda layout, name=name: calls.append((name, layout)) or name)
        for name in ("a", "b", "c")
    }

    samples = fresh_offsets._measure(
        ["l0", "l1", "l2", "l3", "l4"],
        candidates,
        lambda name, result, layout: checked.append((name, result, layout)),
        warmups=2,
        synchronize=lambda: None,
        clock=lambda: next(ticks),
    )

    assert [name for name, _ in calls] == [
        *("a", "b", "c"),
        *("b", "c", "a"),
        *("c", "a", "b"),
        *("a", "b", "c"),
        *("b", "c", "a"),
    ]
    for name in candidates:
        assert [layout for called, layout in calls if called == name] == [
            "l0",
            "l1",
            "l2",
            "l3",
            "l4",
        ]
    assert checked == [(name, name, layout) for name, layout in calls]
    assert samples == {"a": [2.0] * 3, "b": [2.0] * 3, "c": [2.0] * 3}


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
    assert fresh_offsets._headline(summary) == {
        "swage_mixed": 20_000.0,
        "torch": 55.3,
    }


def test_measure_stops_at_the_first_wrong_result(fresh_offsets):
    """Never record a timing sample for an incorrect result."""

    def check(name, result, layout):
        raise AssertionError(f"{name} is wrong on {layout}")

    with pytest.raises(AssertionError, match="torch is wrong on l0"):
        fresh_offsets._measure(
            ["l0", "l1"],
            {"torch": lambda layout: None},
            check,
            warmups=0,
            synchronize=lambda: None,
        )


def test_git_metadata_refuses_a_dirty_worktree(fresh_offsets, monkeypatch):
    """Do not label measurements from modified or untracked sources."""
    monkeypatch.setattr(
        fresh_offsets.subprocess, "run", _git_results("?? scratch.txt\n")
    )

    with pytest.raises(RuntimeError, match="clean source worktree"):
        fresh_offsets._git_metadata(pathlib.Path("."), allow_dirty=False)


def test_git_metadata_records_a_clean_revision(fresh_offsets, monkeypatch):
    """Carry the exact revision and the dirty flag in every record."""
    monkeypatch.setattr(fresh_offsets.subprocess, "run", _git_results(""))

    assert fresh_offsets._git_metadata(
        pathlib.Path("."), allow_dirty=False
    ) == {"revision": "abc123", "worktree_clean": True, "dirty": []}


def test_smoke_record_states_that_the_worktree_was_dirty(
    fresh_offsets, monkeypatch
):
    """Allow a smoke run on a dirty tree only with the flag recorded."""
    monkeypatch.setattr(
        fresh_offsets.subprocess, "run", _git_results("?? scratch.txt\n")
    )

    assert fresh_offsets._git_metadata(
        pathlib.Path("."), allow_dirty=True
    ) == {
        "revision": "abc123",
        "worktree_clean": False,
        "dirty": ["?? scratch.txt"],
    }


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
            smoke=False,
        )


def test_record_rejects_a_dirty_tree_outside_smoke(fresh_offsets):
    """Keep the dirty-tree guard at the point where the record is built."""
    with pytest.raises(ValueError, match="worktree"):
        fresh_offsets._record(
            source={"revision": "abc123", "worktree_clean": False},
            environment=_environment(fresh_offsets),
            configuration={},
            results=[],
            smoke=False,
        )
