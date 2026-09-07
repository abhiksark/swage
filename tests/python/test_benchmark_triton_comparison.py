# tests/python/test_benchmark_triton_comparison.py
"""Tests for comparison timing and repeated-campaign evidence contracts."""

import contextlib
import copy
import importlib
import pathlib

import pytest


@pytest.fixture
def comparison_modules(monkeypatch):
    """Import both standalone benchmark entry points without CUDA."""
    root = pathlib.Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "benchmarks"))
    benchmark = importlib.import_module("benchmark_triton_comparison")
    campaign = importlib.import_module("run_triton_comparison_campaign")
    return benchmark, campaign


def test_segmented_cases_append_real_trace_with_provenance(
    comparison_modules,
):
    """Keep seven synthetic cases unchanged, then append the real trace."""
    benchmark, _ = comparison_modules
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

    cases = list(
        benchmark._segmented_case_inputs(generate_lengths, load_real_trace)
    )
    synthetic_names = [
        "many-tiny",
        "uniform",
        "log-normal",
        "bimodal",
        "zipf-like",
        "few-huge",
        "one-outlier",
    ]

    assert [case["distribution"] for case in cases] == [
        *synthetic_names,
        "soc-epinions1-outdegree-v1",
    ]
    assert generated == [
        (name, benchmark._SEGMENT_COUNT, benchmark._SEED)
        for name in synthetic_names
    ]
    assert all("trace_provenance" not in case for case in cases[:-1])
    assert loaded == ["soc-epinions1-outdegree-v1"]
    assert cases[-1]["lengths"] is real_lengths
    assert cases[-1]["trace_provenance"] is provenance


def test_call_samples_use_balanced_rotating_order(
    comparison_modules, monkeypatch
):
    """Time one candidate per position before beginning the next round."""
    benchmark, _ = comparison_modules
    calls = []

    class FakeCuda:
        @staticmethod
        def synchronize():
            return None

    class FakeTorch:
        cuda = FakeCuda()

    clock = iter(range(100))
    monkeypatch.setattr(
        benchmark.time, "perf_counter_ns", lambda: next(clock)
    )
    launches = {
        name: (lambda candidate=name: calls.append(candidate))
        for name in ("a", "b", "c")
    }

    result = benchmark._interleaved_call_us(
        FakeTorch(), launches, warmups=0, samples=4
    )

    assert calls == [
        "a",
        "b",
        "c",
        "b",
        "c",
        "a",
        "c",
        "a",
        "b",
        "a",
        "b",
        "c",
    ]
    sample_counts = {
        name: len(record["samples_us"]) for name, record in result.items()
    }
    assert sample_counts == {
        "a": 4,
        "b": 4,
        "c": 4,
    }
    orders = benchmark._rotating_orders(launches, 8)
    counts = benchmark._order_position_counts(orders)
    assert all(
        max(positions) - min(positions) <= 1
        for positions in counts.values()
    )


def test_all_graphs_are_captured_before_any_replay(comparison_modules):
    """Prepare every candidate graph before interleaved replay begins."""
    benchmark, _ = comparison_modules
    events = []

    class FakeGraph:
        def __init__(self, graph_id):
            self.graph_id = graph_id

        def replay(self):
            events.append(("replay", self.graph_id))

    class FakeEvent:
        def record(self):
            return None

        def synchronize(self):
            return None

        def elapsed_time(self, other):
            del other
            return 1.0

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
                yield
                events.append(("capture_end", graph.graph_id))

            return capture()

        def Event(self, enable_timing):
            assert enable_timing
            return FakeEvent()

    class FakeTorch:
        cuda = FakeCuda()

    launches = {"a": lambda: None, "b": lambda: None}
    benchmark._interleaved_graph_us(
        FakeTorch(), launches, warmups=0, samples=1
    )

    first_replay = next(
        index for index, event in enumerate(events) if event[0] == "replay"
    )
    capture_ends = [
        index for index, event in enumerate(events) if event[0] == "capture_end"
    ]
    assert len(capture_ends) == 2
    assert max(capture_ends) < first_replay


def _measurement(median):
    return {
        "samples_us": [median],
        "summary_us": {"median": median, "q1": median, "q3": median},
    }


def _child(revision, median):
    timing = _measurement(median)
    graph = {"available": True, **_measurement(median + 2)}
    return {
        "benchmark": "swage-triton-comparison",
        "recorded_at": "ignored",
        "source": {
            "revision": revision,
            "worktree_clean": True,
            "dirty": [],
        },
        "environment": {"gpu": "test", "compute_capability": "sm_00"},
        "methodology": {"candidate_sampling": "rotating"},
        "results": [
            {
                "case": "segmented-sum",
                "distribution": "uniform",
                "timings": {
                    "swage_mixed": {
                        "call": timing,
                        "batched_event": _measurement(median + 1),
                        "graph": graph,
                    }
                },
                "preparation_only": {
                    "geometry": "fixed",
                    "timings": {
                        "swage_mixed": _measurement(median + 2.5)
                    },
                },
                "end_to_end": {
                    "warm_preparation": {
                        "timings": {"swage_mixed": _measurement(median + 3)}
                    },
                    "changing_geometry": {
                        "timings": {"swage_mixed": _measurement(median + 4)}
                    },
                },
            }
        ],
    }


def test_campaign_aggregates_process_medians(comparison_modules):
    """Aggregate child medians while retaining each process value."""
    _, campaign = comparison_modules

    aggregate = campaign._aggregate_children(
        [_child("same", 10.0), _child("same", 20.0)]
    )

    by_name = {
        row["measurement"]: row
        for row in aggregate["process_level_aggregates"]
    }
    call = by_name[
        "segmented-sum/distribution=uniform/swage_mixed/call_us"
    ]
    assert call["child_process_medians_us"] == [10.0, 20.0]
    assert call["median_of_process_medians_us"] == 15.0
    assert call["unit"] == "microseconds"
    preparation = by_name[
        "segmented-sum/distribution=uniform/swage_mixed/"
        "preparation_only_us"
    ]
    assert preparation["child_process_medians_us"] == [12.5, 22.5]
    assert preparation["median_of_process_medians_us"] == 17.5
    assert aggregate["agreement"] == {
        "source": True,
        "environment": True,
        "methodology": True,
        "result_metadata": True,
    }


def test_campaign_rejects_mixed_revisions(comparison_modules):
    """Never combine process evidence from different source revisions."""
    _, campaign = comparison_modules

    with pytest.raises(ValueError, match="source metadata differs"):
        campaign._aggregate_children(
            [_child("revision-a", 10.0), _child("revision-b", 11.0)]
        )


def test_campaign_rejects_mismatched_result_metadata(comparison_modules):
    """Reject process aggregation when result configuration differs."""
    _, campaign = comparison_modules
    changed = copy.deepcopy(_child("same", 11.0))
    changed["results"][0]["statistics"] = {"max": 4096}

    with pytest.raises(ValueError, match="result metadata differs"):
        campaign._aggregate_children([_child("same", 10.0), changed])


def test_nvidia_telemetry_records_unavailable(comparison_modules, monkeypatch):
    """Record an explicit unavailable state when nvidia-smi is absent."""
    _, campaign = comparison_modules
    monkeypatch.setattr(campaign.shutil, "which", lambda _: None)

    assert campaign._nvidia_telemetry() == {
        "available": False,
        "reason": "nvidia-smi not found",
    }


def _install_fake_nvidia_smi(
    campaign, monkeypatch, process_result
):
    """Install a two-query nvidia-smi fake and return captured commands."""
    gpu_result = campaign.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="0, GPU-test, 1410, 5001, 120.5, 55\n",
        stderr="",
    )
    results = iter((gpu_result, process_result))
    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return next(results)

    monkeypatch.setattr(campaign.shutil, "which", lambda _: "nvidia-smi")
    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    return commands


def _process_telemetry(processes):
    """Return the process-observation subset consumed by campaign policy."""
    return {
        "available": True,
        "compute_processes": {
            "available": True,
            "fields": {"used_memory_mib": "MiB"},
            "processes": processes,
        },
    }


def test_campaign_requires_explicit_gpu_execution_mode(
    comparison_modules, monkeypatch, tmp_path
):
    """Boundary samples alone cannot silently imply exclusive allocation."""
    _, campaign = comparison_modules
    base = [
        "run_triton_comparison_campaign.py",
        "--output-dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(campaign.sys, "argv", base)
    with pytest.raises(SystemExit):
        campaign._arguments()

    monkeypatch.setattr(
        campaign.sys,
        "argv",
        base
        + [
            "--exclusive-gpu-allocated",
            "--allow-shared-gpu-engineering",
        ],
    )
    with pytest.raises(SystemExit):
        campaign._arguments()
    monkeypatch.setattr(
        campaign.sys,
        "argv",
        base
        + [
            "--allow-shared-gpu-engineering",
            "--archival-source",
        ],
    )
    with pytest.raises(SystemExit):
        campaign._arguments()


    monkeypatch.setattr(
        campaign.sys, "argv", base + ["--exclusive-gpu-allocated"]
    )
    arguments = campaign._arguments()
    assert arguments.exclusive_gpu_allocated is True
    assert arguments.allow_shared_gpu_engineering is False
    assert arguments.archival_source is False


def test_nvidia_telemetry_records_no_compute_processes(
    comparison_modules, monkeypatch
):
    """An empty successful query is an observation, not an error."""
    _, campaign = comparison_modules
    process_result = campaign.subprocess.CompletedProcess(
        args=[], returncode=0, stdout="", stderr=""
    )
    commands = _install_fake_nvidia_smi(
        campaign, monkeypatch, process_result
    )

    telemetry = campaign._nvidia_telemetry()

    assert telemetry["compute_processes"] == {
        "available": True,
        "fields": {"used_memory_mib": "MiB"},
        "processes": [],
    }
    assert "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" in (
        commands[1]
    )


def test_nvidia_telemetry_records_foreign_compute_process(
    comparison_modules, monkeypatch
):
    """Preserve every field needed to identify a competing CUDA process."""
    _, campaign = comparison_modules
    process_result = campaign.subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout="GPU-other, 4242, /opt/foreign-worker, 8192\n",
        stderr="",
    )
    _install_fake_nvidia_smi(campaign, monkeypatch, process_result)

    telemetry = campaign._nvidia_telemetry()

    assert telemetry["compute_processes"]["processes"] == [
        {
            "gpu_uuid": "GPU-other",
            "pid": 4242,
            "process_name": "/opt/foreign-worker",
            "used_memory_mib": 8192,
        }
    ]


def test_nvidia_telemetry_records_process_query_failure(
    comparison_modules, monkeypatch
):
    """Keep device telemetry while marking process telemetry unavailable."""
    _, campaign = comparison_modules
    process_result = campaign.subprocess.CompletedProcess(
        args=[],
        returncode=9,
        stdout="",
        stderr="permission denied",
    )
    _install_fake_nvidia_smi(campaign, monkeypatch, process_result)

    telemetry = campaign._nvidia_telemetry()

    assert telemetry["available"] is True
    assert telemetry["gpus"][0]["sm_clock_mhz"] == 1410
    assert telemetry["compute_processes"] == {
        "available": False,
        "reason": "nvidia-smi exited with status 9: permission denied",
    }


@pytest.mark.parametrize(
    "telemetry, message",
    [
        (
            _process_telemetry(
                [
                    {
                        "gpu_uuid": "GPU-other",
                        "pid": 4242,
                        "process_name": "foreign-worker",
                        "used_memory_mib": 8192,
                    }
                ]
            ),
            "competing compute process",
        ),
        (
            {
                "available": True,
                "compute_processes": {
                    "available": False,
                    "reason": "permission denied",
                },
            },
            "compute-process telemetry unavailable",
        ),
    ],
)
def test_exclusive_campaign_rejects_before_child_launch(
    comparison_modules, monkeypatch, tmp_path, telemetry, message
):
    """Asserted-exclusive campaigns fail closed before starting a child."""
    _, campaign = comparison_modules
    output_dir = tmp_path / "campaign"
    arguments = campaign.argparse.Namespace(
        output_dir=output_dir,
        repetitions=1,
        suite="all",
        samples=1,
        warmups=0,
        exclusive_gpu_allocated=True,
        allow_shared_gpu_engineering=False,
        archival_source=True,
    )
    source = {
        "revision": "same",
        "worktree_clean": True,
        "dirty": [],
    }
    monkeypatch.setattr(campaign, "_arguments", lambda: arguments)
    monkeypatch.setattr(
        campaign, "_require_source_neutral_output", lambda *_: None
    )
    monkeypatch.setattr(campaign, "_source_metadata", lambda _: source)
    monkeypatch.setattr(campaign, "_nvidia_telemetry", lambda: telemetry)

    def unexpected_child(*_args, **_kwargs):
        raise AssertionError("benchmark child must not be launched")

    monkeypatch.setattr(campaign.subprocess, "run", unexpected_child)

    with pytest.raises(RuntimeError, match=message):
        campaign.main()


def test_shared_gpu_override_is_explicitly_ineligible(
    comparison_modules, monkeypatch, tmp_path
):
    """Engineering override records competitors and cannot look archival."""
    _, campaign = comparison_modules
    output_dir = tmp_path / "campaign"
    arguments = campaign.argparse.Namespace(
        output_dir=output_dir,
        repetitions=1,
        suite="all",
        samples=1,
        warmups=0,
        exclusive_gpu_allocated=False,
        allow_shared_gpu_engineering=True,
        archival_source=False,
    )
    source = {
        "revision": "same",
        "worktree_clean": True,
        "dirty": [],
    }
    process = {
        "gpu_uuid": "GPU-other",
        "pid": 4242,
        "process_name": "foreign-worker",
        "used_memory_mib": 8192,
    }
    telemetry = _process_telemetry([process])
    monkeypatch.setattr(campaign, "_arguments", lambda: arguments)
    monkeypatch.setattr(
        campaign, "_require_source_neutral_output", lambda *_: None
    )
    monkeypatch.setattr(campaign, "_source_metadata", lambda _: source)
    monkeypatch.setattr(campaign, "_nvidia_telemetry", lambda: telemetry)

    def fake_child(command, **_kwargs):
        child_path = pathlib.Path(
            command[command.index("--output") + 1]
        )
        child_path.write_text(campaign.json.dumps(_child("same", 10.0)))
        return campaign.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_child)

    campaign.main()

    manifest = campaign.json.loads(
        (output_dir / "manifest.json").read_text()
    )
    assert manifest["archival_eligible"] is False
    assert manifest["controls"]["gpu_execution_mode"] == (
        "shared-gpu-engineering"
    )
    assert manifest["controls"]["allow_shared_gpu_engineering"] is True
    assert any(
        "exclusive GPU allocation was not asserted" in reason
        for reason in manifest["archival_ineligibility_reasons"]
    )
    assert any(
        "pid=4242" in reason
        for reason in manifest["archival_ineligibility_reasons"]
    )
    assert (
        manifest["children"][0]["nvidia_telemetry"]["pre_process"][
            "compute_processes"
        ]["processes"]
        == [process]
    )



@pytest.mark.parametrize(
    ("archival_source", "eligible", "reasons"),
    [
        (
            False,
            False,
            ["archival source revision was not asserted"],
        ),
        (True, True, []),
    ],
)
def test_exclusive_campaign_archival_eligibility(
    comparison_modules,
    monkeypatch,
    tmp_path,
    archival_source,
    eligible,
    reasons,
):
    """Require isolation and final-source assertions for archival use."""
    _, campaign = comparison_modules
    output_dir = tmp_path / "campaign"
    arguments = campaign.argparse.Namespace(
        output_dir=output_dir,
        repetitions=1,
        suite="all",
        samples=1,
        warmups=0,
        exclusive_gpu_allocated=True,
        allow_shared_gpu_engineering=False,
        archival_source=archival_source,
    )
    source = {
        "revision": "detached-head",
        "worktree_clean": True,
        "dirty": [],
    }
    telemetry = _process_telemetry([])
    monkeypatch.setattr(campaign, "_arguments", lambda: arguments)
    monkeypatch.setattr(
        campaign, "_require_source_neutral_output", lambda *_: None
    )
    monkeypatch.setattr(campaign, "_source_metadata", lambda _: source)
    monkeypatch.setattr(campaign, "_nvidia_telemetry", lambda: telemetry)

    def fake_child(command, **_kwargs):
        child_path = pathlib.Path(
            command[command.index("--output") + 1]
        )
        child = _child("detached-head", 10.0)
        child_path.write_text(campaign.json.dumps(child))
        return campaign.subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_child)

    campaign.main()

    manifest = campaign.json.loads(
        (output_dir / "manifest.json").read_text()
    )
    assert manifest["archival_eligible"] is eligible
    assert manifest["controls"]["exclusive_gpu_allocated"] is True
    assert manifest["controls"]["archival_source"] is archival_source
    assert manifest["archival_ineligibility_reasons"] == reasons


def test_pure_task_partition_and_schema_helpers(comparison_modules):
    """Keep the fused boundary and unit-bearing aggregate names explicit."""
    benchmark, campaign = comparison_modules

    assert benchmark._partition_lengths([0, 1, 32, 33, 4096]) == (
        [0, 1, 2],
        [3, 4],
    )
    with pytest.raises(ValueError, match="up to 4096"):
        benchmark._partition_lengths([4097])

    medians = campaign._process_medians(_child("same", 10.0))
    assert (
        "segmented-sum/distribution=uniform/swage_mixed/"
        "end_to_end_changing_geometry_us"
    ) in medians
    assert (
        "segmented-sum/distribution=uniform/swage_mixed/"
        "preparation_only_us"
    ) in medians
