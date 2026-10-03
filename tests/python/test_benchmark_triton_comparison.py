# tests/python/test_benchmark_triton_comparison.py
"""Tests for comparison timing and repeated-campaign evidence contracts."""

import contextlib
import importlib
import json
import pathlib
import statistics
from types import SimpleNamespace

import pytest
from benchmark_campaign_fixtures import make_child


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
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: next(clock))
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
        max(positions) - min(positions) <= 1 for positions in counts.values()
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
    benchmark._interleaved_graph_us(FakeTorch(), launches, warmups=0, samples=1)

    first_replay = next(
        index for index, event in enumerate(events) if event[0] == "replay"
    )
    capture_ends = [
        index for index, event in enumerate(events) if event[0] == "capture_end"
    ]
    assert len(capture_ends) == 2
    assert max(capture_ends) < first_replay


def test_campaign_aggregates_process_medians(comparison_modules):
    """Aggregate raw process medians while retaining each process value."""
    _, campaign = comparison_modules
    children = [
        make_child(median=median, suite="segmented-sum")
        for median in (10.0, 20.0)
    ]

    aggregate = campaign.aggregate_children(children)

    by_name = {
        row["measurement"]: row for row in aggregate["process_level_aggregates"]
    }
    distribution = children[0]["results"][0]["distribution"]
    prefix = f"segmented-sum/distribution={distribution}/swage_mixed"
    call = by_name[f"{prefix}/call_us"]
    call_medians = [
        statistics.median(
            child["results"][0]["timings"]["swage_mixed"]["call"]["samples_us"]
        )
        for child in children
    ]
    assert call["child_process_medians_us"] == call_medians
    assert call["median_of_process_medians_us"] == statistics.median(
        call_medians
    )
    assert call["unit"] == "microseconds"
    planning = by_name[f"{prefix}/planning_us"]
    planning_medians = [
        statistics.median(
            child["results"][0]["planning"]["timings"]["swage_mixed"][
                "samples_us"
            ]
        )
        for child in children
    ]
    assert planning["child_process_medians_us"] == planning_medians
    assert planning["median_of_process_medians_us"] == statistics.median(
        planning_medians
    )
    compilation = by_name["segmented-sum/swage_total/compilation_us"]
    compilation_medians = [
        statistics.median(
            child["compilation"]["timings"]["swage_total"]["samples_us"]
        )
        for child in children
    ]
    assert compilation["child_process_medians_us"] == compilation_medians
    assert compilation["median_of_process_medians_us"] == statistics.median(
        compilation_medians
    )


def test_campaign_rejects_mixed_revisions(comparison_modules):
    """Never combine process evidence from different source revisions."""
    _, campaign = comparison_modules
    first = make_child()
    changed = make_child(median=11.0)
    changed["source"]["revision"] = "b" * 40
    changed["environment"]["compiler"]["source_revision"] = "b" * 40

    with pytest.raises(ValueError, match="source metadata differs"):
        campaign.aggregate_children([first, changed])


def test_nvidia_telemetry_records_unavailable(comparison_modules, monkeypatch):
    """Record an explicit unavailable state when nvidia-smi is absent."""
    _, campaign = comparison_modules
    monkeypatch.setattr(campaign.shutil, "which", lambda _: None)

    assert campaign._nvidia_telemetry() == {
        "available": False,
        "reason": "nvidia-smi not found",
    }


def _install_fake_nvidia_smi(campaign, monkeypatch, process_result):
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
    """Return a complete NVIDIA boundary observation."""
    return {
        "available": True,
        "recorded_at": "2026-09-08T12:00:00+00:00",
        "fields": {
            "sm_clock_mhz": "MHz",
            "memory_clock_mhz": "MHz",
            "power_draw_watts": "W",
            "temperature_celsius": "degrees Celsius",
        },
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-test",
                "sm_clock_mhz": 1410,
                "memory_clock_mhz": 5001,
                "power_draw_watts": 120.5,
                "temperature_celsius": 55.0,
            }
        ],
        "compute_processes": {
            "available": True,
            "fields": {"used_memory_mib": "MiB"},
            "processes": processes,
        },
    }


@pytest.fixture
def campaign_setup(comparison_modules, monkeypatch, tmp_path):
    """Set up a complete CPU-only campaign with schema-valid children."""
    _, campaign = comparison_modules
    child = make_child()
    arguments = campaign.argparse.Namespace(
        output_dir=tmp_path / "campaign",
        repetitions=1,
        suite="vadd",
        samples=2,
        warmups=1,
        exclusive_gpu_allocated=True,
        allow_shared_gpu_engineering=False,
        archival_source=True,
    )
    monkeypatch.setattr(campaign, "_arguments", lambda: arguments)
    monkeypatch.setattr(campaign, "_source_metadata", lambda _: child["source"])
    monkeypatch.setattr(
        campaign, "_nvidia_telemetry", lambda: _process_telemetry([])
    )

    def fake_child(command, **_kwargs):
        child_path = pathlib.Path(command[command.index("--output") + 1])
        child_path.write_text(json.dumps(child))
        return campaign.subprocess.CompletedProcess(
            command, 0, stdout="child complete\n", stderr=""
        )

    monkeypatch.setattr(campaign.subprocess, "run", fake_child)
    return campaign, arguments, child


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
    commands = _install_fake_nvidia_smi(campaign, monkeypatch, process_result)

    telemetry = campaign._nvidia_telemetry()

    assert telemetry["compute_processes"] == {
        "available": True,
        "fields": {"used_memory_mib": "MiB"},
        "processes": [],
    }
    assert (
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory"
        in (commands[1])
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
                **_process_telemetry([]),
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
    campaign_setup, monkeypatch, telemetry, message
):
    """Fail closed before launch and retain both boundary observations."""
    campaign, arguments, _ = campaign_setup
    monkeypatch.setattr(campaign, "_nvidia_telemetry", lambda: telemetry)

    def unexpected_child(*_args, **_kwargs):
        raise AssertionError("benchmark child must not be launched")

    monkeypatch.setattr(campaign.subprocess, "run", unexpected_child)

    with pytest.raises(RuntimeError, match=message):
        campaign.main()

    assert not (arguments.output_dir / "manifest.json").exists()
    assert not (arguments.output_dir / "process-000.json").exists()
    failure = json.loads(
        (arguments.output_dir / "failure-observations.json").read_text()
    )
    observation = failure["process_observations"][0]
    assert observation["launched"] is False
    assert observation["nvidia_telemetry"] == {
        "pre_process": telemetry,
        "post_process": telemetry,
    }


def test_shared_gpu_override_is_explicitly_ineligible(
    campaign_setup, monkeypatch
):
    """Engineering override records competitors and cannot look archival."""
    campaign, arguments, _ = campaign_setup
    arguments.exclusive_gpu_allocated = False
    arguments.allow_shared_gpu_engineering = True
    arguments.archival_source = False
    process = {
        "gpu_uuid": "GPU-other",
        "pid": 4242,
        "process_name": "foreign-worker",
        "used_memory_mib": 8192,
    }
    telemetry = _process_telemetry([process])
    monkeypatch.setattr(campaign, "_nvidia_telemetry", lambda: telemetry)

    campaign.main()

    manifest_path = arguments.output_dir / "manifest.json"
    manifest = campaign.load_campaign(manifest_path, require_archival=False)[
        "manifest"
    ]
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
    assert manifest["children"][0]["nvidia_telemetry"] == {
        "pre_process": telemetry,
        "post_process": telemetry,
    }
    with pytest.raises(ValueError):
        campaign.load_campaign(manifest_path)


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
    campaign_setup, archival_source, eligible, reasons
):
    """Require isolation and final-source assertions for archival use."""
    campaign, arguments, _ = campaign_setup
    arguments.archival_source = archival_source

    campaign.main()

    manifest = campaign.load_campaign(
        arguments.output_dir / "manifest.json",
        require_archival=eligible,
    )["manifest"]
    assert manifest["schema_version"] == 1
    assert manifest["archival_eligible"] is eligible
    assert manifest["controls"]["exclusive_gpu_allocated"] is True
    assert manifest["controls"]["archival_source"] is archival_source
    assert manifest["archival_ineligibility_reasons"] == reasons


def test_campaign_uses_fresh_external_caches(campaign_setup, monkeypatch):
    """Each child starts empty, cannot see prior caches, and leaves none."""
    campaign, arguments, child = campaign_setup
    arguments.repetitions = 2
    cache_directories = []
    monkeypatch.setenv("SWAGE_CACHE_DIR", "inherited-swage-cache")
    monkeypatch.setenv("TRITON_CACHE_DIR", "inherited-triton-cache")

    def fake_child(command, *, cwd, env, **_kwargs):
        for variable in ("SWAGE_CACHE_DIR", "TRITON_CACHE_DIR"):
            cache = pathlib.Path(env[variable])
            assert cache.is_dir()
            assert list(cache.iterdir()) == []
            assert not cache.is_relative_to(cwd)
            assert not cache.is_relative_to(arguments.output_dir)
            assert cache not in cache_directories
            cache_directories.append(cache)
            (cache / "compiled-artifact").write_bytes(b"cache payload")
        child_path = pathlib.Path(command[command.index("--output") + 1])
        child_path.write_text(json.dumps(child))
        return campaign.subprocess.CompletedProcess(
            command, 0, stdout="", stderr=""
        )

    monkeypatch.setattr(campaign.subprocess, "run", fake_child)

    campaign.main()

    assert all(not cache.exists() for cache in cache_directories)
    loaded = campaign.load_campaign(arguments.output_dir / "manifest.json")
    assert loaded["manifest"]["process_count"] == 2


@pytest.mark.parametrize(
    "failure_kind",
    (
        "child-exit",
        "invalid-child",
        "duplicate-child",
        "post-competition",
        "control-drift",
    ),
)
def test_campaign_failures_preserve_evidence(
    campaign_setup, monkeypatch, failure_kind
):
    """Failed executions and rejected evidence never become a manifest."""
    campaign, arguments, child = campaign_setup
    pre = _process_telemetry([])
    post = _process_telemetry([])
    if failure_kind == "post-competition":
        post["compute_processes"]["processes"] = [
            {
                "gpu_uuid": "GPU-test",
                "pid": 4242,
                "process_name": "competing-worker",
                "used_memory_mib": 8192,
            }
        ]
    if failure_kind == "control-drift":
        arguments.samples = 3
    boundaries = iter((pre, post))
    monkeypatch.setattr(campaign, "_nvidia_telemetry", lambda: next(boundaries))
    raw_child = json.dumps(child)
    if failure_kind in ("child-exit", "invalid-child"):
        raw_child = '{"partial":'
    elif failure_kind == "duplicate-child":
        raw_child = '{"schema_version": 1,' + raw_child[1:]
    cache_directories = []

    def fake_child(command, *, env, **_kwargs):
        cache_directories.extend(
            pathlib.Path(env[variable])
            for variable in ("SWAGE_CACHE_DIR", "TRITON_CACHE_DIR")
        )
        child_path = pathlib.Path(command[command.index("--output") + 1])
        child_path.write_text(raw_child)
        return campaign.subprocess.CompletedProcess(
            command,
            1 if failure_kind == "child-exit" else 0,
            stdout="retained child stdout\n",
            stderr="retained child stderr\n",
        )

    monkeypatch.setattr(campaign.subprocess, "run", fake_child)
    if failure_kind == "child-exit":
        expected_error = campaign.subprocess.CalledProcessError
    elif failure_kind == "post-competition":
        expected_error = RuntimeError
    else:
        expected_error = ValueError

    with pytest.raises(expected_error):
        campaign.main()

    assert not (arguments.output_dir / "manifest.json").exists()
    assert not (arguments.output_dir / "manifest.pending.json").exists()
    assert (arguments.output_dir / "process-000.json").read_text() == raw_child
    assert all(not cache.exists() for cache in cache_directories)
    failure = json.loads(
        (arguments.output_dir / "failure-observations.json").read_text()
    )
    observation = failure["process_observations"][0]
    assert observation["stdout"] == "retained child stdout\n"
    assert observation["stderr"] == "retained child stderr\n"
    assert observation["nvidia_telemetry"] == {
        "pre_process": pre,
        "post_process": post,
    }
    if failure_kind == "control-drift":
        assert (arguments.output_dir / "failure-manifest.json").is_file()


def test_non_release_compiler_is_not_archival(campaign_setup):
    """Exclusive allocation does not make a Debug build archival."""
    campaign, arguments, child = campaign_setup
    child["environment"]["compiler"]["build_type"] = "Debug"

    campaign.main()

    manifest_path = arguments.output_dir / "manifest.json"
    manifest = campaign.load_campaign(manifest_path, require_archival=False)[
        "manifest"
    ]
    assert manifest["archival_eligible"] is False
    with pytest.raises(ValueError):
        campaign.load_campaign(manifest_path)


def test_pure_task_partition_boundary(comparison_modules):
    """Keep the fused manual-partition boundary explicit."""
    benchmark, _ = comparison_modules

    assert benchmark._partition_lengths([0, 1, 32, 33, 4096]) == (
        [0, 1, 2],
        [3, 4],
    )
    with pytest.raises(ValueError, match="up to 4096"):
        benchmark._partition_lengths([4097])


@pytest.mark.parametrize(
    "invalid", ["missing", "file", "nonempty", "same", "alias"]
)
def test_compilation_rejects_unfresh_caches(
    comparison_modules, monkeypatch, tmp_path, invalid
):
    """Missing, populated, or aliased cache locations cannot be called fresh."""
    benchmark, _ = comparison_modules
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
        benchmark._require_empty_compilation_caches()


def _cpu_host_plan(
    torch,
    host_offsets,
    value_count,
    segment_count,
    *,
    warp_max_elements,
    cta_chunk_elements,
):
    """Replace native host classification, retaining real plan/spec types."""
    from swage._segmented_plan import _expected_plan, _SegmentedPlan

    warp, cta, partial, merge = _expected_plan(
        host_offsets, warp_max_elements, cta_chunk_elements
    )
    host_plan = _SegmentedPlan(
        tuple(warp), tuple(cta), tuple(partial), tuple(merge), ()
    )
    return (
        "semantic",
        host_plan,
        "sm_86",
        {
            "warp_max_elements": warp_max_elements,
            "cta_chunk_elements": cta_chunk_elements,
        },
    )


@pytest.fixture
def compilation_setup(comparison_modules, monkeypatch, tmp_path):
    """Provide compiler clocks without CUDA execution or native builds."""
    from swage import _segmented_qualification, _segmented_runtime

    benchmark, _ = comparison_modules
    for variable in ("SWAGE_CACHE_DIR", "TRITON_CACHE_DIR"):
        directory = tmp_path / variable
        directory.mkdir()
        monkeypatch.setenv(variable, str(directory))
    elapsed = [0]
    events = []
    calls = {name: [] for name in ("fixed", "packed", "cta", "fused")}
    monkeypatch.setattr(benchmark.time, "perf_counter_ns", lambda: elapsed[0])

    def materialize(*args, **kwargs):
        elapsed[0] += 1_000_000
        return _cpu_host_plan(*args, **kwargs)

    def compile_artifact(*args, **kwargs):
        events.append("swage")
        elapsed[0] += 2_000

    def forbidden(*args, **kwargs):
        pytest.fail(
            "compile-only phase reached binding, module load, or launch"
        )

    class CompileOnlyKernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            forbidden()

        def warmup(self, *args, **kwargs):
            events.append(self.name)
            calls[self.name].append((args, kwargs))
            elapsed[0] += 1_000

    monkeypatch.setattr(
        _segmented_qualification, "_materialize_planned_sum_host", materialize
    )
    monkeypatch.setattr(
        _segmented_runtime, "_compile_artifact", compile_artifact
    )
    monkeypatch.setattr(_segmented_runtime, "_bind_artifact", forbidden)
    monkeypatch.setattr(_segmented_runtime, "_load_entries", forbidden)
    torch = SimpleNamespace(float32=object(), int32=object())
    kernels = tuple(CompileOnlyKernel(name) for name in calls)
    return benchmark, torch, kernels, calls, events


def test_compilation_covers_scalar_specializations_without_launch(
    compilation_setup,
):
    """Compile every eligible signature, but skip empty partitions."""
    benchmark, torch, kernels, calls, events = compilation_setup
    cases = [
        {"distribution": "tiny", "lengths": [0, 1, 32]},
        {"distribution": "large", "lengths": [33] * 16},
        {"distribution": "mixed", "lengths": [1, 4096]},
        {"distribution": "tiny-repeat", "lengths": [32, 1, 0]},
    ]
    result = benchmark._measure_segmented_compilation(torch, cases, kernels)

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
    assert events == [
        *["swage"] * 3,
        *["fixed"] * len(expected_fixed),
        *["packed"] * 3,
        *["cta"] * 8,
        *["fused"] * 4,
    ]
    assert result["timings"]["swage_total"]["samples_us"] == [6.0]
    assert result["timings"]["triton_total"]["samples_us"] == [
        float(len(expected_fixed) + 3 + 8 + 4)
    ]
    for component in result["component_order"]:
        assert result["timings"][component]["samples_us"][0] > 0

    # Validate the emitted phase against independently declared execution
    # geometry, without pretending the not-yet-run result rows are evidence.
    from benchmark_campaign import _compilation

    rows = [
        {
            "distribution": name,
            "statistics": {"total": total},
            "segment_count": count,
            "triton_sweep_configs": [
                {"block": block, "num_warps": warps}
                for block, warps in fixed_configs
                if block >= maximum
            ],
            "matched_task_partition_triton": {
                "warp_tasks": warp_count,
                "cta_tasks": cta_count,
            },
            "triton_fused_contract": {"warp_programs": warp_programs},
        }
        for (
            name,
            total,
            count,
            maximum,
            warp_count,
            cta_count,
            warp_programs,
        ) in (
            ("tiny", 33, 3, 32, 3, 0, 1),
            ("large", 528, 16, 33, 0, 16, 0),
            ("mixed", 4097, 2, 4096, 1, 1, 1),
            ("tiny-repeat", 33, 3, 32, 3, 0, 1),
        )
    ]
    _compilation(result, rows)


def test_compilation_rejects_split_artifacts_before_timing(compilation_setup):
    """Reject split workloads rather than omit their partial/merge costs."""
    benchmark, torch, kernels, _, events = compilation_setup
    with pytest.raises(ValueError, match="split artifacts"):
        benchmark._measure_segmented_compilation(
            torch, [{"distribution": "split", "lengths": [1, 4097]}], kernels
        )
    assert events == []


def test_swage_planning_materializes_geometry_without_execution(
    comparison_modules, monkeypatch
):
    """Plan changing geometry with compilation, binding, and load forbidden."""
    torch = pytest.importorskip("torch")
    from swage import (
        _segmented_qualification,
        _segmented_runtime,
        _segmented_validation,
    )

    benchmark, _ = comparison_modules
    validate_shapes = _segmented_validation._validate_shapes
    monkeypatch.setattr(
        _segmented_validation,
        "_validate_shapes",
        lambda values, offsets, output, validator: validate_shapes(
            values, offsets, output, validator, require_cuda=False
        ),
    )
    monkeypatch.setattr(
        _segmented_qualification,
        "_materialize_planned_sum_host",
        _cpu_host_plan,
    )
    synchronized = []
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda: synchronized.append(True)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: SimpleNamespace(record=lambda stream: None)
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "planning reached compilation, binding, loading, "
            "or output allocation"
        )

    for name in ("_compile_artifact", "_bind_artifact", "_load_entries"):
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

    first = benchmark._plan_swage_sum(torch, values, offsets, output)
    changed = benchmark._plan_swage_sum(torch, values, changed_offsets, output)
    assert first.buffers["mixed_task_ids"].tolist() == [0, 1, 3, 2]
    assert changed.buffers["mixed_task_ids"].tolist() == [1, 2, 3, 0]
    assert first.buffers["task_ids"].tolist() == [0, 1, 2, 3]
    assert output.tolist() == [-1.0] * 4
    assert synchronized == [True, True]
    with pytest.raises(ValueError, match="nondecreasing"):
        benchmark._plan_swage_sum(torch, values, invalid_offsets, output)


@pytest.mark.parametrize(
    ("lengths", "expected_rows"),
    [
        ([0, 1, 3], [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        ([0, 0, 0], [[], [], []]),
        ([], []),
    ],
)
def test_padded_identity_reduction_and_storage(
    comparison_modules, lengths, expected_rows
):
    """Padding preserves exact sums and reports actual storage, even empty."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    benchmark, _ = comparison_modules
    padded, layout = benchmark._padded_identity_input(torch, lengths)
    output = torch.full((len(lengths),), float("nan"), device="cuda")
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


def test_comparison_emits_framework_version_subclasses(
    comparison_modules, monkeypatch, tmp_path
):
    """Framework string subclasses must produce schema-valid CLI evidence."""
    from swage import _cuda_backend, env

    benchmark, campaign = comparison_modules
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
        benchmark.sys.modules,
        "torch",
        SimpleNamespace(
            __version__=FrameworkVersion("2.12.0+cu130"),
            version=SimpleNamespace(cuda="13.0"),
            cuda=cuda,
        ),
    )
    monkeypatch.setitem(
        benchmark.sys.modules, "triton", SimpleNamespace(__version__="3.7.0")
    )
    monkeypatch.setattr(
        env,
        "report",
        lambda: {
            "native": {**expected["environment"]["compiler"], "available": True}
        },
    )
    monkeypatch.setattr(_cuda_backend, "driver_version", lambda: "13.0")
    monkeypatch.setattr(
        benchmark, "_git_metadata", lambda _: expected["source"]
    )
    monkeypatch.setattr(benchmark, "_run_vadd", lambda *_: expected["results"])
    output = tmp_path / "child.json"
    monkeypatch.setattr(
        benchmark,
        "_arguments",
        lambda: SimpleNamespace(
            output=output, samples=2, warmups=1, suite="vadd"
        ),
    )
    benchmark.main()
    child = campaign.load_unique_json(output)
    campaign.validate_child(child)
    assert child["environment"]["pytorch"] == "2.12.0+cu130"
