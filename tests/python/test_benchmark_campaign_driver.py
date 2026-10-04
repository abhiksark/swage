# tests/python/test_benchmark_campaign_driver.py
"""Tests for the driver that repeats a benchmark in independent processes.

The driver is benchmarks/run_triton_comparison_campaign.py. It took over
the summary, the references, and the re-summarizing of the process driver
of the review stack, whose tests are ported here.
"""

import copy
import hashlib
import importlib
import json
import lzma
import pathlib
import statistics
import subprocess
import sys

import pytest
from benchmark_campaign_fixtures import make_child

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RECORD = _ROOT / "benchmarks/results/segmented-sum-a6000-sm86-453c56e"
_FRESH = "benchmarks/benchmark_fresh_offsets.py"


@pytest.fixture
def campaign(monkeypatch):
    """Import the driver as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("run_triton_comparison_campaign")


def _provenance(seen=False):
    """Return the provenance block of one process."""
    return {
        "gpu": "NVIDIA RTX A6000",
        "gpu_uuid": "GPU-1",
        "cpu_model": "Test CPU",
        "pytorch": "2.12.0+cu130",
        "triton": "3.7.0",
        "native_sha256": {"/build/lib.so": "a" * 64},
        "loaded_ptx": [
            {"kernel": "segmented_sum", "sha256": "b" * 64, "bytes": 2}
        ],
        "other_compute_process_seen": seen,
        "gpu_state_before": {"gpu": {"temperature.gpu": "50"}},
        "gpu_state_after": {"gpu": {"temperature.gpu": "60"}},
        "cpu_frequency_before": {"governors": {"powersave": 24}},
        "cpu_frequency_after": {"governors": {"powersave": 24}},
        "cpu_governor_unchanged": True,
    }


def _fresh_record(swage, torch, pad=None, seen=False, looped=None):
    """Return a fresh-offsets record with one power-law row."""
    medians = {"swage_mixed": swage, "torch": torch}
    if pad is not None:
        medians["torch_pad_to_max"] = pad
    if looped is not None:
        medians["triton_looped_b256_w4"] = looped
    return {
        "benchmark": "fresh-offsets-segmented-sum",
        "recorded_at": "2026-10-01T00:00:00+00:00",
        "smoke": False,
        "source": {"revision": "abc123", "worktree_clean": True, "dirty": []},
        "provenance": _provenance(seen),
        "results": [
            {
                "distribution": "power-law",
                "summary_us": {
                    name: {"median": median, "q1": median, "q3": median}
                    for name, median in medians.items()
                },
                "effective_gb_per_s": {
                    name: {"median": 1_000.0 / median}
                    for name, median in medians.items()
                },
            }
        ],
    }


def _comparison_record():
    """Return a comparison record of an earlier revision.

    It has a vadd row and a segmented row, and no raw samples.
    """

    def timing(call, graph):
        return {
            "call": {
                "summary_us": {"median": call},
                "effective_gb_per_s": 100.0 / call,
            },
            "batched_event": {
                "summary_us": {"median": call / 2},
                "effective_gb_per_s": 200.0 / call,
            },
            "graph": (
                {"available": False, "error": "capture failed"}
                if graph is None
                else {
                    "available": True,
                    "summary_us": {"median": graph},
                    "effective_gb_per_s": 100.0 / graph,
                }
            ),
        }

    return {
        "benchmark": "swage-triton-comparison",
        "recorded_at": "2026-10-01T00:00:00+00:00",
        "source": {"revision": "abc123", "worktree_clean": True, "dirty": []},
        "provenance": _provenance(),
        "results": [
            {
                "case": "vadd",
                "n": 1024,
                "timings": {
                    "swage": timing(20.0, 2.0),
                    "torch": timing(10.0, 1.0),
                },
            },
            {
                "case": "segmented-sum",
                "distribution": "power-law",
                "seed": 7,
                "timings": {
                    "swage_mixed": timing(30.0, 6.0),
                    "torch": timing(15.0, 4.0),
                    "triton_looped_b128_w1": timing(12.0, None),
                },
            },
        ],
    }


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


def _writing_run(campaign, records, calls):
    """Return a fake ``subprocess.run`` that writes one record per child."""
    remaining = iter(records)

    def run(command, **options):
        calls.append((command, options))
        output = pathlib.Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(next(remaining)))
        return campaign.subprocess.CompletedProcess(command, 0, "", "")

    return run


@pytest.fixture
def quiet_machine(campaign, monkeypatch):
    """Report a clean source and an idle GPU without running anything."""
    source = {"revision": "a" * 40, "worktree_clean": True, "dirty": []}
    monkeypatch.setattr(campaign, "_source_metadata", lambda _: source)
    monkeypatch.setattr(
        campaign, "_nvidia_telemetry", lambda: _process_telemetry([])
    )
    return campaign


# The repeated comparison campaign.


def test_campaign_aggregates_process_medians(campaign):
    """Aggregate raw process medians while retaining each process value."""
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


def test_campaign_rejects_mixed_revisions(campaign):
    """Never combine process evidence from different source revisions."""
    first = make_child()
    changed = make_child(median=11.0)
    changed["source"]["revision"] = "b" * 40
    changed["environment"]["compiler"]["source_revision"] = "b" * 40

    with pytest.raises(ValueError, match="source metadata differs"):
        campaign.aggregate_children([first, changed])


def test_campaign_accepts_the_timer_ticks_of_each_process(campaign):
    """Timer ticks are measured per process and listed per process."""
    children = [make_child(), make_child()]
    children[1]["methodology"]["timer_ticks_us"]["clock"] = 0.06
    for row in children[1]["results"]:
        for metrics in row["timings"].values():
            call = metrics["call"]
            call["timer_tick_us"] = 0.06
            call["tick_fraction_of_sample"] = (
                0.06 / call["summary_us"]["median"]
            )

    aggregate = campaign.aggregate_children(children)

    assert "timer_ticks_us" not in aggregate["methodology"]
    assert aggregate["methodology"]["timer_ticks_us_per_process"] == [
        {"clock": 0.05, "event": 0.032},
        {"clock": 0.06, "event": 0.032},
    ]
    children[1]["methodology"]["values"] = "normal"
    with pytest.raises(ValueError, match="methodology metadata differs"):
        campaign.aggregate_children(children)


def test_nvidia_telemetry_records_unavailable(campaign, monkeypatch):
    """Record an explicit unavailable state when nvidia-smi is absent."""
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


@pytest.fixture
def campaign_setup(quiet_machine, monkeypatch, tmp_path):
    """Set up a complete CPU-only campaign with schema-valid children."""
    campaign = quiet_machine
    child = make_child()
    arguments = campaign._arguments(
        [
            "--output-dir",
            str(tmp_path / "campaign"),
            "--repetitions",
            "2",
            "--suite",
            "vadd",
            "--samples",
            "2",
            "--warmups",
            "1",
            "--exclusive-gpu-allocated",
            "--archival-source",
        ]
    )
    monkeypatch.setattr(campaign, "_arguments", lambda argv=None: arguments)

    def fake_child(command, **_kwargs):
        child_path = pathlib.Path(command[command.index("--output") + 1])
        child_path.write_text(json.dumps(child))
        return campaign.subprocess.CompletedProcess(
            command, 0, stdout="child complete\n", stderr=""
        )

    monkeypatch.setattr(campaign.subprocess, "run", fake_child)
    return campaign, arguments, child


def test_campaign_requires_explicit_gpu_execution_mode(
    campaign, monkeypatch, tmp_path
):
    """Boundary samples alone cannot silently imply exclusive allocation."""
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
    assert (arguments.suite, arguments.samples, arguments.warmups) == (
        "all",
        100,
        25,
    )


def test_nvidia_telemetry_records_no_compute_processes(campaign, monkeypatch):
    """An empty successful query is an observation, not an error."""
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
    campaign, monkeypatch
):
    """Preserve every field needed to identify a competing CUDA process."""
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


def test_nvidia_telemetry_records_process_query_failure(campaign, monkeypatch):
    """Keep device telemetry while marking process telemetry unavailable."""
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
    assert manifest["schema_version"] == 2
    assert manifest["archival_eligible"] is eligible
    assert manifest["controls"]["exclusive_gpu_allocated"] is True
    assert manifest["controls"]["archival_source"] is archival_source
    assert manifest["archival_ineligibility_reasons"] == reasons


def test_campaign_uses_fresh_external_caches(campaign_setup, monkeypatch):
    """Each child starts empty, cannot see prior caches, and leaves none."""
    campaign, arguments, child = campaign_setup
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
    arguments.repetitions = 1
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
        raw_child = '{"schema_version": 2,' + raw_child[1:]
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
    assert not (arguments.output_dir / "summary.json").exists()
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


def test_comparison_campaign_writes_a_manifest_and_a_summary(
    quiet_machine, monkeypatch, tmp_path, capsys
):
    """Summarize kernels, phases, and compilation beside the manifest."""
    campaign = quiet_machine
    calls = []
    children = [
        make_child(median=median, suite="all") for median in (10.0, 30.0)
    ]
    monkeypatch.setattr(
        campaign.subprocess, "run", _writing_run(campaign, children, calls)
    )
    output_dir = tmp_path / "run"

    campaign.main(
        [
            "--output-dir",
            str(output_dir),
            "--repetitions",
            "2",
            "--samples",
            "2",
            "--warmups",
            "1",
            "--allow-shared-gpu-engineering",
            "--",
            "--seeds",
            "7",
        ]
    )

    assert [command[4:] for command, _ in calls] == [
        ["--suite", "all", "--samples", "2", "--warmups", "1", "--seeds", "7"]
    ] * 2
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "manifest.json",
        "process-000.json",
        "process-001.json",
        "summary.json",
    ]
    manifest = campaign.load_campaign(
        output_dir / "manifest.json", require_archival=False
    )["manifest"]
    assert manifest["controls"]["harness_options"] == ["--seeds", "7"]
    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["references"] == ["torch", "torch_segment_reduce"]
    assert summary["command"] == [
        "benchmarks/benchmark_triton_comparison.py",
        "--suite",
        "all",
        "--samples",
        "2",
        "--warmups",
        "1",
        "--seeds",
        "7",
    ]
    row = summary["rows"]["many-tiny seed=7"]
    assert set(row) == {
        "call",
        "batched_event",
        "graph",
        "planning",
        "end_to_end_warm_preparation",
        "end_to_end_changing_geometry",
    }
    mixed = row["graph"]["swage_mixed"]
    assert mixed["median_us"]["process_values"] == [10.0, 30.0]
    assert mixed["ratio_to_torch_segment_reduce"]["median"] == 1.0
    assert "effective_gb_per_s" in mixed
    assert "effective_gb_per_s" not in row["planning"]["swage_mixed"]
    assert summary["rows"]["compilation"]["compilation"]["swage_total"][
        "median_us"
    ]["process_values"] == [30.0, 90.0]
    assert summary["reference_missing"]["torch"]["compilation"] == [
        "compilation"
    ]
    assert "vadd n=1024" in summary["reference_missing"]["torch_segment_reduce"]
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert printed == {
        "output": str(output_dir / "manifest.json"),
        "summary": str(output_dir / "summary.json"),
    }


# Arguments, output directory, and children.


def test_arguments_default_to_five_processes(campaign):
    """Repeat a configuration in five processes unless told otherwise."""
    arguments = campaign._arguments(
        [
            "--output-dir",
            "out",
            "--allow-shared-gpu-engineering",
            "--harness",
            _FRESH,
            "--",
            "--smoke",
        ]
    )

    assert arguments.repetitions == 5
    assert arguments.reference is None
    assert arguments.summarize is None
    assert arguments.output_dir == pathlib.Path("out")
    assert arguments.harness == pathlib.Path(_FRESH)
    assert arguments.options == ["--smoke"]
    assert campaign._harness_command(arguments) == [_FRESH, "--smoke"]
    assert (
        campaign._arguments(
            [
                "--repetitions",
                "3",
                "--output-dir",
                "out",
                "--exclusive-gpu-allocated",
            ]
        ).repetitions
        == 3
    )


def test_arguments_take_several_references_and_a_summarize_directory(
    campaign,
):
    """Name the candidates of interest; summarize records already taken."""
    run = campaign._arguments(
        [
            "--output-dir",
            "out",
            "--allow-shared-gpu-engineering",
            "--reference",
            "torch",
            "triton_looped_b256_w4",
            "--",
            "--distributions",
            "power-law",
        ]
    )
    again = campaign._arguments(
        ["--summarize", "out", "--reference", "triton_planned_looped_b256_w4"]
    )

    assert run.reference == ["torch", "triton_looped_b256_w4"]
    assert run.options == ["--distributions", "power-law"]
    assert again.summarize == pathlib.Path("out")
    assert again.reference == ["triton_planned_looped_b256_w4"]
    assert again.options == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--output-dir", "out"],
        [
            "--output-dir",
            "out",
            "--exclusive-gpu-allocated",
            "--repetitions",
            "1",
        ],
        ["--exclusive-gpu-allocated", "--", "--seeds", "7"],
        [
            "--output-dir",
            "out",
            "--exclusive-gpu-allocated",
            "--",
            "--output=x",
        ],
        [
            "--output-dir",
            "out",
            "--exclusive-gpu-allocated",
            "--",
            "--samples",
            "3",
        ],
        [
            "--output-dir",
            "out",
            "--exclusive-gpu-allocated",
            "--harness",
            _FRESH,
            "--samples",
            "3",
        ],
        [
            "--output-dir",
            "out",
            "--exclusive-gpu-allocated",
            "--harness",
            _FRESH,
            "--",
            "--output",
            "x.json",
        ],
        ["--summarize", "out", "--", "--smoke"],
        ["--summarize", "out", "--output-dir", "other"],
        ["--summarize", "out", "--repetitions", "3"],
        ["--summarize", "out", "--allow-shared-gpu-engineering"],
    ],
)
def test_arguments_reject_what_cannot_be_repeated(campaign, argv):
    """Require a directory, a GPU mode, and at least two processes."""
    with pytest.raises(SystemExit):
        campaign._arguments(argv)


def test_output_directory_must_stay_out_of_the_worktree(campaign, tmp_path):
    """Keep process records from making the measured worktree dirty."""
    campaign._require_source_neutral_output(_ROOT, tmp_path / "runs" / "law")
    # build/ is ignored by Git, so records there leave the worktree clean.
    campaign._require_source_neutral_output(_ROOT, _ROOT / "build" / "run")
    with pytest.raises(ValueError, match="outside the source worktree"):
        campaign._require_source_neutral_output(
            _ROOT, _ROOT / "benchmarks" / "results" / "run"
        )
    with pytest.raises(ValueError, match="outside the source worktree"):
        campaign._require_source_neutral_output(_ROOT, _ROOT)


def test_output_directory_must_not_hold_earlier_records(campaign, tmp_path):
    """Refuse to mix a new run with the records of an older one."""
    used = tmp_path / "used"
    used.mkdir()
    (used / "process-000.json").write_text("{}")

    with pytest.raises(ValueError, match="not empty"):
        campaign._require_empty_output(used)
    empty = tmp_path / "empty"
    empty.mkdir()
    campaign._require_empty_output(empty)
    campaign._require_empty_output(tmp_path / "new")


def test_children_run_one_after_another_in_fresh_interpreters(
    quiet_machine, monkeypatch, tmp_path
):
    """Start every process from the same command with its own output."""
    campaign = quiet_machine
    calls = []
    records = [_fresh_record(100.0 + index, 10.0) for index in range(3)]
    monkeypatch.setattr(
        campaign.subprocess, "run", _writing_run(campaign, records, calls)
    )
    output_dir = tmp_path / "run"

    campaign.main(
        [
            "--output-dir",
            str(output_dir),
            "--repetitions",
            "3",
            "--allow-shared-gpu-engineering",
            "--harness",
            _FRESH,
            "--",
            "--distributions",
            "power-law",
        ]
    )

    paths = [output_dir / f"process-{index:03d}.json" for index in range(3)]
    assert [command for command, _ in calls] == [
        [
            sys.executable,
            str(_ROOT / _FRESH),
            "--output",
            str(path),
            "--distributions",
            "power-law",
        ]
        for path in paths
    ]
    assert all(options["cwd"] == _ROOT for _, options in calls)
    assert [json.loads(path.read_text()) for path in paths] == records


def test_a_failed_child_stops_the_run(quiet_machine, monkeypatch, tmp_path):
    """Do not summarize a run in which a process failed."""
    campaign = quiet_machine
    calls = []

    def run(command, **options):
        calls.append(command)
        return campaign.subprocess.CompletedProcess(command, 1, "", "failed")

    monkeypatch.setattr(campaign.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.main(
            [
                "--output-dir",
                str(tmp_path / "run"),
                "--allow-shared-gpu-engineering",
                "--harness",
                _FRESH,
            ]
        )
    assert len(calls) == 1
    assert not (tmp_path / "run" / "summary.json").exists()


# The summary.


def test_series_reads_the_fresh_offsets_rows(campaign):
    """Take each candidate's own median and rate from a process record."""
    assert campaign._series(_fresh_record(100.0, 10.0, pad=400.0)) == {
        "power-law": {
            "end_to_end": {
                "swage_mixed": {"median_us": 100.0, "gb_per_s": 10.0},
                "torch": {"median_us": 10.0, "gb_per_s": 100.0},
                "torch_pad_to_max": {"median_us": 400.0, "gb_per_s": 2.5},
            }
        }
    }


def test_series_reads_every_timing_method_of_the_comparison(campaign):
    """Keep the timing methods apart and drop a method that did not run."""
    series = campaign._series(_comparison_record())

    assert set(series) == {"vadd n=1024", "power-law seed=7"}
    assert set(series["vadd n=1024"]) == {"call", "batched_event", "graph"}
    assert series["vadd n=1024"]["graph"]["swage"] == {
        "median_us": 2.0,
        "gb_per_s": 50.0,
    }
    segmented = series["power-law seed=7"]
    assert set(segmented["graph"]) == {"swage_mixed", "torch"}
    assert set(segmented["call"]) == {
        "swage_mixed",
        "torch",
        "triton_looped_b128_w1",
    }


def test_series_takes_medians_from_raw_samples(campaign):
    """Read a median from the samples a record keeps, not its summary."""
    child = make_child(suite="segmented-sum")
    call = child["results"][0]["timings"]["swage_mixed"]["call"]
    call["samples_us"] = [1.0, 3.0]
    call["summary_us"]["median"] = 99.0
    child["results"][0]["planning"] = {"status": "not-run", "reason": "x"}

    series = campaign._series(child)

    row = series["many-tiny seed=7"]
    assert row["call"]["swage_mixed"]["median_us"] == 2.0
    assert "planning" not in row
    assert set(series["compilation"]["compilation"]) >= {
        "swage_total",
        "triton_total",
    }


def test_series_rejects_an_unknown_record(campaign):
    """Fail on a record the driver does not know how to read."""
    with pytest.raises(ValueError, match="frozen-mixed-policy"):
        campaign._series({"benchmark": "frozen-mixed-policy-segmented-sum"})


def test_summary_reports_the_median_and_the_range_across_processes(campaign):
    """Combine per-process medians; pair each ratio inside its process."""
    records = [
        _fresh_record(100.0, 10.0),
        _fresh_record(130.0, 20.0),
        _fresh_record(90.0, 10.0),
        _fresh_record(120.0, 12.0),
        _fresh_record(110.0, 11.0),
    ]

    rows, incomplete, missing = campaign._summarize(
        [campaign._series(record) for record in records], ["torch"]
    )

    assert missing == {}
    swage = rows["power-law"]["end_to_end"]["swage_mixed"]
    assert swage["median_us"] == {
        "process_values": [100.0, 130.0, 90.0, 120.0, 110.0],
        "median": 110.0,
        "min": 90.0,
        "max": 130.0,
    }
    assert swage["ratio_to_torch"] == {
        "process_values": [10.0, 6.5, 9.0, 10.0, 10.0],
        "median": 10.0,
        "min": 6.5,
        "max": 10.0,
    }
    assert swage["effective_gb_per_s"]["median"] == pytest.approx(1000 / 110)
    torch = rows["power-law"]["end_to_end"]["torch"]
    assert torch["median_us"]["median"] == 11.0
    assert torch["ratio_to_torch"]["process_values"] == [1.0] * 5
    assert incomplete == {}


def test_summary_lists_a_candidate_that_some_process_did_not_time(campaign):
    """Never combine a candidate across fewer processes than the run has."""
    records = [
        _fresh_record(100.0, 10.0, pad=400.0),
        _fresh_record(100.0, 10.0),
        _fresh_record(100.0, 10.0, pad=500.0),
    ]

    rows, incomplete, _ = campaign._summarize(
        [campaign._series(record) for record in records], ["torch"]
    )

    assert set(rows["power-law"]["end_to_end"]) == {"swage_mixed", "torch"}
    assert incomplete == {
        "power-law": {"end_to_end": {"torch_pad_to_max": [1, 3]}}
    }


def test_summary_forms_ratios_against_every_named_reference(campaign):
    """Express Swage against a Triton candidate, not only against torch."""
    records = [
        _fresh_record(100.0, 10.0, looped=50.0),
        _fresh_record(120.0, 20.0, looped=40.0),
        _fresh_record(90.0, 10.0, looped=45.0),
    ]

    rows, _, missing = campaign._summarize(
        [campaign._series(record) for record in records],
        ["torch", "triton_looped_b256_w4"],
    )

    swage = rows["power-law"]["end_to_end"]["swage_mixed"]
    assert swage["ratio_to_torch"]["process_values"] == [10.0, 6.0, 9.0]
    assert swage["ratio_to_triton_looped_b256_w4"] == {
        "process_values": [2.0, 3.0, 2.0],
        "median": 2.0,
        "min": 2.0,
        "max": 3.0,
    }
    assert missing == {}


def test_a_reference_missing_from_a_row_is_listed_not_defaulted(campaign):
    """Say where a reference was not timed; form no other ratio instead."""
    with_pad = campaign._series(_fresh_record(100.0, 10.0, pad=400.0))
    with_pad["uniform"] = with_pad.pop("power-law")
    without_pad = campaign._series(_fresh_record(100.0, 10.0))
    series = [{**with_pad, **without_pad}] * 2

    rows, _, missing = campaign._summarize(series, ["torch_pad_to_max"])

    assert (
        rows["uniform"]["end_to_end"]["swage_mixed"][
            "ratio_to_torch_pad_to_max"
        ]["median"]
        == 0.25
    )
    assert set(rows["power-law"]["end_to_end"]["swage_mixed"]) == {
        "median_us",
        "effective_gb_per_s",
    }
    assert missing == {"torch_pad_to_max": {"power-law": ["end_to_end"]}}


def test_a_reference_that_one_process_did_not_time_gives_no_ratio(campaign):
    """Pair a ratio inside every process or not at all."""
    records = [
        _fresh_record(100.0, 10.0, pad=400.0),
        _fresh_record(100.0, 10.0),
        _fresh_record(100.0, 10.0, pad=500.0),
    ]

    rows, incomplete, missing = campaign._summarize(
        [campaign._series(record) for record in records],
        ["torch_pad_to_max"],
    )

    assert (
        "ratio_to_torch_pad_to_max"
        not in (rows["power-law"]["end_to_end"]["swage_mixed"])
    )
    assert missing == {"torch_pad_to_max": {"power-law": ["end_to_end"]}}
    assert incomplete == {
        "power-law": {"end_to_end": {"torch_pad_to_max": [1, 3]}}
    }


def test_a_reference_that_no_row_timed_is_an_error(campaign):
    """Refuse a reference name that matches nothing in the records."""
    series = campaign._series(_fresh_record(100.0, 10.0, looped=50.0))

    campaign._require_references(series, ["torch", "triton_looped_b256_w4"])
    with pytest.raises(
        ValueError, match="triton_looped_b256_w8.*swage_mixed, torch"
    ):
        campaign._require_references(series, ["torch", "triton_looped_b256_w8"])


def test_default_references_are_the_timed_pytorch_candidates(campaign):
    """Take ratios against torch where it runs and torch_segment_reduce."""
    assert campaign._default_references(
        campaign._series(_fresh_record(1.0, 1.0))
    ) == ["torch"]
    assert campaign._default_references(
        campaign._series(make_child(suite="all"))
    ) == ["torch", "torch_segment_reduce"]


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("source", "revision"), "def456", "revision"),
        (
            ("provenance", "native_sha256"),
            {"/build/lib.so": "c" * 64},
            "native_sha256",
        ),
        (("provenance", "loaded_ptx"), [], "loaded_ptx"),
        (("provenance", "gpu_uuid"), "GPU-2", "gpu_uuid"),
        (("provenance", "pytorch"), "2.13.0", "pytorch"),
        (("benchmark",), "swage-triton-comparison", "benchmark"),
    ],
)
def test_processes_must_have_measured_the_same_code(
    campaign, path, value, field
):
    """Refuse to combine records of different binaries or machines."""
    records = [_fresh_record(100.0, 10.0) for _ in range(3)]
    changed = copy.deepcopy(records[2])
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    records[2] = changed

    with pytest.raises(ValueError, match=f"process 3 .*{field}"):
        campaign._code_identity(records)
    assert campaign._code_identity(records[:2])["revision"] == "abc123"


def test_main_writes_the_process_records_and_one_summary(
    quiet_machine, tmp_path, monkeypatch, capsys
):
    """Run the children, then write the summary beside their records."""
    campaign = quiet_machine
    calls = []
    records = [
        _fresh_record(100.0, 10.0),
        _fresh_record(120.0, 10.0, seen=True),
        _fresh_record(110.0, 10.0),
    ]
    monkeypatch.setattr(
        campaign.subprocess, "run", _writing_run(campaign, records, calls)
    )
    output_dir = tmp_path / "run"

    campaign.main(
        [
            "--repetitions",
            "3",
            "--output-dir",
            str(output_dir),
            "--allow-shared-gpu-engineering",
            "--harness",
            _FRESH,
            "--",
            "--distributions",
            "power-law",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "process-000.json",
        "process-001.json",
        "process-002.json",
        "summary.json",
    ]
    assert summary["benchmark"] == "fresh-offsets-segmented-sum"
    assert summary["processes"] == 3
    assert summary["references"] == ["torch"]
    assert summary["reference_missing"] == {}
    assert summary["smoke"] is False
    assert summary["command"] == [_FRESH, "--distributions", "power-law"]
    assert summary["code"]["revision"] == "abc123"
    assert summary["code"]["native_sha256"] == {"/build/lib.so": "a" * 64}
    assert [entry["record"] for entry in summary["process_records"]] == [
        "process-000.json",
        "process-001.json",
        "process-002.json",
    ]
    assert [
        entry["other_compute_process_seen"]
        for entry in summary["process_records"]
    ] == [False, True, False]
    assert (
        summary["process_records"][0]["sha256"]
        == hashlib.sha256(
            (output_dir / "process-000.json").read_bytes()
        ).hexdigest()
    )
    assert summary["process_records"][0]["gpu_state_before"] == {
        "gpu": {"temperature.gpu": "50"}
    }
    assert summary["process_records"][0]["cpu_frequency_before"] == {
        "governors": {"powersave": 24}
    }
    assert [
        entry["cpu_governor_unchanged"] for entry in summary["process_records"]
    ] == [True] * 3
    assert summary["rows"]["power-law"]["end_to_end"]["swage_mixed"][
        "median_us"
    ] == {
        "process_values": [100.0, 120.0, 110.0],
        "median": 110.0,
        "min": 100.0,
        "max": 120.0,
    }
    assert "selected" in summary["statistics"]
    assert str(output_dir / "summary.json") in capsys.readouterr().out


def test_main_marks_a_summary_of_smoke_records(
    quiet_machine, tmp_path, monkeypatch
):
    """Carry the smoke label of the process records into the summary."""
    campaign = quiet_machine
    records = [_fresh_record(100.0, 10.0) for _ in range(2)]
    for record in records:
        record["smoke"] = True
    monkeypatch.setattr(
        campaign.subprocess, "run", _writing_run(campaign, records, [])
    )

    campaign.main(
        [
            "--repetitions",
            "2",
            "--output-dir",
            str(tmp_path / "run"),
            "--allow-shared-gpu-engineering",
            "--harness",
            "benchmarks/benchmark_fresh_offsets.py",
        ]
    )

    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["smoke"] is True


def test_main_stops_after_one_process_when_a_reference_is_unknown(
    quiet_machine, tmp_path, monkeypatch
):
    """Fail early instead of running five processes for no ratio."""
    campaign = quiet_machine
    calls = []
    records = [_fresh_record(100.0, 10.0) for _ in range(3)]
    monkeypatch.setattr(
        campaign.subprocess, "run", _writing_run(campaign, records, calls)
    )
    output_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="reference triton_looped_b256_w4"):
        campaign.main(
            [
                "--repetitions",
                "3",
                "--output-dir",
                str(output_dir),
                "--allow-shared-gpu-engineering",
                "--harness",
                _FRESH,
                "--reference",
                "torch",
                "triton_looped_b256_w4",
            ]
        )

    assert len(calls) == 1
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "failure-observations.json",
        "process-000.json",
    ]


def _write_records(directory, records):
    """Write process records as the driver of earlier revisions left them."""
    directory.mkdir(parents=True)
    for index, record in enumerate(records, start=1):
        (directory / f"process-{index}.json").write_text(json.dumps(record))


def test_summarize_reads_existing_records_against_another_reference(
    campaign, tmp_path, monkeypatch, capsys
):
    """Form new ratios from a finished run without running it again."""
    records = [
        _fresh_record(100.0 + index, 10.0, looped=50.0) for index in range(11)
    ]
    run = tmp_path / "run"
    _write_records(run, records)
    (run / "summary.json").write_text("{}")

    def no_run(command, **options):
        raise AssertionError("summarizing must not start a process")

    monkeypatch.setattr(campaign.subprocess, "run", no_run)

    campaign.main(
        ["--summarize", str(run), "--reference", "triton_looped_b256_w4"]
    )

    output = run / "summary-triton_looped_b256_w4.json"
    summary = json.loads(output.read_text())
    assert str(output) in capsys.readouterr().out
    assert (run / "summary.json").read_text() == "{}"
    assert summary["processes"] == 11
    assert summary["command"] is None
    assert summary["references"] == ["triton_looped_b256_w4"]
    # Process 10 and 11 come after process 9, not after process 1.
    assert [entry["record"] for entry in summary["process_records"]] == [
        f"process-{index}.json" for index in range(1, 12)
    ]
    swage = summary["rows"]["power-law"]["end_to_end"]["swage_mixed"]
    assert swage["median_us"]["process_values"] == [
        100.0 + index for index in range(11)
    ]
    assert swage["ratio_to_triton_looped_b256_w4"]["min"] == 2.0
    assert "ratio_to_torch" not in swage


def test_summarize_does_not_overwrite_or_invent(campaign, tmp_path):
    """Refuse an existing summary, an empty directory, a wrong reference."""
    run = tmp_path / "run"
    _write_records(run, [_fresh_record(100.0, 10.0) for _ in range(2)])
    arguments = ["--summarize", str(run), "--reference", "torch"]

    campaign.main(arguments)
    with pytest.raises(FileExistsError, match="summary-torch.json"):
        campaign.main(arguments)
    with pytest.raises(FileExistsError, match="summary-torch.json"):
        campaign.main(["--summarize", str(run)])
    with pytest.raises(ValueError, match="reference triton_looped_b256_w4"):
        campaign.main(
            ["--summarize", str(run), "--reference", "triton_looped_b256_w4"]
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no process records"):
        campaign.main(["--summarize", str(empty)])


@pytest.mark.parametrize("run", ["comparison", "fresh-2048-all"])
def test_summarize_reproduces_the_committed_summaries(campaign, tmp_path, run):
    """Summarize the committed process records into the committed summary.

    The records of the 453c56e campaign stay readable: decompressed, they
    give back every row, rate, and ratio of their committed summary.json.
    """
    committed = json.loads((_RECORD / run / "summary.json").read_text())
    directory = tmp_path / run
    directory.mkdir()
    for packed in sorted((_RECORD / run).glob("process-*.json.xz")):
        (directory / packed.name.removesuffix(".xz")).write_bytes(
            lzma.decompress(packed.read_bytes())
        )

    campaign.main(
        ["--summarize", str(directory), "--reference", *committed["references"]]
    )

    output = directory / f"summary-{'-'.join(committed['references'])}.json"
    summary = json.loads(output.read_text())
    for key in (
        "benchmark",
        "code",
        "incomplete",
        "process_records",
        "processes",
        "reference_missing",
        "references",
        "rows",
        "smoke",
        "statistics",
    ):
        assert summary[key] == committed[key], key
