# tests/python/test_benchmark_processes.py
"""Tests for the driver that repeats a benchmark in independent processes."""

import copy
import hashlib
import importlib
import json
import pathlib
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def processes(monkeypatch):
    """Import the driver as its script entry point does."""
    monkeypatch.syspath_prepend(str(_ROOT / "benchmarks"))
    return importlib.import_module("benchmark_processes")


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
    """Return a comparison record with a vadd row and a segmented row."""

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


def _writing_run(records, calls):
    """Return a fake ``subprocess.run`` that writes one record per child."""
    remaining = iter(records)

    def run(command, **options):
        calls.append((command, options))
        output = pathlib.Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(next(remaining)))
        return subprocess.CompletedProcess(command, 0)

    return run


def test_arguments_default_to_five_processes(processes):
    """Repeat a configuration in five processes unless told otherwise."""
    arguments = processes._arguments(
        ["--output-dir", "out", "--", "bench.py", "--smoke"]
    )

    assert arguments.processes == 5
    assert arguments.reference == ["torch"]
    assert arguments.summarize is None
    assert arguments.output_dir == pathlib.Path("out")
    assert arguments.command == ["bench.py", "--smoke"]
    assert (
        processes._arguments(
            ["--processes", "3", "--output-dir", "out", "bench.py"]
        ).processes
        == 3
    )


def test_arguments_take_several_references_and_a_summarize_directory(
    processes,
):
    """Name the candidates of interest; summarize records already taken."""
    run = processes._arguments(
        [
            "--output-dir",
            "out",
            "--reference",
            "torch",
            "triton_looped_b256_w4",
            "--",
            "bench.py",
        ]
    )
    again = processes._arguments(
        ["--summarize", "out", "--reference", "triton_planned_looped_b256_w4"]
    )

    assert run.reference == ["torch", "triton_looped_b256_w4"]
    assert run.command == ["bench.py"]
    assert again.summarize == pathlib.Path("out")
    assert again.reference == ["triton_planned_looped_b256_w4"]
    assert again.command == []


@pytest.mark.parametrize(
    "argv",
    [
        ["--output-dir", "out"],
        ["--output-dir", "out", "--processes", "1", "bench.py"],
        ["--output-dir", "out", "bench.py", "--output", "x.json"],
        ["bench.py"],
        ["--summarize", "out", "bench.py"],
        ["--summarize", "out", "--output-dir", "other"],
        ["--summarize", "out", "--processes", "3"],
    ],
)
def test_arguments_reject_what_cannot_be_repeated(processes, argv):
    """Require a command, a directory, and at least two processes."""
    with pytest.raises(SystemExit):
        processes._arguments(argv)


def test_output_directory_must_stay_out_of_the_checkout(processes, tmp_path):
    """Keep process records from making the measured worktree dirty."""
    checkout = tmp_path / "swage"
    (checkout / "benchmarks" / "results").mkdir(parents=True)
    outside = tmp_path / "runs" / "power-law"

    processes._check_output_dir(checkout, outside)
    with pytest.raises(ValueError, match="outside the checkout"):
        processes._check_output_dir(
            checkout, checkout / "benchmarks" / "results" / "run"
        )
    with pytest.raises(ValueError, match="outside the checkout"):
        processes._check_output_dir(checkout, checkout)


def test_output_directory_must_not_hold_earlier_records(processes, tmp_path):
    """Refuse to mix a new run with the records of an older one."""
    used = tmp_path / "used"
    used.mkdir()
    (used / "process-1.json").write_text("{}")

    with pytest.raises(ValueError, match="not empty"):
        processes._check_output_dir(tmp_path / "swage", used)
    empty = tmp_path / "empty"
    empty.mkdir()
    processes._check_output_dir(tmp_path / "swage", empty)


def test_children_run_one_after_another_in_fresh_interpreters(
    processes, tmp_path
):
    """Start every process from the same command with its own output."""
    calls = []
    records = [_fresh_record(100.0 + index, 10.0) for index in range(3)]

    paths = processes._run_processes(
        ["bench.py", "--distributions", "power-law"],
        tmp_path,
        3,
        run=_writing_run(records, calls),
    )

    assert paths == [tmp_path / f"process-{index}.json" for index in (1, 2, 3)]
    assert [command for command, _ in calls] == [
        [
            sys.executable,
            "bench.py",
            "--distributions",
            "power-law",
            "--output",
            str(path),
        ]
        for path in paths
    ]
    assert all(options == {"check": True} for _, options in calls)
    assert [json.loads(path.read_text()) for path in paths] == records


def test_a_failed_child_stops_the_run(processes, tmp_path):
    """Do not summarize a run in which a process failed."""

    def run(command, **options):
        raise subprocess.CalledProcessError(1, command)

    with pytest.raises(subprocess.CalledProcessError):
        processes._run_processes(["bench.py"], tmp_path, 2, run=run)


def test_series_reads_the_fresh_offsets_rows(processes):
    """Take each candidate's own median and rate from a process record."""
    assert processes._series(_fresh_record(100.0, 10.0, pad=400.0)) == {
        "power-law": {
            "end_to_end": {
                "swage_mixed": {"median_us": 100.0, "gb_per_s": 10.0},
                "torch": {"median_us": 10.0, "gb_per_s": 100.0},
                "torch_pad_to_max": {"median_us": 400.0, "gb_per_s": 2.5},
            }
        }
    }


def test_series_reads_every_timing_method_of_the_comparison(processes):
    """Keep the timing methods apart and drop a method that did not run."""
    series = processes._series(_comparison_record())

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


def test_series_rejects_an_unknown_record(processes):
    """Fail on a record the driver does not know how to read."""
    with pytest.raises(ValueError, match="frozen-mixed-policy"):
        processes._series({"benchmark": "frozen-mixed-policy-segmented-sum"})


def test_summary_reports_the_median_and_the_range_across_processes(processes):
    """Combine per-process medians; pair each ratio inside its process."""
    records = [
        _fresh_record(100.0, 10.0),
        _fresh_record(130.0, 20.0),
        _fresh_record(90.0, 10.0),
        _fresh_record(120.0, 12.0),
        _fresh_record(110.0, 11.0),
    ]

    rows, incomplete, missing = processes._summarize(
        [processes._series(record) for record in records], ["torch"]
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


def test_summary_lists_a_candidate_that_some_process_did_not_time(processes):
    """Never combine a candidate across fewer processes than the run has."""
    records = [
        _fresh_record(100.0, 10.0, pad=400.0),
        _fresh_record(100.0, 10.0),
        _fresh_record(100.0, 10.0, pad=500.0),
    ]

    rows, incomplete, _ = processes._summarize(
        [processes._series(record) for record in records], ["torch"]
    )

    assert set(rows["power-law"]["end_to_end"]) == {"swage_mixed", "torch"}
    assert incomplete == {
        "power-law": {"end_to_end": {"torch_pad_to_max": [1, 3]}}
    }


def test_summary_forms_ratios_against_every_named_reference(processes):
    """Express Swage against a Triton candidate, not only against torch."""
    records = [
        _fresh_record(100.0, 10.0, looped=50.0),
        _fresh_record(120.0, 20.0, looped=40.0),
        _fresh_record(90.0, 10.0, looped=45.0),
    ]

    rows, _, missing = processes._summarize(
        [processes._series(record) for record in records],
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


def test_a_reference_missing_from_a_row_is_listed_not_defaulted(processes):
    """Say where a reference was not timed; form no other ratio instead."""
    with_pad = processes._series(_fresh_record(100.0, 10.0, pad=400.0))
    with_pad["uniform"] = with_pad.pop("power-law")
    without_pad = processes._series(_fresh_record(100.0, 10.0))
    series = [{**with_pad, **without_pad}] * 2

    rows, _, missing = processes._summarize(series, ["torch_pad_to_max"])

    assert rows["uniform"]["end_to_end"]["swage_mixed"][
        "ratio_to_torch_pad_to_max"
    ]["median"] == 0.25
    assert set(rows["power-law"]["end_to_end"]["swage_mixed"]) == {
        "median_us",
        "effective_gb_per_s",
    }
    assert missing == {"torch_pad_to_max": {"power-law": ["end_to_end"]}}


def test_a_reference_that_one_process_did_not_time_gives_no_ratio(
    processes,
):
    """Pair a ratio inside every process or not at all."""
    records = [
        _fresh_record(100.0, 10.0, pad=400.0),
        _fresh_record(100.0, 10.0),
        _fresh_record(100.0, 10.0, pad=500.0),
    ]

    rows, incomplete, missing = processes._summarize(
        [processes._series(record) for record in records],
        ["torch_pad_to_max"],
    )

    assert "ratio_to_torch_pad_to_max" not in (
        rows["power-law"]["end_to_end"]["swage_mixed"]
    )
    assert missing == {"torch_pad_to_max": {"power-law": ["end_to_end"]}}
    assert incomplete == {
        "power-law": {"end_to_end": {"torch_pad_to_max": [1, 3]}}
    }


def test_a_reference_that_no_row_timed_is_an_error(processes):
    """Refuse a reference name that matches nothing in the records."""
    series = processes._series(_fresh_record(100.0, 10.0, looped=50.0))

    processes._require_references(series, ["torch", "triton_looped_b256_w4"])
    with pytest.raises(
        ValueError, match="triton_looped_b256_w8.*swage_mixed, torch"
    ):
        processes._require_references(
            series, ["torch", "triton_looped_b256_w8"]
        )


@pytest.mark.parametrize(
    ("path", "value", "field"),
    [
        (("source", "revision"), "def456", "revision"),
        (("provenance", "native_sha256"), {"/build/lib.so": "c" * 64},
         "native_sha256"),
        (("provenance", "loaded_ptx"), [], "loaded_ptx"),
        (("provenance", "gpu_uuid"), "GPU-2", "gpu_uuid"),
        (("provenance", "pytorch"), "2.13.0", "pytorch"),
        (("benchmark",), "swage-triton-comparison", "benchmark"),
    ],
)
def test_processes_must_have_measured_the_same_code(
    processes, path, value, field
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
        processes._code_identity(records)
    assert processes._code_identity(records[:2])["revision"] == "abc123"


def test_main_writes_the_process_records_and_one_summary(
    processes, tmp_path, monkeypatch, capsys
):
    """Run the children, then write the summary beside their records."""
    calls = []
    records = [
        _fresh_record(100.0, 10.0),
        _fresh_record(120.0, 10.0, seen=True),
        _fresh_record(110.0, 10.0),
    ]
    monkeypatch.setattr(
        processes.subprocess, "run", _writing_run(records, calls)
    )
    output_dir = tmp_path / "run"

    processes.main(
        [
            "--processes",
            "3",
            "--output-dir",
            str(output_dir),
            "--",
            "benchmarks/benchmark_fresh_offsets.py",
            "--distributions",
            "power-law",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "process-1.json",
        "process-2.json",
        "process-3.json",
        "summary.json",
    ]
    assert summary["benchmark"] == "fresh-offsets-segmented-sum"
    assert summary["processes"] == 3
    assert summary["references"] == ["torch"]
    assert summary["reference_missing"] == {}
    assert summary["smoke"] is False
    assert summary["command"] == [
        "benchmarks/benchmark_fresh_offsets.py",
        "--distributions",
        "power-law",
    ]
    assert summary["code"]["revision"] == "abc123"
    assert summary["code"]["native_sha256"] == {"/build/lib.so": "a" * 64}
    assert [entry["record"] for entry in summary["process_records"]] == [
        "process-1.json",
        "process-2.json",
        "process-3.json",
    ]
    assert [
        entry["other_compute_process_seen"]
        for entry in summary["process_records"]
    ] == [False, True, False]
    assert summary["process_records"][0]["sha256"] == hashlib.sha256(
        (output_dir / "process-1.json").read_bytes()
    ).hexdigest()
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
    processes, tmp_path, monkeypatch
):
    """Carry the smoke label of the process records into the summary."""
    records = [_fresh_record(100.0, 10.0) for _ in range(2)]
    for record in records:
        record["smoke"] = True
    monkeypatch.setattr(processes.subprocess, "run", _writing_run(records, []))

    processes.main(
        ["--processes", "2", "--output-dir", str(tmp_path / "run"), "b.py"]
    )

    summary = json.loads((tmp_path / "run" / "summary.json").read_text())
    assert summary["smoke"] is True


def test_main_stops_after_one_process_when_a_reference_is_unknown(
    processes, tmp_path, monkeypatch
):
    """Fail early instead of running five processes for no ratio."""
    calls = []
    records = [_fresh_record(100.0, 10.0) for _ in range(3)]
    monkeypatch.setattr(
        processes.subprocess, "run", _writing_run(records, calls)
    )
    output_dir = tmp_path / "run"

    with pytest.raises(ValueError, match="reference triton_looped_b256_w4"):
        processes.main(
            [
                "--processes",
                "3",
                "--output-dir",
                str(output_dir),
                "--reference",
                "torch",
                "triton_looped_b256_w4",
                "--",
                "b.py",
            ]
        )

    assert len(calls) == 1
    assert [path.name for path in output_dir.iterdir()] == ["process-1.json"]


def _write_records(directory, records):
    """Write process records as a finished run leaves them."""
    directory.mkdir(parents=True)
    for index, record in enumerate(records, start=1):
        (directory / f"process-{index}.json").write_text(json.dumps(record))


def test_summarize_reads_existing_records_against_another_reference(
    processes, tmp_path, monkeypatch, capsys
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

    monkeypatch.setattr(processes.subprocess, "run", no_run)

    processes.main(
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


def test_summarize_does_not_overwrite_or_invent(processes, tmp_path):
    """Refuse an existing summary, an empty directory, a wrong reference."""
    run = tmp_path / "run"
    _write_records(run, [_fresh_record(100.0, 10.0) for _ in range(2)])
    arguments = ["--summarize", str(run), "--reference", "torch"]

    processes.main(arguments)
    with pytest.raises(FileExistsError, match="summary-torch.json"):
        processes.main(arguments)
    with pytest.raises(ValueError, match="reference triton_looped_b256_w4"):
        processes.main(
            ["--summarize", str(run), "--reference", "triton_looped_b256_w4"]
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no process records"):
        processes.main(["--summarize", str(empty)])
