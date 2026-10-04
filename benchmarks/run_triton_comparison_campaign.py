# benchmarks/run_triton_comparison_campaign.py
"""Run a benchmark harness in independent processes and summarize them.

Samples taken inside one process share its allocator state, its compiled
kernels, and whatever the machine was doing at the time, so their spread
understates how far a median moves from one run to the next. This driver
starts the same harness command in several fresh interpreters, one after
another, each with its own empty Swage and Triton cache directories, and
records nvidia-smi observations at every process boundary.

Usage, from a clean checkout with PYTHONPATH=python:build/python_packages:

    python benchmarks/run_triton_comparison_campaign.py --output-dir DIR
        --exclusive-gpu-allocated --archival-source [--repetitions 5]
        [--suite all] [--samples 100] [--warmups 25] [-- OPTIONS]

This runs benchmarks/benchmark_triton_comparison.py with the suite, sample,
and warmup controls, and forwards the options after the separator, such as
--distributions or --candidates. DIR then holds process-NNN.json, one record
per process, manifest.json with the validated process-level aggregates, and
summary.json. --harness names another harness, such as
benchmarks/benchmark_fresh_offsets.py; every option of that harness follows
the separator, and DIR holds its records and summary.json. The driver adds
--output to every process itself.

summary.json holds, for every row, timing method, and candidate, the
per-process medians with their median, minimum, and maximum, and the same
for the ratio to each reference candidate named with --reference. A
reference that no row timed is an error, raised after the first process.
Records that a run left behind can be summarized again against other
references, without running anything:

    python benchmarks/run_triton_comparison_campaign.py --summarize DIR
        --reference triton_looped_b256_w4

This writes DIR/summary-<references>.json and never replaces a summary. The
driver reads the records of benchmark_triton_comparison.py and
benchmark_fresh_offsets.py, also those of earlier revisions named
process-N.json. It imports neither PyTorch nor Triton.
"""

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from benchmark_campaign import (
    SCHEMA_VERSION,
    aggregate_children,
    compute_process_reasons,
    load_campaign,
    load_unique_json,
    validate_child,
)

_REPETITIONS = 5
_COMPARISON = pathlib.Path("benchmarks/benchmark_triton_comparison.py")
# The options of the comparison harness that the driver sets itself.
_DRIVER_OPTIONS = ("--output", "--suite", "--samples", "--warmups")
# The PyTorch candidates that ratios are taken against when no reference is
# named, each where the records time it.
_DEFAULT_REFERENCES = ("torch", "torch_segment_reduce")
_PROCESS_FIELDS = (
    "other_compute_process_seen",
    "gpu_state_before",
    "gpu_state_after",
    "cpu_frequency_before",
    "cpu_frequency_after",
    "cpu_governor_unchanged",
)
_CODE_FIELDS = (
    "gpu",
    "gpu_uuid",
    "cpu_model",
    "pytorch",
    "triton",
    "native_sha256",
    "loaded_ptx",
)


def _arguments(argv=None):
    """Parse the campaign controls, or the directory to summarize again.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        The parsed arguments; ``options`` holds the harness options that
        follow the separator.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help=(
            "New or empty directory, outside the checkout or ignored by Git, "
            "for a run."
        ),
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        help=(
            "Independent processes to run, one after another. The default "
            f"is {_REPETITIONS}."
        ),
    )
    parser.add_argument(
        "--harness",
        type=pathlib.Path,
        default=_COMPARISON,
        help=(
            "Harness script to repeat, relative to the checkout. The "
            f"default is {_COMPARISON}."
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("all", "vadd", "segmented-sum"),
        help="Comparison suite; the default is all.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Comparison samples per process; the default is 100.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        help="Comparison warmups per process; the default is 25.",
    )
    gpu_mode = parser.add_mutually_exclusive_group()
    gpu_mode.add_argument(
        "--exclusive-gpu-allocated",
        action="store_true",
        help=(
            "assert exclusive GPU allocation and fail closed when active "
            "compute-process telemetry is unavailable or non-empty"
        ),
    )
    gpu_mode.add_argument(
        "--allow-shared-gpu-engineering",
        action="store_true",
        help=(
            "permit unavailable or non-empty compute-process telemetry; "
            "the resulting manifest is explicitly ineligible for archival use"
        ),
    )
    parser.add_argument(
        "--archival-source",
        action="store_true",
        help=(
            "assert HEAD is the durable final source revision "
            "accompanying the archived records"
        ),
    )
    parser.add_argument(
        "--summarize",
        type=pathlib.Path,
        metavar="DIR",
        help=(
            "Summarize the process records of a finished run in DIR "
            "instead of running a harness."
        ),
    )
    parser.add_argument(
        "--reference",
        nargs="+",
        metavar="NAME",
        help=(
            "Candidates that ratios are taken against, each by its full "
            "name, such as triton_looped_b256_w4. The default is torch and "
            "torch_segment_reduce, each where the records time it. End the "
            "list with -- before harness options."
        ),
    )
    parser.add_argument(
        "options",
        nargs=argparse.REMAINDER,
        help="Harness options after --, without --output.",
    )
    arguments = parser.parse_args(argv)
    if arguments.options[:1] == ["--"]:
        del arguments.options[0]
    comparison = arguments.harness == _COMPARISON
    if arguments.summarize is not None:
        if (
            arguments.options
            or arguments.output_dir is not None
            or arguments.repetitions is not None
            or arguments.harness != _COMPARISON
            or arguments.suite is not None
            or arguments.samples is not None
            or arguments.warmups is not None
            or arguments.exclusive_gpu_allocated
            or arguments.allow_shared_gpu_engineering
            or arguments.archival_source
        ):
            parser.error(
                "--summarize reads the records of a finished run; it takes "
                "only --reference"
            )
        return arguments
    if arguments.output_dir is None:
        parser.error("--output-dir or --summarize is required")
    if not (
        arguments.exclusive_gpu_allocated
        or arguments.allow_shared_gpu_engineering
    ):
        parser.error(
            "one of --exclusive-gpu-allocated and "
            "--allow-shared-gpu-engineering is required"
        )
    if arguments.archival_source and arguments.allow_shared_gpu_engineering:
        parser.error(
            "--archival-source cannot be combined with "
            "--allow-shared-gpu-engineering"
        )
    if arguments.repetitions is None:
        arguments.repetitions = _REPETITIONS
    if arguments.repetitions < 2:
        parser.error("repetitions must be at least 2; one has no spread")
    reserved = _DRIVER_OPTIONS if comparison else ("--output",)
    for option in arguments.options:
        if option.split("=", 1)[0] in reserved:
            parser.error(f"the driver passes {option} to every process itself")
    if comparison:
        arguments.suite = arguments.suite or "all"
        arguments.samples = (
            100 if arguments.samples is None else (arguments.samples)
        )
        arguments.warmups = (
            25 if arguments.warmups is None else (arguments.warmups)
        )
        if arguments.samples <= 0 or arguments.warmups < 0:
            parser.error("samples must be positive and warmups nonnegative")
    elif (
        arguments.suite is not None
        or arguments.samples is not None
        or arguments.warmups is not None
    ):
        parser.error(
            "--suite, --samples, and --warmups control the comparison "
            "harness; pass the options of another harness after --"
        )
    return arguments


def _harness_command(arguments):
    """Return the harness and its options, without the interpreter and output.

    This is the command that every process runs and that the summary
    records; the driver appends --output.
    """
    command = [arguments.harness.as_posix()]
    if arguments.harness == _COMPARISON:
        command += [
            "--suite",
            arguments.suite,
            "--samples",
            str(arguments.samples),
            "--warmups",
            str(arguments.warmups),
        ]
    return [*command, *arguments.options]


def _source_metadata(root: pathlib.Path) -> dict[str, object]:
    """Return source metadata, rejecting a dirty input worktree."""
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if dirty:
        raise RuntimeError(
            "campaign requires a clean source worktree; dirty entries: "
            + ", ".join(dirty)
        )
    return {"revision": revision, "worktree_clean": True, "dirty": []}


def _require_source_neutral_output(
    root: pathlib.Path, output_dir: pathlib.Path
) -> None:
    """Reject in-worktree output paths unless Git explicitly ignores them."""
    resolved_root = root.resolve()
    resolved_output = output_dir.resolve()
    try:
        relative_output = resolved_output.relative_to(resolved_root)
    except ValueError:
        return
    ignored = subprocess.run(
        [
            "git",
            "check-ignore",
            "--quiet",
            "--no-index",
            str(relative_output),
        ],
        cwd=root,
        check=False,
    )
    if ignored.returncode != 0:
        raise ValueError(
            "output directory must be outside the source worktree or "
            "matched by a Git ignore rule"
        )


def _sha256(path: pathlib.Path) -> str:
    """Hash one preserved child JSON file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _telemetry_value(value: str, *, integral: bool):
    """Parse one nvidia-smi value without inventing unavailable data."""
    stripped = value.strip()
    if stripped in {"", "N/A", "[N/A]"}:
        return None
    try:
        numeric = float(stripped)
    except ValueError:
        return None
    return int(numeric) if integral else numeric


def _nvidia_query(
    executable: str, query_kind: str, fields: str
) -> tuple[list[list[str]] | None, str | None]:
    """Run one nvidia-smi CSV query without discarding its failure reason."""
    command = [
        executable,
        f"--query-{query_kind}={fields}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        return None, f"nvidia-smi execution failed: {error}"
    if completed.returncode != 0:
        detail = completed.stderr.strip()
        reason = f"nvidia-smi exited with status {completed.returncode}"
        if detail:
            reason = f"{reason}: {detail}"
        return None, reason
    return [
        [column.strip() for column in row]
        for row in csv.reader(completed.stdout.splitlines())
        if row
    ], None


def _nvidia_telemetry() -> dict[str, object]:
    """Capture NVIDIA device and active compute-process observations."""
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"available": False, "reason": "nvidia-smi not found"}

    gpu_fields = (
        "index,uuid,clocks.current.sm,clocks.current.memory,"
        "power.draw,temperature.gpu"
    )
    gpu_rows, gpu_error = _nvidia_query(executable, "gpu", gpu_fields)
    if gpu_error is not None:
        return {"available": False, "reason": gpu_error}
    gpus = []
    for columns in gpu_rows or []:
        if len(columns) != 6:
            return {
                "available": False,
                "reason": "nvidia-smi returned an unexpected GPU row schema",
            }
        gpus.append(
            {
                "index": _telemetry_value(columns[0], integral=True),
                "uuid": columns[1] or None,
                "sm_clock_mhz": _telemetry_value(columns[2], integral=True),
                "memory_clock_mhz": _telemetry_value(columns[3], integral=True),
                "power_draw_watts": _telemetry_value(
                    columns[4], integral=False
                ),
                "temperature_celsius": _telemetry_value(
                    columns[5], integral=False
                ),
            }
        )
    if not gpus:
        return {
            "available": False,
            "reason": "nvidia-smi returned no GPU rows",
        }

    process_fields = "gpu_uuid,pid,process_name,used_gpu_memory"
    process_rows, process_error = _nvidia_query(
        executable, "compute-apps", process_fields
    )
    if process_error is not None:
        compute_processes = {
            "available": False,
            "reason": process_error,
        }
    else:
        processes = []
        for columns in process_rows or []:
            if len(columns) != 4:
                compute_processes = {
                    "available": False,
                    "reason": (
                        "nvidia-smi returned an unexpected "
                        "compute-process row schema"
                    ),
                }
                break
            pid = _telemetry_value(columns[1], integral=True)
            if not columns[0] or pid is None or not columns[2]:
                compute_processes = {
                    "available": False,
                    "reason": (
                        "nvidia-smi returned an incomplete compute-process row"
                    ),
                }
                break
            processes.append(
                {
                    "gpu_uuid": columns[0],
                    "pid": pid,
                    "process_name": columns[2],
                    "used_memory_mib": _telemetry_value(
                        columns[3], integral=True
                    ),
                }
            )
        else:
            compute_processes = {
                "available": True,
                "fields": {"used_memory_mib": "MiB"},
                "processes": processes,
            }

    return {
        "available": True,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "fields": {
            "sm_clock_mhz": "MHz",
            "memory_clock_mhz": "MHz",
            "power_draw_watts": "W",
            "temperature_celsius": "degrees Celsius",
        },
        "gpus": gpus,
        "compute_processes": compute_processes,
    }


def _apply_compute_process_policy(
    telemetry: dict[str, object],
    boundary: str,
    *,
    exclusive_gpu_allocated: bool,
) -> list[str]:
    """Fail closed for asserted-exclusive runs and return observed issues."""
    reasons = compute_process_reasons(telemetry, boundary)
    if reasons and exclusive_gpu_allocated:
        raise RuntimeError("; ".join(reasons))
    return reasons


def _external_temporary_parent(
    root: pathlib.Path, output_dir: pathlib.Path
) -> pathlib.Path:
    """Choose temporary storage outside both source and campaign output."""
    excluded = (root.resolve(), output_dir.resolve())
    for candidate in (
        pathlib.Path(tempfile.gettempdir()),
        pathlib.Path("/tmp"),
        pathlib.Path("/var/tmp"),
    ):
        resolved = candidate.resolve()
        if resolved.is_dir() and not any(
            resolved.is_relative_to(path) for path in excluded
        ):
            return resolved
    raise RuntimeError("no external temporary directory is available")


def _write_failure_observations(
    root: pathlib.Path,
    output_dir: pathlib.Path,
    observations: list[dict[str, object]],
    error: BaseException,
) -> None:
    """Retain failed-run output and boundary observations outside source."""
    destination = output_dir
    if destination.is_relative_to(root.resolve()):
        destination = pathlib.Path(
            tempfile.mkdtemp(
                prefix="swage-comparison-failure-",
                dir=_external_temporary_parent(root, output_dir),
            )
        )
    path = destination / "failure-observations.json"
    record = {
        "benchmark": "swage-triton-comparison-process-campaign-failure",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "error": {"type": type(error).__name__, "message": str(error)},
        "process_observations": observations,
    }
    with path.open("x") as output:
        output.write(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps({"failure_observations": str(path)}, sort_keys=True),
        file=sys.stderr,
    )


def _record_paths(directory):
    """Return the process records of a finished run, in run order.

    Records are named process-NNN.json, from zero, by this driver, and
    process-N.json, from one, by the driver of earlier revisions.

    Raises:
        ValueError: If the directory holds no process record.
    """
    paths = sorted(
        directory.glob("process-*.json"),
        key=lambda path: int(path.stem.partition("-")[2]),
    )
    if not paths:
        raise ValueError(f"no process records in {directory}")
    return paths


def _median_us(timing):
    """Return the median of a timing entry from its raw samples, if kept."""
    if "samples_us" in timing:
        return statistics.median(timing["samples_us"])
    return timing["summary_us"]["median"]


def _phases(row):
    """Yield the timed planning and end-to-end phases of a comparison row."""
    planning = row.get("planning")
    if planning is not None and planning.get("status") != "not-run":
        yield "planning", planning
    end = row.get("end_to_end")
    if end is not None and end.get("status") != "not-run":
        for mode in ("warm_preparation", "changing_geometry"):
            yield f"end_to_end_{mode}", end[mode]


def _series(record):
    """Return each candidate's median and rate from one process record.

    Returns:
        A mapping from row label to timing method to candidate to its
        ``median_us`` and ``gb_per_s``. A comparison record contributes
        its kernel timing methods, its planning and end-to-end phases, and
        its compilation components under the row ``compilation``.

    Raises:
        ValueError: If the record is not one this driver can read.
    """
    benchmark = record.get("benchmark")
    if benchmark == "fresh-offsets-segmented-sum":
        return {
            row["distribution"]: {
                "end_to_end": {
                    candidate: {
                        "median_us": summary["median"],
                        "gb_per_s": row["effective_gb_per_s"][candidate][
                            "median"
                        ],
                    }
                    for candidate, summary in row["summary_us"].items()
                }
            }
            for row in record["results"]
        }
    if benchmark == "swage-triton-comparison":
        series = {}
        for row in record["results"]:
            label = (
                f"vadd n={row['n']}"
                if row["case"] == "vadd"
                else f"{row['distribution']} seed={row['seed']}"
            )
            methods = series.setdefault(label, {})
            for candidate, timings in row["timings"].items():
                for method, timing in timings.items():
                    if "summary_us" in timing:
                        methods.setdefault(method, {})[candidate] = {
                            "median_us": _median_us(timing),
                            "gb_per_s": timing.get("effective_gb_per_s"),
                        }
            for method, phase in _phases(row):
                for candidate, timing in phase["timings"].items():
                    methods.setdefault(method, {})[candidate] = {
                        "median_us": _median_us(timing),
                        "gb_per_s": None,
                    }
        compilation = record.get("compilation", {})
        if compilation.get("status") == "measured":
            series["compilation"] = {
                "compilation": {
                    component: {
                        "median_us": _median_us(timing),
                        "gb_per_s": None,
                    }
                    for component, timing in compilation["timings"].items()
                }
            }
        return series
    raise ValueError(f"cannot summarize records of benchmark {benchmark!r}")


def _timed_candidates(series):
    """Return every candidate that some row and method of a series timed."""
    return {
        candidate
        for methods in series.values()
        for candidates in methods.values()
        for candidate in candidates
    }


def _default_references(series):
    """Return the PyTorch reference candidates that the records time."""
    timed = _timed_candidates(series)
    return [name for name in _DEFAULT_REFERENCES if name in timed]


def _spread(values):
    """Return per-process values with their median, minimum, and maximum."""
    return {
        "process_values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _require_references(series, references):
    """Refuse a reference that the records cannot be compared against.

    Args:
        series: ``_series`` of one process record.
        references: The candidates ratios are to be taken against.

    Raises:
        ValueError: If a reference was not timed in any row. Ratios against
            a misspelled or filtered-out candidate would otherwise be
            missing from the summary without a word.
    """
    timed = _timed_candidates(series)
    for reference in references:
        if reference not in timed:
            raise ValueError(
                f"reference {reference} was not timed in any row; the "
                f"candidates are {', '.join(sorted(timed))}"
            )


def _summarize(series, references):
    """Combine the per-process medians of every candidate.

    Args:
        series: ``_series`` of each process record, in run order.
        references: Candidates every ratio is taken against.

    Returns:
        The summary rows, the incomplete candidates, and the rows without
        a reference. A candidate is summarized only when every process
        timed it; otherwise it is listed under ``incomplete`` with the
        processes that did, counted from one. A ratio is formed inside each
        process, from that process's two medians, before the processes are
        combined. Where a reference was not timed by every process no ratio
        against it is formed, and the row and method are listed for it.
    """
    rows = {}
    incomplete = {}
    missing = {}
    labels = {
        (row, method, candidate)
        for process in series
        for row, methods in process.items()
        for method, candidates in methods.items()
        for candidate in candidates
    }
    for row, method, candidate in sorted(labels):
        timed = [
            index
            for index, process in enumerate(series, start=1)
            if candidate in process.get(row, {}).get(method, {})
        ]
        if len(timed) != len(series):
            incomplete.setdefault(row, {}).setdefault(method, {})[candidate] = (
                timed
            )
            continue
        entries = [process[row][method] for process in series]
        summary = {
            "median_us": _spread(
                [entry[candidate]["median_us"] for entry in entries]
            )
        }
        rates = [entry[candidate]["gb_per_s"] for entry in entries]
        if None not in rates:
            summary["effective_gb_per_s"] = _spread(rates)
        for reference in references:
            if all(reference in entry for entry in entries):
                summary[f"ratio_to_{reference}"] = _spread(
                    [
                        entry[candidate]["median_us"]
                        / entry[reference]["median_us"]
                        for entry in entries
                    ]
                )
            else:
                methods = missing.setdefault(reference, {}).setdefault(row, [])
                if method not in methods:
                    methods.append(method)
        rows.setdefault(row, {}).setdefault(method, {})[candidate] = summary
    return rows, incomplete, missing


def _code_identity(records):
    """Return what every process measured, refusing a mixed run.

    Raises:
        ValueError: If two records differ in benchmark, revision, machine,
            library versions, native library hashes, or loaded PTX hashes.
    """

    def identity(record):
        return {
            "benchmark": record["benchmark"],
            "revision": record["source"]["revision"],
            **{
                field: record["provenance"].get(field) for field in _CODE_FIELDS
            },
        }

    first = identity(records[0])
    for index, record in enumerate(records[1:], start=2):
        for field, value in identity(record).items():
            if value != first[field]:
                raise ValueError(
                    f"process {index} differs from process 1 in {field}: "
                    f"{value!r} against {first[field]!r}; the summary "
                    "would mix different code or machines"
                )
    return first


def _summary(paths, records, references, command):
    """Return the summary of one run's process records.

    Args:
        paths: The record paths in run order.
        records: The records read from them.
        references: Candidates every ratio is taken against.
        command: The harness command, or None when it is not known.

    Returns:
        The summary.

    Raises:
        ValueError: If the processes measured different code, or if a
            reference was not timed in any row.
    """
    code = _code_identity(records)
    series = [_series(record) for record in records]
    _require_references(series[0], references)
    rows, incomplete, missing = _summarize(series, references)
    return {
        "benchmark": code["benchmark"],
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "smoke": any(record.get("smoke", False) for record in records),
        "command": command,
        "processes": len(records),
        "references": references,
        "statistics": (
            "every process contributes the median of its own samples; the "
            "summary lists those per-process medians with their median, "
            "minimum, and maximum; a ratio is formed inside each process "
            "from its two medians, once against each reference; every "
            "candidate is reported and none is selected"
        ),
        "code": code,
        "process_records": [
            {
                "record": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "recorded_at": record.get("recorded_at"),
                "worktree_clean": record["source"].get("worktree_clean"),
                **{
                    field: record["provenance"].get(field)
                    for field in _PROCESS_FIELDS
                },
            }
            for path, record in zip(paths, records, strict=True)
        ],
        "rows": rows,
        "incomplete": incomplete,
        "reference_missing": missing,
    }


def _summarize_finished_run(directory, references):
    """Summarize the records of a finished run again; return the output.

    Raises:
        FileExistsError: If the summary of these references exists.
        ValueError: If the directory holds no record, or a reference was
            not timed in any row.
    """
    paths = _record_paths(directory)
    records = [load_unique_json(path) for path in paths]
    if references is None:
        references = _default_references(_series(records[0]))
    output = directory / f"summary-{'-'.join(references)}.json"
    if output.exists():
        raise FileExistsError(f"{output} exists; a summary is never replaced")
    summary = _summary(paths, records, references, None)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output


def _require_empty_output(output_dir):
    """Refuse an output directory that holds the files of another run."""
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError(
            f"the output directory {output_dir} is not empty; give every "
            "run its own directory"
        )


def main(argv=None) -> None:
    """Run each repetition in a fresh process and write the run's outputs.

    A comparison run writes manifest.json, whose aggregates are validated
    again from the raw records before it is renamed into place, and
    summary.json. A run of another harness writes summary.json.
    """
    arguments = _arguments(argv)
    if arguments.summarize is not None:
        output = _summarize_finished_run(
            arguments.summarize, arguments.reference
        )
        print(json.dumps({"output": str(output)}, sort_keys=True))
        return
    root = pathlib.Path(__file__).resolve().parents[1]
    comparison = arguments.harness == _COMPARISON
    harness = (root / arguments.harness).resolve()
    if not harness.is_file() or not harness.is_relative_to(root):
        raise ValueError(f"harness {arguments.harness} is not in {root}")
    output_dir = arguments.output_dir.resolve()
    _require_source_neutral_output(root, output_dir)
    _require_empty_output(output_dir)
    expected_source = _source_metadata(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    pending_path = output_dir / "manifest.pending.json"
    failure_manifest_path = output_dir / "failure-manifest.json"
    summary_path = output_dir / "summary.json"
    command = _harness_command(arguments)
    references = arguments.reference

    archival_ineligibility_reasons = []
    if arguments.allow_shared_gpu_engineering:
        archival_ineligibility_reasons.append(
            "shared-GPU engineering mode selected; "
            "exclusive GPU allocation was not asserted"
        )
    if not arguments.archival_source:
        archival_ineligibility_reasons.append(
            "archival source revision was not asserted"
        )
    children = []
    paths = []
    child_records = []
    observations = []
    try:
        temporary_parent = _external_temporary_parent(root, output_dir)
        for index in range(arguments.repetitions):
            if _source_metadata(root) != expected_source:
                raise RuntimeError("source changed during benchmark campaign")
            child_path = output_dir / f"process-{index:03d}.json"
            child_command = [
                sys.executable,
                str(harness),
                "--output",
                str(child_path),
                *command[1:],
            ]
            telemetry = {"pre_process": _nvidia_telemetry()}
            observation = {
                "process_index": index,
                "path": child_path.name,
                "command": child_command,
                "launched": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "nvidia_telemetry": telemetry,
            }
            observations.append(observation)
            try:
                pre_reasons = _apply_compute_process_policy(
                    telemetry["pre_process"],
                    f"process {index} pre-process boundary",
                    exclusive_gpu_allocated=arguments.exclusive_gpu_allocated,
                )
                archival_ineligibility_reasons.extend(pre_reasons)
                with tempfile.TemporaryDirectory(
                    prefix=f"swage-comparison-{index:03d}-",
                    dir=temporary_parent,
                ) as temporary:
                    cache_root = pathlib.Path(temporary)
                    child_environment = os.environ.copy()
                    for variable, directory in (
                        ("SWAGE_CACHE_DIR", "swage"),
                        ("TRITON_CACHE_DIR", "triton"),
                    ):
                        cache = cache_root / directory
                        cache.mkdir()
                        child_environment[variable] = str(cache)
                    observation["launched"] = True
                    completed = subprocess.run(
                        child_command,
                        cwd=root,
                        env=child_environment,
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    observation["returncode"] = completed.returncode
                    observation["stdout"] = completed.stdout
                    observation["stderr"] = completed.stderr
                    if completed.stdout:
                        print(completed.stdout, end="")
                    if completed.stderr:
                        print(completed.stderr, end="", file=sys.stderr)
            finally:
                telemetry["post_process"] = _nvidia_telemetry()
                if child_path.is_file():
                    observation["sha256"] = _sha256(child_path)
            completed.check_returncode()
            post_reasons = _apply_compute_process_policy(
                telemetry["post_process"],
                f"process {index} post-process boundary",
                exclusive_gpu_allocated=arguments.exclusive_gpu_allocated,
            )
            archival_ineligibility_reasons.extend(post_reasons)
            if _source_metadata(root) != expected_source:
                raise RuntimeError("source changed during benchmark process")
            child = load_unique_json(child_path)
            if comparison:
                validate_child(child)
            if index == 0:
                # A reference that the first process did not time will not
                # be timed by the others either.
                series = _series(child)
                if references is None:
                    references = _default_references(series)
                _require_references(series, references)
            children.append(child)
            paths.append(child_path)
            child_records.append(
                {
                    "process_index": index,
                    "path": child_path.name,
                    "sha256": observation["sha256"],
                    "recorded_at": child["recorded_at"],
                    "nvidia_telemetry": telemetry,
                }
            )

        summary = _summary(paths, children, references, command)
        if comparison:
            aggregate = aggregate_children(children)
            if aggregate["source"] != expected_source:
                raise RuntimeError("child source does not match campaign input")
            if aggregate["environment"]["compiler"]["build_type"] != "Release":
                archival_ineligibility_reasons.append(
                    "compiler build type is not Release"
                )
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "benchmark": "swage-triton-comparison-process-campaign",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "independent_processes": True,
                "archival_eligible": not archival_ineligibility_reasons,
                "archival_ineligibility_reasons": (
                    archival_ineligibility_reasons
                ),
                "controls": {
                    "suite": arguments.suite,
                    "repetitions": arguments.repetitions,
                    "samples_per_process": arguments.samples,
                    "warmups_per_process": arguments.warmups,
                    "harness_options": list(arguments.options),
                    "gpu_execution_mode": (
                        "exclusive-asserted"
                        if arguments.exclusive_gpu_allocated
                        else "shared-gpu-engineering"
                    ),
                    "exclusive_gpu_allocated": (
                        arguments.exclusive_gpu_allocated
                    ),
                    "allow_shared_gpu_engineering": (
                        arguments.allow_shared_gpu_engineering
                    ),
                    "archival_source": arguments.archival_source,
                    "compute_process_observation_scope": (
                        "pre/post child-process boundary samples; empty "
                        "samples do not prove exclusive allocation"
                    ),
                },
                "children": child_records,
                **aggregate,
            }
            with pending_path.open("x") as output:
                output.write(json.dumps(manifest, indent=2) + "\n")
            load_campaign(
                pending_path, require_archival=manifest["archival_eligible"]
            )
            pending_path.rename(manifest_path)
        with summary_path.open("x") as output:
            output.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    except BaseException as error:
        if pending_path.exists():
            pending_path.rename(failure_manifest_path)
        _write_failure_observations(root, output_dir, observations, error)
        raise
    written = {"output": str(manifest_path if comparison else summary_path)}
    written["summary"] = str(summary_path)
    print(json.dumps(written, sort_keys=True))


if __name__ == "__main__":
    main()
