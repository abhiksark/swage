# benchmarks/benchmark_processes.py
"""Repeat one benchmark configuration in independent processes.

Samples taken inside one process share its allocator state, its compiled
kernels, and whatever the machine was doing at the time, so their spread
understates how far a median moves from one run to the next. This driver
starts the same harness command in several fresh interpreters, one after
another, and summarizes the per-process medians.

Usage, from the repository root with PYTHONPATH=python:build/python_packages
and the harness command after the separator:

    python3 benchmarks/benchmark_processes.py --output-dir DIR --
        benchmarks/benchmark_fresh_offsets.py --distributions power-law

Each process writes DIR/process-<k>.json; the driver adds --output itself.
DIR/summary.json then holds, for every row, timing method, and candidate,
the per-process medians with their median, minimum, and maximum, and the
same for the ratio to each reference candidate named with --reference. A
reference that no row timed is an error, raised after the first process.

Records that a run left behind can be summarized again against other
references, without running anything:

    python3 benchmarks/benchmark_processes.py --summarize DIR
        --reference triton_looped_b256_w4

This writes DIR/summary-<references>.json and never replaces a summary.
The driver understands the records of benchmark_fresh_offsets.py and
benchmark_triton_comparison.py. It imports neither PyTorch nor Triton.
"""

import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import sys
from datetime import datetime, timezone

_PROCESSES = 5
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
    """Parse the process count, the output directory, and the command."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--processes",
        type=int,
        help=(
            "Independent processes to run, one after another. The default "
            f"is {_PROCESSES}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="New or empty directory outside the checkout, for a run.",
    )
    parser.add_argument(
        "--summarize",
        type=pathlib.Path,
        metavar="DIR",
        help=(
            "Summarize the process records of a finished run in DIR "
            "instead of running a command."
        ),
    )
    parser.add_argument(
        "--reference",
        nargs="+",
        default=["torch"],
        metavar="NAME",
        help=(
            "Candidates that ratios are taken against, each by its full "
            "name, such as triton_looped_b256_w4. The default is torch. "
            "End the list with -- before a harness command."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Harness script and its options, without --output.",
    )
    arguments = parser.parse_args(argv)
    if arguments.command[:1] == ["--"]:
        del arguments.command[0]
    if arguments.summarize is not None:
        if (
            arguments.command
            or arguments.output_dir is not None
            or arguments.processes is not None
        ):
            parser.error(
                "--summarize reads the records of a finished run; it takes "
                "no command, no --output-dir, and no --processes"
            )
        return arguments
    if arguments.output_dir is None:
        parser.error("--output-dir or --summarize is required")
    if not arguments.command:
        parser.error("a harness command is required")
    if "--output" in arguments.command:
        parser.error("the driver passes --output to every process itself")
    if arguments.processes is None:
        arguments.processes = _PROCESSES
    if arguments.processes < 2:
        parser.error("processes must be at least 2; one has no spread")
    return arguments


def _check_output_dir(root, output_dir):
    """Refuse an output directory that would spoil the run or an old one.

    Args:
        root: Root of the checkout that holds the benchmarks.
        output_dir: Directory for the process records and the summary.

    Raises:
        ValueError: If the directory is inside the checkout, where the
            first process record would make the worktree dirty for the
            next process, or if it already holds files.
    """
    root = root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(root):
        raise ValueError(
            f"the output directory must be outside the checkout {root}: "
            f"records written inside it make the worktree dirty for the "
            f"next process; got {output_dir}"
        )
    if output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError(
            f"the output directory {output_dir} is not empty; give every "
            "run its own directory"
        )


def _run_processes(command, output_dir, processes, run=None, first=None):
    """Run the command once per process and return the record paths.

    Args:
        command: Harness script and its options.
        output_dir: Directory that receives ``process-<k>.json``.
        processes: Number of processes.
        run: ``subprocess.run`` or a stand-in.
        first: Callable given the record path of the first process before
            the second one starts. An exception it raises stops the run.

    Returns:
        The record paths in run order.

    Raises:
        subprocess.CalledProcessError: If a process fails; the run stops.
    """
    run = run or subprocess.run
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(1, processes + 1):
        path = output_dir / f"process-{index}.json"
        run([sys.executable, *command, "--output", str(path)], check=True)
        paths.append(path)
        if index == 1 and first is not None:
            first(path)
    return paths


def _record_paths(directory):
    """Return the process records of a finished run, in run order.

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


def _series(record):
    """Return each candidate's median and rate from one process record.

    Returns:
        A mapping from row label to timing method to candidate to its
        ``median_us`` and ``gb_per_s``.

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
                            "median_us": timing["summary_us"]["median"],
                            "gb_per_s": timing.get("effective_gb_per_s"),
                        }
        return series
    raise ValueError(f"cannot summarize records of benchmark {benchmark!r}")


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
    timed = {
        candidate
        for methods in series.values()
        for candidates in methods.values()
        for candidate in candidates
    }
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
        processes that did. A ratio is formed inside each process, from
        that process's two medians, before the processes are combined.
        Where a reference was not timed by every process no ratio against
        it is formed, and the row and method are listed for it.
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
            incomplete.setdefault(row, {}).setdefault(method, {})[
                candidate
            ] = timed
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
                field: record["provenance"].get(field)
                for field in _CODE_FIELDS
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


def main(argv=None):
    """Run the processes, or read their records, and write the summary."""
    arguments = _arguments(argv)
    references = arguments.reference
    if arguments.summarize is not None:
        paths = _record_paths(arguments.summarize)
        output = arguments.summarize / (
            f"summary-{'-'.join(references)}.json"
        )
        if output.exists():
            raise FileExistsError(
                f"{output} exists; a summary is never replaced"
            )
    else:
        root = pathlib.Path(__file__).resolve().parents[1]
        _check_output_dir(root, arguments.output_dir)
        paths = _run_processes(
            arguments.command,
            arguments.output_dir,
            arguments.processes,
            run=subprocess.run,
            # A reference that the first process did not time will not be
            # timed by the others either.
            first=lambda path: _require_references(
                _series(json.loads(path.read_text())), references
            ),
        )
        output = arguments.output_dir / "summary.json"
    records = [json.loads(path.read_text()) for path in paths]
    summary = _summary(
        paths, records, references, arguments.command or None
    )
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
