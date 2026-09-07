# benchmarks/run_triton_comparison_campaign.py
"""Run independent Swage/Triton benchmark processes and aggregate them."""

import argparse
import csv
import hashlib
import json
import math
import pathlib
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone


def _arguments():
    """Parse repeated-campaign controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--suite",
        choices=("all", "vadd", "segmented-sum"),
        default="all",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=25)
    gpu_mode = parser.add_mutually_exclusive_group(required=True)
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
    arguments = parser.parse_args()
    if (
        arguments.archival_source
        and arguments.allow_shared_gpu_engineering
    ):
        parser.error(
            "--archival-source cannot be combined with "
            "--allow-shared-gpu-engineering"
        )
    return arguments


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


def _case_identity(row: dict[str, object]) -> str:
    """Return a stable, explicit benchmark case identity."""
    case = row.get("case")
    if case == "vadd" and type(row.get("n")) is int:
        return f"vadd/n={row['n']}"
    if case == "segmented-sum" and isinstance(
        row.get("distribution"), str
    ):
        return f"segmented-sum/distribution={row['distribution']}"
    raise ValueError("child contains an unrecognized result case")


def _summary_median(measurement: object, path: str) -> float:
    """Read one finite numeric summary median from a measurement."""
    if not isinstance(measurement, dict):
        raise ValueError(f"{path} must be a measurement object")
    samples = measurement.get("samples_us")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} must contain raw samples_us")
    numeric_samples = []
    for sample in samples:
        if type(sample) not in {int, float} or not math.isfinite(sample):
            raise ValueError(f"{path} samples must be finite numbers")
        numeric_samples.append(float(sample))
    summary = measurement.get("summary_us")
    if not isinstance(summary, dict):
        raise ValueError(f"{path} is missing summary_us")
    median = summary.get("median")
    if type(median) not in {int, float}:
        raise ValueError(f"{path} median must be numeric")
    numeric = float(median)
    if not math.isfinite(numeric):
        raise ValueError(f"{path} median must be finite")
    if numeric != statistics.median(numeric_samples):
        raise ValueError(f"{path} median does not match raw samples")
    return numeric


def _process_medians(child: dict[str, object]) -> dict[str, float]:
    """Flatten all child process medians under unit-bearing names."""
    rows = child.get("results")
    if not isinstance(rows, list):
        raise ValueError("child results must be a list")
    medians = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each child result must be an object")
        case = _case_identity(row)
        timings = row.get("timings")
        if not isinstance(timings, dict):
            raise ValueError(f"{case} timings must be an object")
        for candidate, candidate_timings in timings.items():
            if not isinstance(candidate_timings, dict):
                raise ValueError(f"{case}/{candidate} must be an object")
            for metric in ("call", "batched_event", "graph"):
                measurement = candidate_timings.get(metric)
                if (
                    metric == "graph"
                    and isinstance(measurement, dict)
                    and measurement.get("available") is False
                ):
                    continue
                path = f"{case}/{candidate}/{metric}_us"
                if path in medians:
                    raise ValueError(f"duplicate measurement {path}")
                medians[path] = _summary_median(measurement, path)
        preparation = row.get("preparation_only")
        if preparation is not None:
            if not isinstance(preparation, dict):
                raise ValueError(
                    f"{case}/preparation_only must be an object"
                )
            preparation_timings = preparation.get("timings")
            if not isinstance(preparation_timings, dict):
                raise ValueError(
                    f"{case}/preparation_only/timings must be an object"
                )
            for candidate, measurement in preparation_timings.items():
                path = f"{case}/{candidate}/preparation_only_us"
                if path in medians:
                    raise ValueError(f"duplicate measurement {path}")
                medians[path] = _summary_median(measurement, path)
        end_to_end = row.get("end_to_end")
        if end_to_end is None:
            continue
        if not isinstance(end_to_end, dict):
            raise ValueError(f"{case}/end_to_end must be an object")
        for mode in ("warm_preparation", "changing_geometry"):
            mode_record = end_to_end.get(mode)
            if not isinstance(mode_record, dict):
                raise ValueError(f"{case}/{mode} must be an object")
            mode_timings = mode_record.get("timings")
            if not isinstance(mode_timings, dict):
                raise ValueError(f"{case}/{mode}/timings must be an object")
            for candidate, measurement in mode_timings.items():
                path = f"{case}/{candidate}/end_to_end_{mode}_us"
                if path in medians:
                    raise ValueError(f"duplicate measurement {path}")
                medians[path] = _summary_median(measurement, path)
    if not medians:
        raise ValueError("child does not contain process medians")
    return medians


def _without_raw_timings(value):
    """Project nested result records onto configuration metadata."""
    if isinstance(value, dict):
        return {
            key: _without_raw_timings(item)
            for key, item in value.items()
            if key != "timings"
        }
    if isinstance(value, list):
        return [_without_raw_timings(item) for item in value]
    return value


def _result_metadata(child: dict[str, object]):
    """Return result/config metadata with all timed values removed."""
    results = child.get("results")
    if not isinstance(results, list):
        raise ValueError("child results must be a list")
    return _without_raw_timings(results)


def _aggregate_children(
    children: list[dict[str, object]],
) -> dict[str, object]:
    """Validate child agreement and aggregate process-level medians."""
    if not children:
        raise ValueError("at least one child result is required")
    reference = children[0]
    if reference.get("benchmark") != "swage-triton-comparison":
        raise ValueError("child benchmark identity is invalid")
    source = reference.get("source")
    environment = reference.get("environment")
    methodology = reference.get("methodology")
    if (
        not isinstance(source, dict)
        or source.get("worktree_clean") is not True
        or not isinstance(source.get("revision"), str)
        or not source["revision"]
    ):
        raise ValueError("children must report a clean source revision")
    if source.get("dirty") != []:
        raise ValueError("children must report an empty dirty source list")
    if not isinstance(environment, dict):
        raise ValueError("child environment must be an object")
    if not isinstance(methodology, dict):
        raise ValueError("child methodology must be an object")
    result_metadata = _result_metadata(reference)
    flattened = []
    expected_keys = None
    for index, child in enumerate(children):
        if child.get("benchmark") != reference.get("benchmark"):
            raise ValueError(f"child {index} benchmark identity differs")
        if child.get("source") != source:
            raise ValueError(f"child {index} source metadata differs")
        if child.get("environment") != environment:
            raise ValueError(f"child {index} environment metadata differs")
        if child.get("methodology") != methodology:
            raise ValueError(f"child {index} methodology metadata differs")
        if _result_metadata(child) != result_metadata:
            raise ValueError(f"child {index} result metadata differs")
        process_values = _process_medians(child)
        keys = set(process_values)
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError(f"child {index} measurement schema differs")
        flattened.append(process_values)
    aggregates = []
    for measurement in sorted(expected_keys or ()):
        values = [process[measurement] for process in flattened]
        aggregates.append(
            {
                "measurement": measurement,
                "unit": "microseconds",
                "child_process_medians_us": values,
                "median_of_process_medians_us": statistics.median(values),
            }
        )
    return {
        "source": source,
        "environment": environment,
        "methodology": methodology,
        "agreement": {
            "source": True,
            "environment": True,
            "methodology": True,
            "result_metadata": True,
        },
        "process_count": len(children),
        "process_level_aggregates": aggregates,
    }


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
                "sm_clock_mhz": _telemetry_value(
                    columns[2], integral=True
                ),
                "memory_clock_mhz": _telemetry_value(
                    columns[3], integral=True
                ),
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
                        "nvidia-smi returned an incomplete "
                        "compute-process row"
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


def _compute_process_reasons(
    telemetry: dict[str, object], boundary: str
) -> list[str]:
    """Describe why one boundary cannot establish an uncontended run."""
    process_telemetry = telemetry.get("compute_processes")
    if not isinstance(process_telemetry, dict):
        reason = telemetry.get("reason", "observation missing")
        return [
            f"{boundary}: compute-process telemetry unavailable ({reason})"
        ]
    if process_telemetry.get("available") is not True:
        reason = process_telemetry.get("reason", "observation unavailable")
        return [
            f"{boundary}: compute-process telemetry unavailable ({reason})"
        ]
    processes = process_telemetry.get("processes")
    if not isinstance(processes, list):
        return [f"{boundary}: compute-process telemetry malformed"]
    reasons = []
    for process in processes:
        if not isinstance(process, dict):
            reasons.append(f"{boundary}: compute-process record malformed")
            continue
        reasons.append(
            f"{boundary}: competing compute process "
            f"gpu_uuid={process.get('gpu_uuid')!r} "
            f"pid={process.get('pid')!r} "
            f"process_name={process.get('process_name')!r} "
            f"used_memory_mib={process.get('used_memory_mib')!r}"
        )
    return reasons


def _apply_compute_process_policy(
    telemetry: dict[str, object],
    boundary: str,
    *,
    exclusive_gpu_allocated: bool,
) -> list[str]:
    """Fail closed for asserted-exclusive runs and return observed issues."""
    reasons = _compute_process_reasons(telemetry, boundary)
    if reasons and exclusive_gpu_allocated:
        raise RuntimeError("; ".join(reasons))
    return reasons


def main() -> None:
    """Run each repetition in a fresh process and write the manifest."""
    arguments = _arguments()
    if arguments.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if arguments.samples <= 0 or arguments.warmups < 0:
        raise ValueError("samples must be positive and warmups nonnegative")
    root = pathlib.Path(__file__).resolve().parents[1]
    benchmark = root / "benchmarks" / "benchmark_triton_comparison.py"
    _require_source_neutral_output(root, arguments.output_dir)
    expected_source = _source_metadata(root)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = arguments.output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite {manifest_path}")

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
    child_records = []
    for index in range(arguments.repetitions):
        if _source_metadata(root) != expected_source:
            raise RuntimeError("source changed during benchmark campaign")
        child_path = arguments.output_dir / f"process-{index:03d}.json"
        if child_path.exists():
            raise FileExistsError(f"refusing to overwrite {child_path}")
        command = [
            sys.executable,
            str(benchmark),
            "--output",
            str(child_path),
            "--suite",
            arguments.suite,
            "--samples",
            str(arguments.samples),
            "--warmups",
            str(arguments.warmups),
        ]
        telemetry_pre = _nvidia_telemetry()
        pre_reasons = _apply_compute_process_policy(
            telemetry_pre,
            f"process {index} pre-process boundary",
            exclusive_gpu_allocated=arguments.exclusive_gpu_allocated,
        )
        archival_ineligibility_reasons.extend(pre_reasons)
        completed = subprocess.run(command, cwd=root, check=False)
        telemetry_post = _nvidia_telemetry()
        completed.check_returncode()
        post_reasons = _apply_compute_process_policy(
            telemetry_post,
            f"process {index} post-process boundary",
            exclusive_gpu_allocated=arguments.exclusive_gpu_allocated,
        )
        archival_ineligibility_reasons.extend(post_reasons)
        if _source_metadata(root) != expected_source:
            raise RuntimeError("source changed during benchmark process")
        child = json.loads(child_path.read_text())
        children.append(child)
        child_records.append(
            {
                "process_index": index,
                "path": child_path.name,
                "sha256": _sha256(child_path),
                "recorded_at": child.get("recorded_at"),
                "nvidia_telemetry": {
                    "pre_process": telemetry_pre,
                    "post_process": telemetry_post,
                },
            }
        )

    aggregate = _aggregate_children(children)
    if aggregate["source"] != expected_source:
        raise RuntimeError("child source does not match campaign input")
    manifest = {
        "benchmark": "swage-triton-comparison-process-campaign",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "independent_processes": True,
        "archival_eligible": (
            arguments.exclusive_gpu_allocated
            and arguments.archival_source
        ),
        "archival_ineligibility_reasons": archival_ineligibility_reasons,
        "controls": {
            "suite": arguments.suite,
            "repetitions": arguments.repetitions,
            "samples_per_process": arguments.samples,
            "warmups_per_process": arguments.warmups,
            "gpu_execution_mode": (
                "exclusive-asserted"
                if arguments.exclusive_gpu_allocated
                else "shared-gpu-engineering"
            ),
            "exclusive_gpu_allocated": arguments.exclusive_gpu_allocated,
            "allow_shared_gpu_engineering": (
                arguments.allow_shared_gpu_engineering
            ),
            "archival_source": arguments.archival_source,
            "compute_process_observation_scope": (
                "pre/post child-process boundary samples; empty samples "
                "do not prove exclusive allocation"
            ),
        },
        "children": child_records,
        **aggregate,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
