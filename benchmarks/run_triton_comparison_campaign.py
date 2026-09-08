# benchmarks/run_triton_comparison_campaign.py
"""Run independent Swage/Triton benchmark processes and aggregate them."""

import argparse
import csv
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

from benchmark_campaign import (
    aggregate_children,
    compute_process_reasons,
    load_campaign,
    load_unique_json,
    validate_child,
)


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
    if arguments.archival_source and arguments.allow_shared_gpu_engineering:
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


def main() -> None:
    """Run each repetition in a fresh process and write a valid manifest."""
    arguments = _arguments()
    if arguments.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if arguments.samples <= 0 or arguments.warmups < 0:
        raise ValueError("samples must be positive and warmups nonnegative")
    root = pathlib.Path(__file__).resolve().parents[1]
    benchmark = root / "benchmarks" / "benchmark_triton_comparison.py"
    output_dir = arguments.output_dir.resolve()
    _require_source_neutral_output(root, output_dir)
    expected_source = _source_metadata(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    pending_path = output_dir / "manifest.pending.json"
    failure_manifest_path = output_dir / "failure-manifest.json"
    for path in (
        manifest_path,
        pending_path,
        failure_manifest_path,
        output_dir / "failure-observations.json",
    ):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    if any(output_dir.glob("process-*.json")):
        raise FileExistsError("output directory already contains child records")

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
    observations = []
    try:
        temporary_parent = _external_temporary_parent(root, output_dir)
        for index in range(arguments.repetitions):
            if _source_metadata(root) != expected_source:
                raise RuntimeError("source changed during benchmark campaign")
            child_path = output_dir / f"process-{index:03d}.json"
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
            telemetry = {"pre_process": _nvidia_telemetry()}
            observation = {
                "process_index": index,
                "path": child_path.name,
                "command": command,
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
                        command,
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
            validate_child(child)
            children.append(child)
            child_records.append(
                {
                    "process_index": index,
                    "path": child_path.name,
                    "sha256": observation["sha256"],
                    "recorded_at": child["recorded_at"],
                    "nvidia_telemetry": telemetry,
                }
            )

        aggregate = aggregate_children(children)
        if aggregate["source"] != expected_source:
            raise RuntimeError("child source does not match campaign input")
        if aggregate["environment"]["compiler"]["build_type"] != "Release":
            archival_ineligibility_reasons.append(
                "compiler build type is not Release"
            )
        manifest = {
            "schema_version": 1,
            "benchmark": "swage-triton-comparison-process-campaign",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "independent_processes": True,
            "archival_eligible": not archival_ineligibility_reasons,
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
        with pending_path.open("x") as output:
            output.write(json.dumps(manifest, indent=2) + "\n")
        load_campaign(
            pending_path, require_archival=manifest["archival_eligible"]
        )
        pending_path.rename(manifest_path)
    except BaseException as error:
        if pending_path.exists():
            pending_path.rename(failure_manifest_path)
        _write_failure_observations(root, output_dir, observations, error)
        raise
    print(json.dumps({"output": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
