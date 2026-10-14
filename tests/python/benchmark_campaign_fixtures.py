# tests/python/benchmark_campaign_fixtures.py
"""Complete CPU-only comparison evidence fixtures shared by harness tests."""

import copy
import hashlib
import importlib.util
import json
import pathlib
import sys
import types
from functools import lru_cache

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_TIMESTAMP = "2026-09-08T12:00:00+00:00"
_DISTRIBUTIONS = (
    "many-tiny",
    "uniform",
    "log-normal",
    "bimodal",
    "zipf-like",
    "few-huge",
    "one-outlier",
    "soc-epinions1-outdegree-v1",
)
TICKS = {"clock": 0.05, "event": 0.032}


@lru_cache(maxsize=None)
def _module(name):
    benchmarks = str(_ROOT / "benchmarks")
    if benchmarks not in sys.path:
        # The harness imports its sibling modules by name.
        sys.path.insert(0, benchmarks)
    spec = importlib.util.spec_from_file_location(
        f"_fixture_{name}", _ROOT / "benchmarks" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restamp(measurement, *, useful_bytes, tick):
    """Recompute what a timing entry derives from its raw samples."""
    samples = measurement["samples_us"]
    summary = _module("benchmark_campaign").summarize_us(samples)
    measurement["summary_us"] = summary
    if "launches_per_sample" in measurement:
        sample = summary["median"] * measurement["launches_per_sample"]
        measurement["timer_tick_us"] = tick
        measurement["tick_fraction_of_sample"] = (
            None if tick is None else tick / sample
        )
        measurement["effective_gb_per_s"] = useful_bytes / (
            summary["median"] * 1_000.0
        )
    return measurement


def _measurement(median, samples):
    values = [float(median)] * samples
    return {
        "samples_us": values,
        "summary_us": _module("benchmark_campaign").summarize_us(values),
    }


def _timing_entry(median, samples, *, metric, useful_bytes):
    entry = {
        "samples_us": [float(median)] * samples,
        "launches_per_sample": 1 if metric == "call" else 32,
    }
    if metric == "graph":
        entry = {"available": True, **entry}
    return restamp(
        entry,
        useful_bytes=useful_bytes,
        tick=TICKS["clock" if metric == "call" else "event"],
    )


def _timing_method(candidates, samples, *, kernel=False):
    candidates = list(candidates)
    width = len(candidates)
    method = {
        "sampling": "deterministic_rotating_interleaved",
        "base_candidate_order": candidates,
        "round_rotation": "left by round_index modulo candidate_count",
        "timed_rounds": samples,
        "order_position_counts": {
            name: [
                sum(
                    candidates[(position + round_index) % width] == name
                    for round_index in range(samples)
                )
                for position in range(width)
            ]
            for name in candidates
        },
    }
    if kernel:
        method.update(
            {
                "graph_preparation": "all graphs captured before timed replay",
                "units": {
                    "call": "microseconds per synchronized Python call",
                    "batched_event": "microseconds per launch in a batch",
                    "graph": "microseconds per launch in a graph replay",
                },
            }
        )
    else:
        method["unit"] = "microseconds per synchronized operation"
    return method


def _kernel_timings(candidates, median, samples, useful_bytes):
    return {
        "timing_method": _timing_method(candidates, samples, kernel=True),
        "timings": {
            name: {
                metric: _timing_entry(
                    median, samples, metric=metric, useful_bytes=useful_bytes
                )
                for metric in ("call", "batched_event", "graph")
            }
            for name in candidates
        },
    }


def _phase(candidates, median, samples, *, planning=False, changing=False):
    phase = {
        "geometry": "rotated lengths" if changing else "fixed offsets reused",
        "included": [
            "tensor validation",
            "host planning",
            "device synchronization",
        ],
        "timing_method": _timing_method(candidates, samples),
        "timings": {name: _measurement(median, samples) for name in candidates},
    }
    if planning:
        phase["excluded"] = [
            "kernel compilation/memo lookup",
            "contract binding",
            "module lease/load",
            "output allocation",
            "kernel launch",
        ]
    else:
        phase["included"].extend(["output allocation", "kernel launch"])
    return phase


@lru_cache(maxsize=1)
def _segmented_geometry():
    distributions = _module("distributions")
    harness = _module("benchmark_triton_comparison")
    rows = []
    for name in _DISTRIBUTIONS:
        if name == _DISTRIBUTIONS[-1]:
            lengths, provenance = _module("real_traces").load_real_trace(name)
        else:
            lengths = distributions.generate_lengths(name, 32_768, 7)
            provenance = None
        stats = distributions.summarize_lengths(lengths)
        warp_tasks = sum(length <= 32 for length in lengths)
        cta_tasks = len(lengths) - warp_tasks
        warp_programs = (warp_tasks + 3) // 4
        row = {
            "case": "segmented-sum",
            "distribution": name,
            "seed": 7,
            "values": "ones",
            "segment_count": len(lengths),
            "statistics": stats,
            "useful_bytes": harness._useful_bytes(stats["total"], len(lengths)),
            "check": {
                "exact_segments": len(lengths),
                "bounded_segments": 0,
                "unchecked_segments": 0,
            },
            "skipped": {},
            "excluded": [],
            "triton_sweep_configs": [
                {"block": block, "num_warps": warps}
                for block, warps in harness._triton_sum_configs(stats["max"])
            ],
            "triton_looped_sweep_configs": [
                {"block": block, "num_warps": warps}
                for block, warps in harness._triton_looped_configs()
            ],
            "matched_task_partition_triton": {
                "comparison": (
                    "same host task partition; not identical execution"
                ),
                "launches": "packed short tasks plus CTA tasks when nonempty",
                "warp_threshold_elements": 32,
                "warp_tasks": warp_tasks,
                "cta_tasks": cta_tasks,
                "short_tasks_per_program": 4,
                "cta_block_elements": 4096,
                "primary_cta_num_warps": 1,
                "cta_num_warps_sweep": [1, 2, 4, 8],
            },
            "triton_fused_contract": {
                "result_name": "triton_fused",
                "launch_count": 1,
                "logical_lanes_per_program": 128,
                "short_task_slots": 4,
                "lanes_per_short_task": 32,
                "program_order": "packed short tasks first, then CTA tasks",
                "cta_accumulation": (
                    "128-lane block-stride loads through 4096 elements"
                ),
                "maximum_segment_length": 4096,
                "physical_num_warps": 4,
                "warp_programs": warp_programs,
                "cta_programs": cta_tasks,
                "grid_programs": warp_programs + cta_tasks,
            },
            "padded_layout": harness._padded_layout(lengths),
        }
        if provenance is not None:
            row["trace_provenance"] = provenance
        rows.append(row)
    return rows


def _compilation(rows, method, median):
    campaign = _module("benchmark_campaign")
    signatures = campaign._signatures(rows, method)
    swage = ["swage_warp", "swage_cta", "swage_mixed"]
    if not any(
        candidate.startswith("swage_")
        for signature in signatures
        for candidate in signature["candidates"]
    ):
        swage = []
    triton, configurations = campaign._expected_triton_components(signatures)
    timings = {name: _measurement(median, 1) for name in swage + triton}
    totals = {
        name: components
        for name, components in (
            ("swage_total", swage),
            ("triton_total", triton),
        )
        if components
    }
    for name, components in totals.items():
        timings[name] = _measurement(
            sum(
                timings[component]["samples_us"][0] for component in components
            ),
            1,
        )
    return {
        "status": "measured",
        "case": "segmented-sum",
        "timing_method": "time.perf_counter_ns wall clock; no kernel launch",
        "compiler_order": ["swage", "triton"],
        "component_order": swage + triton,
        "configurations": {
            "swage_kernels": [name[len("swage_") :] for name in swage],
            **configurations,
            "triton_case_signatures": signatures,
        },
        "derived_totals": totals,
        "cache_policy": {
            "fresh_process": True,
            "unique_directories": True,
            "swage_initially_empty": True,
            "triton_initially_empty": True,
        },
        "scope": {
            "swage": {
                "included": ["lowering", "PTX production"],
                "excluded": ["host classification", "module load", "launch"],
            },
            "triton": {
                "included": ["warmup specialization and compilation"],
                "excluded": ["imports", "JIT function construction", "launch"],
            },
        },
        "not_applicable": {
            "torch_segment_reduce": (
                "eager operator; no measured JIT compilation"
            ),
            "torch_padded": "eager operator; no measured JIT compilation",
        },
        "timings": timings,
    }


def methodology(*, samples=2, warmups=1, suite="vadd", only=None, exclude=()):
    """Return the methodology the harness writes for these controls."""
    arguments = types.SimpleNamespace(
        suite=suite,
        warmups=warmups,
        samples=samples,
        distributions=list(_DISTRIBUTIONS),
        segment_count=32_768,
        seeds=[7],
        values="ones",
        candidates=only,
        exclude_candidates=list(exclude),
    )
    method = _module("benchmark_triton_comparison")._methodology(
        arguments, dict(TICKS)
    )
    return method


def provenance():
    """Return a complete provenance block of one process."""
    return {
        "gpu": "NVIDIA RTX A6000",
        "gpu_uuid": "GPU-fixture",
        "cpu_model": "Fixture CPU",
        "pytorch": "2.12.0+cu130",
        "triton": "3.7.0",
        "swage": "0.5.2",
        "llvm_pin": "llvmorg-22.1.8",
        "llvm_linked": "22.1.8",
        "cuda_driver": "13.0",
        "native_sha256": {"/build/libSwage.so": "a" * 64},
        "loaded_ptx": [
            {"kernel": "segmented_sum", "sha256": "b" * 64, "bytes": 2}
        ],
        "gpu_state_before": {"gpu": {"temperature.gpu": "50"}},
        "cpu_frequency_before": {"governors": {"performance": 24}},
        "gpu_state_after": {"gpu": {"temperature.gpu": "55"}},
        "other_compute_process_seen": False,
        "cpu_frequency_after": {"governors": {"performance": 24}},
        "cpu_governor_unchanged": True,
    }


def make_child(
    *,
    median=10.0,
    samples=2,
    warmups=1,
    suite="vadd",
    only=None,
    exclude=(),
):
    """Return a complete valid child; every ordinary measurement uses median.

    Args:
        median: Every sample of every ordinary measurement.
        samples: Samples per measurement.
        warmups: Warmups per measurement.
        suite: ``vadd``, ``segmented-sum``, or ``all``.
        only: Candidate filter selectors to keep, or None for all.
        exclude: Candidate filter selectors to leave out.
    """
    if suite not in {"vadd", "segmented-sum", "all"}:
        raise ValueError("unknown fixture suite")
    harness = _module("benchmark_triton_comparison")
    method = methodology(
        samples=samples,
        warmups=warmups,
        suite=suite,
        only=only,
        exclude=exclude,
    )
    record = {
        "schema_version": 2,
        "benchmark": "swage-triton-comparison",
        "recorded_at": _TIMESTAMP,
        "source": {"revision": "a" * 40, "worktree_clean": True, "dirty": []},
        "environment": {
            "platform": "Linux",
            "python": "3.12.0",
            "pytorch": "2.9.0",
            "pytorch_cuda": "12.8",
            "triton": "3.7.0",
            "cuda_driver": "13.0",
            "gpu": "NVIDIA RTX A6000",
            "compute_capability": "sm_86",
            "multiprocessors": 84,
            "total_memory_bytes": 48 * 1024**3,
            "compiler": {
                "package_version": "0.5.2",
                "source_revision": "a" * 40,
                "source_clean": True,
                "llvm_version": "llvmorg-22.1.8",
                "llvm_pin": "llvmorg-22.1.8",
                "build_type": "Release",
            },
        },
        "provenance": provenance(),
        "methodology": method,
        "compilation": {
            "status": "not-run",
            "reason": "segmented-sum suite not selected",
        },
        "results": [],
    }
    if suite in {"vadd", "all"}:
        candidates = [
            "swage",
            "torch",
            "triton_b128",
            "triton_b256",
            "triton_b512",
            "triton_b1024",
        ]
        for exponent in (10, 12, 14, 16, 18, 20, 22):
            n = 1 << exponent
            record["results"].append(
                {
                    "case": "vadd",
                    "n": n,
                    "seed": 7,
                    "useful_bytes": 12 * n,
                    "swage_block": 256,
                    "swage_grid": (n + 255) // 256,
                    "triton_sweep_blocks": [128, 256, 512, 1024],
                    "launch_contract": {
                        "swage": "BLOCK=256 for the vector-add campaign",
                        "triton": "compile-time BLOCK, one program per block",
                    },
                    **_kernel_timings(candidates, median, samples, 12 * n),
                }
            )
    if suite in {"segmented-sum", "all"}:
        rows = copy.deepcopy(_segmented_geometry())
        for row in rows:
            maximum = row["statistics"]["max"]
            candidates = harness._case_candidates(maximum, only, exclude)
            full = harness._row_candidates(maximum)
            wanted = harness._select(full, only, exclude)
            row["excluded"] = [name for name in full if name not in wanted]
            row["candidate_order"] = candidates
            row.update(
                _kernel_timings(
                    candidates, median, samples, row["useful_bytes"]
                )
            )
            orchestrated = [
                name
                for name in ("swage_mixed", "triton_matched_task_partition")
                if name in candidates
            ]
            if not orchestrated:
                not_run = {"status": "not-run", "reason": "not timed"}
                row["planning"] = not_run
                row["end_to_end"] = dict(not_run)
                continue
            row["planning"] = _phase(
                orchestrated, median, samples, planning=True
            )
            row["end_to_end"] = {
                "artifact_jit_warmup": (
                    "one untimed complete operation before all samples"
                ),
                "compilation_excluded": True,
                "graph_samples_combined": False,
                "warm_preparation": _phase(orchestrated, median, samples),
                "changing_geometry": _phase(
                    orchestrated, median, samples, changing=True
                ),
            }
        record["results"].extend(rows)
        record["compilation"] = _compilation(rows, method, median)
    return record


def make_telemetry(*, processes=None):
    """Return a full NVIDIA boundary observation, optionally with contenders."""
    return {
        "available": True,
        "recorded_at": _TIMESTAMP,
        "fields": {
            "sm_clock_mhz": "MHz",
            "memory_clock_mhz": "MHz",
            "power_draw_watts": "W",
            "temperature_celsius": "degrees Celsius",
        },
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-fixture",
                "sm_clock_mhz": 1800,
                "memory_clock_mhz": 8000,
                "power_draw_watts": 80.5,
                "temperature_celsius": 45.0,
            }
        ],
        "compute_processes": {
            "available": True,
            "fields": {"used_memory_mib": "MiB"},
            "processes": copy.deepcopy(processes) if processes else [],
        },
    }


def write_campaign(directory, *, children=None, archival=True):
    """Write a complete campaign; defaults to five segmented child processes."""
    campaign = _module("benchmark_campaign")
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if children is None:
        children = [
            make_child(median=median, suite="segmented-sum")
            for median in (10.0, 20.0, 30.0, 40.0, 100.0)
        ]
    aggregate = campaign.aggregate_children(children)
    rows = children[0]["results"]
    suites = {row["case"] for row in rows}
    suite = "all" if len(suites) == 2 else next(iter(suites))
    method = aggregate["methodology"]
    entries = []
    for index, child in enumerate(children):
        path = directory / f"process-{index:03d}.json"
        path.write_text(json.dumps(child, indent=2) + "\n")
        entries.append(
            {
                "process_index": index,
                "path": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "recorded_at": child["recorded_at"],
                "nvidia_telemetry": {
                    "pre_process": make_telemetry(),
                    "post_process": make_telemetry(),
                },
            }
        )
    reasons = (
        []
        if archival
        else [
            "shared-GPU engineering mode selected; "
            "exclusive GPU allocation was not asserted",
            "archival source revision was not asserted",
        ]
    )
    if aggregate["environment"]["compiler"]["build_type"] != "Release":
        reasons.append("compiler build type is not Release")
    manifest = {
        "schema_version": 2,
        "benchmark": "swage-triton-comparison-process-campaign",
        "recorded_at": _TIMESTAMP,
        "independent_processes": True,
        "archival_eligible": not reasons,
        "archival_ineligibility_reasons": reasons,
        "controls": {
            "suite": suite,
            "repetitions": len(children),
            "samples_per_process": method[
                "samples_per_candidate_per_measurement"
            ],
            "warmups_per_process": method[
                "warmups_per_candidate_per_measurement"
            ],
            "harness_options": [],
            "gpu_execution_mode": "exclusive-asserted"
            if archival
            else "shared-gpu-engineering",
            "exclusive_gpu_allocated": archival,
            "allow_shared_gpu_engineering": not archival,
            "archival_source": archival,
            "compute_process_observation_scope": (
                "pre/post child-process boundary samples; empty samples "
                "do not prove exclusive allocation"
            ),
        },
        "children": entries,
        **aggregate,
    }
    path = directory / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
