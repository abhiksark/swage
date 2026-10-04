# tests/python/benchmark_campaign_fixtures.py
"""Complete CPU-only comparison evidence fixtures shared by harness tests."""

import copy
import hashlib
import importlib.util
import json
import pathlib
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


@lru_cache(maxsize=None)
def _module(name):
    spec = importlib.util.spec_from_file_location(
        f"_fixture_{name}", _ROOT / "benchmarks" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _measurement(median, samples, *, graph=False):
    values = [float(median)] * samples
    measurement = {
        "samples_us": values,
        "summary_us": _module("benchmark_campaign").summarize_us(values),
    }
    if graph:
        measurement["available"] = True
    return measurement


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
                    "batched_event": (
                        "microseconds per launch in a "
                        "32-launch CUDA-event batch"
                    ),
                    "graph": (
                        "microseconds per launch in a captured "
                        "32-launch graph replay"
                    ),
                },
            }
        )
    else:
        method["unit"] = "microseconds per synchronized operation"
    return method


def _kernel_timings(candidates, median, samples):
    return {
        "timing_method": _timing_method(candidates, samples, kernel=True),
        "timings": {
            name: {
                metric: _measurement(median, samples, graph=metric == "graph")
                for metric in ("call", "batched_event", "graph")
            }
            for name in candidates
        },
    }


def _phase(median, samples, *, planning=False, changing=False):
    candidates = ["swage_mixed", "triton_matched_task_partition"]
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
            "artifact compilation/cache lookup",
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
        configs = [
            {"block": block, "num_warps": warps}
            for block in (32, 64, 128, 256, 512, 1024, 2048, 4096)
            if block >= stats["max"]
            for warps in (1, 2, 4, 8)
            if warps <= block // 32
        ]
        padded_elements = len(lengths) * stats["max"]
        padding = padded_elements - stats["total"]
        row = {
            "case": "segmented-sum",
            "distribution": name,
            "segment_count": len(lengths),
            "statistics": stats,
            "triton_sweep_configs": configs,
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
            "padded_layout": {
                "rows": len(lengths),
                "columns": stats["max"],
                "packed_elements": stats["total"],
                "padded_elements": padded_elements,
                "padding_elements": padding,
                "padding_fraction": padding / padded_elements
                if padded_elements
                else 0.0,
                "storage_bytes": padded_elements * 4,
                "dtype": "float32",
                "input_materialization_timed": False,
                "output_preallocated": True,
            },
        }
        if provenance is not None:
            row["trace_provenance"] = provenance
        rows.append(row)
    return rows


def _compilation(rows, median):
    signatures = [
        {
            "distribution": row["distribution"],
            "value_count": row["statistics"]["total"],
            "segment_count": row["segment_count"],
            "warp_task_count": row["matched_task_partition_triton"][
                "warp_tasks"
            ],
            "cta_task_count": row["matched_task_partition_triton"]["cta_tasks"],
            "warp_programs": row["triton_fused_contract"]["warp_programs"],
        }
        for row in rows
    ]
    fixed = sorted(
        {
            (config["block"], config["num_warps"])
            for row in rows
            for config in row["triton_sweep_configs"]
        }
    )
    swage = ["swage_warp", "swage_cta", "swage_mixed"]
    triton = [f"triton_b{block}_w{warps}" for block, warps in fixed]
    triton += [
        "triton_matched_packed",
        *(f"triton_matched_cta_w{warps}" for warps in (1, 2, 4, 8)),
        "triton_fused",
    ]
    timings = {name: _measurement(median, 1) for name in swage + triton}
    totals = {"swage_total": swage, "triton_total": triton}
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
            "swage_artifacts": ["warp", "cta", "mixed"],
            "triton_fixed": [
                {"block": block, "num_warps": warps} for block, warps in fixed
            ],
            "triton_matched_cta_num_warps": [1, 2, 4, 8],
            "triton_fused_warp_programs": sorted(
                {s["warp_programs"] for s in signatures}
            ),
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
                "included": [
                    "specialization",
                    "lowering",
                    "persistent-cache write",
                ],
                "excluded": [
                    "host plan materialization",
                    "module load",
                    "launch",
                ],
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


def make_child(*, median=10.0, samples=2, warmups=1, suite="vadd"):
    """Return a complete valid child; every ordinary measurement uses median."""
    if suite not in {"vadd", "segmented-sum", "all"}:
        raise ValueError("unknown fixture suite")
    record = {
        "schema_version": 1,
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
        "methodology": {
            "warmups_per_candidate_per_measurement": warmups,
            "samples_per_candidate_per_measurement": samples,
            "candidate_sampling": "deterministic rotating/interleaved order",
            "batched_launches": 32,
            "graph_replay_launches": 32,
            "graph_capture_before_interleaved_replay": True,
            "kernel_timing_compilation_excluded": True,
            "end_to_end_compilation_excluded_after_explicit_warmup": True,
            "planning_output_preallocated": True,
            "planning_kernel_launch_excluded": True,
            "planning_compilation_excluded": True,
            "end_to_end_not_combined_with_graph_samples": True,
            "correctness_checked_before_timing": True,
            "triton_dependency": "optional runtime import; not a project dep",
        },
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
                    "swage_block": 256,
                    "swage_grid": (n + 255) // 256,
                    "triton_sweep_blocks": [128, 256, 512, 1024],
                    "launch_contract": {
                        "swage": "BLOCK=256 for the vector-add campaign",
                        "triton": "compile-time BLOCK, one program per block",
                    },
                    **_kernel_timings(candidates, median, samples),
                }
            )
    if suite in {"segmented-sum", "all"}:
        rows = copy.deepcopy(_segmented_geometry())
        for row in rows:
            candidates = [
                "swage_warp",
                "swage_cta",
                "swage_mixed",
                "torch_segment_reduce",
                "torch_padded",
                "triton_fused",
                *(
                    f"triton_b{c['block']}_w{c['num_warps']}"
                    for c in row["triton_sweep_configs"]
                ),
                "triton_matched_task_partition",
                *(f"triton_matched_task_partition_w{n}" for n in (2, 4, 8)),
            ]
            row.update(_kernel_timings(candidates, median, samples))
            row["planning"] = _phase(median, samples, planning=True)
            row["end_to_end"] = {
                "artifact_jit_warmup": (
                    "one untimed complete operation before all samples"
                ),
                "compilation_excluded": True,
                "graph_samples_combined": False,
                "warm_preparation": _phase(median, samples),
                "changing_geometry": _phase(median, samples, changing=True),
            }
        record["results"].extend(rows)
        record["compilation"] = _compilation(rows, median)
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
        "schema_version": 1,
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
