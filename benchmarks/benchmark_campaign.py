# benchmarks/benchmark_campaign.py
"""Validate source-only comparison evidence and aggregate raw CUDA samples.

Stored summaries and campaign aggregates are integrity checks, never inputs to
aggregation. This module deliberately has no optional runtime dependencies.
"""

import hashlib
import importlib.util
import json
import math
import pathlib
import re
import statistics
from datetime import datetime
from functools import lru_cache

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
_VADD_SIZES = tuple(1 << exponent for exponent in (10, 12, 14, 16, 18, 20, 22))
_ORCHESTRATION = ("swage_mixed", "triton_matched_task_partition")
_AGGREGATE_KEYS = {
    "source",
    "environment",
    "methodology",
    "agreement",
    "process_count",
    "process_level_aggregates",
}
_OBSERVATION_SCOPE = (
    "pre/post child-process boundary samples; empty samples "
    "do not prove exclusive allocation"
)


def _object(value, keys, path):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError(f"{path} must have exactly keys {sorted(keys)}")


def _list(value, path, *, nonempty=True):
    if type(value) is not list or (nonempty and not value):
        raise ValueError(
            f"{path} must be a {'nonempty ' if nonempty else ''}list"
        )


def _string(value, path):
    if type(value) is not str or not value.strip():
        raise ValueError(f"{path} must be a nonempty string")


def _strings(value, path, *, nonempty=True):
    _list(value, path, nonempty=nonempty)
    for item in value:
        _string(item, path)
    if len(value) != len(set(value)):
        raise ValueError(f"{path} contains duplicate strings")


def _integer(value, path, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{path} must be an integer >= {minimum}")


def _number(value, path, *, positive=False):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(
            f"{path} must be a finite {'positive ' if positive else ''}number"
        )


def _equal(value, expected, path):
    # JSON numeric fields permit int/float interchange only where a numeric
    # validator explicitly allows it. In particular True never stands for 1.
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{path} must equal {expected!r}")


def _timestamp(value, path):
    _string(value, path)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"{path} must be an aware ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{path} must be an aware ISO-8601 timestamp")
    return parsed


def summarize_us(samples):
    """Return median and inclusive quartiles of positive finite samples."""
    values = list(samples)
    if not values:
        raise ValueError("samples_us must not be empty")
    for value in values:
        _number(value, "samples_us", positive=True)
    if len(values) == 1:
        q1 = q3 = values[0]
    else:
        q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    summary = {"median": statistics.median(values), "q1": q1, "q3": q3}
    for value in summary.values():
        _number(value, "summary_us", positive=True)
    return summary


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant {value}")


def _parse_unique_json(data, path):
    try:
        return json.loads(
            data,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load JSON evidence {path}: {error}"
        ) from error


def load_unique_json(path):
    """Read JSON evidence, rejecting duplicate keys and nonfinite constants."""
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot load JSON evidence {path}: {error}"
        ) from error
    return _parse_unique_json(data, path)


def _measurement(value, count, path, *, graph=False):
    if graph:
        if type(value) is not dict or type(value.get("available")) is not bool:
            raise ValueError(f"{path} requires boolean graph availability")
        if not value["available"]:
            _object(value, {"available", "error"}, path)
            _string(value["error"], f"{path}.error")
            return
    keys = {"samples_us", "summary_us"}
    if graph:
        keys.add("available")
    _object(value, keys, path)
    _list(value["samples_us"], f"{path}.samples_us")
    if len(value["samples_us"]) != count:
        raise ValueError(f"{path} sample count does not match methodology")
    expected = summarize_us(value["samples_us"])
    summary = value["summary_us"]
    _object(summary, {"median", "q1", "q3"}, f"{path}.summary_us")
    for key, actual in summary.items():
        _number(actual, f"{path}.summary_us.{key}", positive=True)
    if summary != expected:
        raise ValueError(f"{path} summary_us does not match raw samples")


def _timing_method(value, candidates, count, path, *, kernel=False):
    keys = {
        "sampling",
        "base_candidate_order",
        "round_rotation",
        "timed_rounds",
        "order_position_counts",
    }
    keys |= {"graph_preparation", "units"} if kernel else {"unit"}
    _object(value, keys, path)
    _equal(value["sampling"], "deterministic_rotating_interleaved", path)
    _equal(
        value["round_rotation"],
        "left by round_index modulo candidate_count",
        path,
    )
    _equal(value["timed_rounds"], count, path)
    order = value["base_candidate_order"]
    _strings(order, f"{path}.base_candidate_order")
    if set(order) != set(candidates):
        raise ValueError(f"{path} candidate order does not match timings")
    counts = value["order_position_counts"]
    _object(counts, candidates, f"{path}.order_position_counts")
    width = len(order)
    for index, name in enumerate(order):
        _list(counts[name], f"{path}.{name}")
        for item in counts[name]:
            _integer(item, f"{path}.{name}")
        expected = [
            count // width + ((index - position) % width < count % width)
            for position in range(width)
        ]
        if counts[name] != expected:
            raise ValueError(f"{path} incorrect rotating order position counts")
    if kernel:
        _string(value["graph_preparation"], path)
        _object(value["units"], {"call", "batched_event", "graph"}, path)
        for unit in value["units"].values():
            _string(unit, path)
    else:
        _string(value["unit"], path)


def _kernel_timings(row, candidates, count, path):
    _object(row["timings"], candidates, f"{path}.timings")
    for name, metrics in row["timings"].items():
        _object(metrics, {"call", "batched_event", "graph"}, f"{path}.{name}")
        for metric, measurement in metrics.items():
            _measurement(
                measurement,
                count,
                f"{path}/{name}/{metric}",
                graph=metric == "graph",
            )
    _timing_method(
        row["timing_method"],
        candidates,
        count,
        f"{path}.timing_method",
        kernel=True,
    )


def _phase(value, count, path, *, planning=False):
    keys = {"geometry", "included", "timing_method", "timings"}
    if planning:
        keys.add("excluded")
    _object(value, keys, path)
    _string(value["geometry"], path)
    _strings(value["included"], f"{path}.included")
    if planning:
        _strings(value["excluded"], f"{path}.excluded")
        if set(value["included"]) & set(value["excluded"]):
            raise ValueError(f"{path} included/excluded scopes overlap")
    _object(value["timings"], _ORCHESTRATION, f"{path}.timings")
    for name, measurement in value["timings"].items():
        _measurement(measurement, count, f"{path}/{name}")
    _timing_method(value["timing_method"], _ORCHESTRATION, count, path)


def _configs(maximum):
    return [
        {"block": block, "num_warps": warps}
        for block in (32, 64, 128, 256, 512, 1024, 2048, 4096)
        if block >= maximum
        for warps in (1, 2, 4, 8)
        if warps <= block // 32
    ]


def _config_list(value, expected, path):
    _list(value, path)
    for config in value:
        _object(config, {"block", "num_warps"}, path)
        _integer(config["block"], path, minimum=1)
        _integer(config["num_warps"], path, minimum=1)
    if value != expected:
        raise ValueError(f"{path} does not match declared configurations")


@lru_cache(maxsize=1)
def _trace_reference():
    path = pathlib.Path(__file__).with_name("real_traces.py")
    spec = importlib.util.spec_from_file_location("_campaign_real_traces", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, provenance = module.load_real_trace(_DISTRIBUTIONS[-1])
    return module, provenance


def _trace_provenance(row):
    module, expected = _trace_reference()
    provenance = row["trace_provenance"]
    module._validate_schema(
        provenance, module._PROVENANCE_SCHEMA, "trace_provenance"
    )
    if provenance != expected:
        raise ValueError("trace provenance differs from frozen trace pins")
    derived = provenance["derived_trace"]
    stats = row["statistics"]
    if any(
        stats[key] != derived["sum" if key == "total" else key] for key in stats
    ):
        raise ValueError("trace provenance statistics do not match result")


def _segmented_row(row, count):
    path = f"segmented-sum/{row.get('distribution')}"
    keys = {
        "case",
        "distribution",
        "segment_count",
        "statistics",
        "triton_sweep_configs",
        "matched_task_partition_triton",
        "triton_fused_contract",
        "timing_method",
        "timings",
        "planning",
        "end_to_end",
        "padded_layout",
    }
    if row.get("distribution") == _DISTRIBUTIONS[-1]:
        keys.add("trace_provenance")
    _object(row, keys, path)
    if row["distribution"] not in _DISTRIBUTIONS:
        raise ValueError(f"{path} unknown distribution")
    _equal(row["segment_count"], 32_768, path)
    stats = row["statistics"]
    _object(stats, {"count", "total", "min", "median", "p95", "max"}, path)
    for key in ("count", "total", "min", "p95", "max"):
        _integer(stats[key], f"{path}.statistics.{key}")
    _number(stats["median"], path)
    if not (
        stats["count"] == row["segment_count"]
        and 0
        <= stats["min"]
        <= stats["median"]
        <= stats["p95"]
        <= stats["max"]
        <= 4096
        and stats["min"] * stats["count"]
        <= stats["total"]
        <= stats["max"] * stats["count"]
    ):
        raise ValueError(f"{path} inconsistent length statistics")
    configs = _configs(stats["max"])
    _config_list(row["triton_sweep_configs"], configs, path)
    partition = row["matched_task_partition_triton"]
    _object(
        partition,
        {
            "comparison",
            "launches",
            "warp_threshold_elements",
            "warp_tasks",
            "cta_tasks",
            "short_tasks_per_program",
            "cta_block_elements",
            "primary_cta_num_warps",
            "cta_num_warps_sweep",
        },
        path,
    )
    for key in ("comparison", "launches"):
        _string(partition[key], path)
    for key, expected in {
        "warp_threshold_elements": 32,
        "short_tasks_per_program": 4,
        "cta_block_elements": 4096,
        "primary_cta_num_warps": 1,
    }.items():
        _equal(partition[key], expected, path)
    _integer_list(partition["cta_num_warps_sweep"], [1, 2, 4, 8], path)
    for key in ("warp_tasks", "cta_tasks"):
        _integer(partition[key], path)
    if partition["warp_tasks"] + partition["cta_tasks"] != stats["count"]:
        raise ValueError(f"{path} partition count mismatch")
    if (
        (stats["max"] <= 32 and partition["cta_tasks"] != 0)
        or (stats["max"] > 32 and partition["cta_tasks"] == 0)
        or (stats["min"] > 32 and partition["warp_tasks"] != 0)
    ):
        raise ValueError(f"{path} partition disagrees with length bounds")
    fused = row["triton_fused_contract"]
    _object(
        fused,
        {
            "result_name",
            "launch_count",
            "logical_lanes_per_program",
            "short_task_slots",
            "lanes_per_short_task",
            "program_order",
            "cta_accumulation",
            "maximum_segment_length",
            "physical_num_warps",
            "warp_programs",
            "cta_programs",
            "grid_programs",
        },
        path,
    )
    warp_programs = (partition["warp_tasks"] + 3) // 4
    for key, expected in {
        "result_name": "triton_fused",
        "launch_count": 1,
        "logical_lanes_per_program": 128,
        "short_task_slots": 4,
        "lanes_per_short_task": 32,
        "maximum_segment_length": 4096,
        "physical_num_warps": 4,
        "warp_programs": warp_programs,
        "cta_programs": partition["cta_tasks"],
        "grid_programs": warp_programs + partition["cta_tasks"],
    }.items():
        _equal(fused[key], expected, path)
    for key in ("program_order", "cta_accumulation"):
        _string(fused[key], path)
    padded = row["padded_layout"]
    _object(
        padded,
        {
            "rows",
            "columns",
            "packed_elements",
            "padded_elements",
            "padding_elements",
            "padding_fraction",
            "storage_bytes",
            "dtype",
            "input_materialization_timed",
            "output_preallocated",
        },
        path,
    )
    elements = stats["count"] * stats["max"]
    for key, expected in {
        "rows": stats["count"],
        "columns": stats["max"],
        "packed_elements": stats["total"],
        "padded_elements": elements,
        "padding_elements": elements - stats["total"],
        "storage_bytes": elements * 4,
        "dtype": "float32",
        "input_materialization_timed": False,
        "output_preallocated": True,
    }.items():
        _equal(padded[key], expected, f"{path}.padded_layout.{key}")
    fraction = (elements - stats["total"]) / elements if elements else 0.0
    _number(padded["padding_fraction"], path)
    if padded["padding_fraction"] != fraction:
        raise ValueError(f"{path} padding_fraction mismatch")
    candidates = {
        "swage_warp",
        "swage_cta",
        "swage_mixed",
        "torch_segment_reduce",
        "torch_padded",
        "triton_fused",
        "triton_matched_task_partition",
        *(f"triton_matched_task_partition_w{n}" for n in (2, 4, 8)),
        *(f"triton_b{c['block']}_w{c['num_warps']}" for c in configs),
    }
    _kernel_timings(row, candidates, count, path)
    _phase(row["planning"], count, f"{path}.planning", planning=True)
    end = row["end_to_end"]
    _object(
        end,
        {
            "artifact_jit_warmup",
            "compilation_excluded",
            "graph_samples_combined",
            "warm_preparation",
            "changing_geometry",
        },
        path,
    )
    _string(end["artifact_jit_warmup"], path)
    _equal(end["compilation_excluded"], True, path)
    _equal(end["graph_samples_combined"], False, path)
    for mode in ("warm_preparation", "changing_geometry"):
        _phase(end[mode], count, f"{path}.{mode}")
    if "trace_provenance" in row:
        _trace_provenance(row)


def _integer_list(value, expected, path):
    _list(value, path)
    for item in value:
        _integer(item, path)
    if value != expected:
        raise ValueError(f"{path} must equal {expected!r}")


def _vadd_row(row, count):
    path = f"vadd/{row.get('n')}"
    _object(
        row,
        {
            "case",
            "n",
            "swage_block",
            "swage_grid",
            "triton_sweep_blocks",
            "launch_contract",
            "timing_method",
            "timings",
        },
        path,
    )
    _integer(row["n"], path, minimum=1)
    if row["n"] not in _VADD_SIZES:
        raise ValueError(f"{path} undeclared vector size")
    _equal(row["swage_block"], 256, path)
    _equal(row["swage_grid"], (row["n"] + 255) // 256, path)
    _integer_list(row["triton_sweep_blocks"], [128, 256, 512, 1024], path)
    _object(row["launch_contract"], {"swage", "triton"}, path)
    for value in row["launch_contract"].values():
        _string(value, path)
    candidates = {
        "swage",
        "torch",
        *(f"triton_b{n}" for n in (128, 256, 512, 1024)),
    }
    _kernel_timings(row, candidates, count, path)


def _compilation(value, rows):
    if not rows:
        _object(value, {"status", "reason"}, "compilation")
        _equal(value["status"], "not-run", "compilation.status")
        _equal(
            value["reason"],
            "segmented-sum suite not selected",
            "compilation.reason",
        )
        return
    _object(
        value,
        {
            "status",
            "case",
            "timing_method",
            "compiler_order",
            "component_order",
            "configurations",
            "derived_totals",
            "cache_policy",
            "scope",
            "not_applicable",
            "timings",
        },
        "compilation",
    )
    _equal(value["status"], "measured", "compilation.status")
    _equal(value["case"], "segmented-sum", "compilation.case")
    _string(value["timing_method"], "compilation.timing_method")
    _equal(
        value["compiler_order"],
        ["swage", "triton"],
        "compilation.compiler_order",
    )
    cache = value["cache_policy"]
    _object(
        cache,
        {
            "fresh_process",
            "unique_directories",
            "swage_initially_empty",
            "triton_initially_empty",
        },
        "compilation.cache_policy",
    )
    for item in cache.values():
        _equal(item, True, "compilation.cache_policy")
    configs = value["configurations"]
    _object(
        configs,
        {
            "swage_artifacts",
            "triton_fixed",
            "triton_matched_cta_num_warps",
            "triton_fused_warp_programs",
            "triton_case_signatures",
        },
        "compilation.configurations",
    )
    _equal(
        configs["swage_artifacts"], ["warp", "cta", "mixed"], "swage_artifacts"
    )
    fixed = sorted(
        {
            (config["block"], config["num_warps"])
            for row in rows
            for config in row["triton_sweep_configs"]
        }
    )
    _config_list(
        configs["triton_fixed"],
        [{"block": block, "num_warps": warps} for block, warps in fixed],
        "compilation.triton_fixed",
    )
    _integer_list(
        configs["triton_matched_cta_num_warps"], [1, 2, 4, 8], "matched warps"
    )
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
    actual = configs["triton_case_signatures"]
    _list(actual, "triton_case_signatures")
    for signature in actual:
        _object(signature, signatures[0], "triton_case_signatures")
        _string(signature["distribution"], "triton_case_signatures")
        for key in set(signature) - {"distribution"}:
            _integer(signature[key], f"triton_case_signatures.{key}")
    if actual != signatures:
        raise ValueError(
            "compilation case signatures do not match result geometry"
        )
    _integer_list(
        configs["triton_fused_warp_programs"],
        sorted({row["warp_programs"] for row in signatures}),
        "fused warp programs",
    )
    swage = ["swage_warp", "swage_cta", "swage_mixed"]
    triton = [f"triton_b{block}_w{warps}" for block, warps in fixed]
    if any(row["warp_task_count"] for row in signatures):
        triton.append("triton_matched_packed")
    if any(row["cta_task_count"] for row in signatures):
        triton.extend(f"triton_matched_cta_w{warps}" for warps in (1, 2, 4, 8))
    triton.append("triton_fused")
    _equal(
        value["component_order"], swage + triton, "compilation.component_order"
    )
    totals = {"swage_total": swage, "triton_total": triton}
    _object(value["derived_totals"], totals, "compilation.derived_totals")
    if value["derived_totals"] != totals:
        raise ValueError("compilation derived_totals component mismatch")
    timings = value["timings"]
    _object(timings, {*swage, *triton, *totals}, "compilation.timings")
    for name, measurement in timings.items():
        _measurement(measurement, 1, f"compilation/{name}")
    for name, components in totals.items():
        expected = sum(timings[part]["samples_us"][0] for part in components)
        if timings[name]["samples_us"] != [expected]:
            raise ValueError(
                f"compilation {name} is not the exact component sum"
            )
    _object(value["scope"], {"swage", "triton"}, "compilation.scope")
    for compiler, scope in value["scope"].items():
        _object(scope, {"included", "excluded"}, f"scope.{compiler}")
        for key, descriptions in scope.items():
            _strings(descriptions, f"scope.{compiler}.{key}")
        if set(scope["included"]) & set(scope["excluded"]):
            raise ValueError(
                f"compilation {compiler} included/excluded scopes overlap"
            )
    _object(
        value["not_applicable"],
        {"torch_segment_reduce", "torch_padded"},
        "not_applicable",
    )
    for reason in value["not_applicable"].values():
        _string(reason, "not_applicable")


def validate_child(record):
    """Reject malformed schema-v1 comparison children; return no value."""
    _object(
        record,
        {
            "schema_version",
            "benchmark",
            "recorded_at",
            "source",
            "environment",
            "methodology",
            "compilation",
            "results",
        },
        "child",
    )
    _equal(record["schema_version"], 1, "child.schema_version")
    _equal(record["benchmark"], "swage-triton-comparison", "child.benchmark")
    _timestamp(record["recorded_at"], "child.recorded_at")
    source = record["source"]
    _object(source, {"revision", "worktree_clean", "dirty"}, "source")
    _string(source["revision"], "source.revision")
    if not re.fullmatch(r"[0-9a-f]{40}", source["revision"]):
        raise ValueError(
            "source.revision must be 40 lowercase hexadecimal characters"
        )
    _strings(source["dirty"], "source.dirty", nonempty=False)
    _equal(
        source["worktree_clean"], not source["dirty"], "source.worktree_clean"
    )
    environment = record["environment"]
    _object(
        environment,
        {
            "platform",
            "python",
            "pytorch",
            "pytorch_cuda",
            "triton",
            "cuda_driver",
            "gpu",
            "compute_capability",
            "multiprocessors",
            "total_memory_bytes",
            "compiler",
        },
        "environment",
    )
    for key in set(environment) - {
        "compiler",
        "multiprocessors",
        "total_memory_bytes",
    }:
        _string(environment[key], f"environment.{key}")
    if not re.fullmatch(r"sm_[0-9]{2,3}", environment["compute_capability"]):
        raise ValueError("environment.compute_capability is invalid")
    for key in ("multiprocessors", "total_memory_bytes"):
        _integer(environment[key], f"environment.{key}", minimum=1)
    compiler = environment["compiler"]
    _object(
        compiler,
        {
            "package_version",
            "source_revision",
            "source_clean",
            "llvm_version",
            "llvm_pin",
            "build_type",
        },
        "environment.compiler",
    )
    for key in set(compiler) - {"source_clean"}:
        _string(compiler[key], f"compiler.{key}")
    _equal(
        compiler["source_revision"],
        source["revision"],
        "compiler.source_revision",
    )
    _equal(
        compiler["source_clean"],
        source["worktree_clean"],
        "compiler.source_clean",
    )
    _equal(
        compiler["llvm_version"], compiler["llvm_pin"], "compiler.llvm_version"
    )
    method = record["methodology"]
    flags = {
        "graph_capture_before_interleaved_replay",
        "kernel_timing_compilation_excluded",
        "end_to_end_compilation_excluded_after_explicit_warmup",
        "planning_output_preallocated",
        "planning_kernel_launch_excluded",
        "planning_compilation_excluded",
        "end_to_end_not_combined_with_graph_samples",
        "correctness_checked_before_timing",
    }
    _object(
        method,
        {
            "warmups_per_candidate_per_measurement",
            "samples_per_candidate_per_measurement",
            "candidate_sampling",
            "batched_launches",
            "graph_replay_launches",
            "triton_dependency",
            *flags,
        },
        "methodology",
    )
    for flag in flags:
        _equal(method[flag], True, f"methodology.{flag}")
    _integer(method["warmups_per_candidate_per_measurement"], "warmups")
    count = method["samples_per_candidate_per_measurement"]
    _integer(count, "samples", minimum=1)
    for key in ("batched_launches", "graph_replay_launches"):
        _equal(method[key], 32, f"methodology.{key}")
    _equal(
        method["candidate_sampling"],
        "deterministic rotating/interleaved order",
        "candidate_sampling",
    )
    _string(method["triton_dependency"], "triton_dependency")
    _list(record["results"], "results")
    vadd, segmented = [], []
    for row in record["results"]:
        if type(row) is not dict:
            raise ValueError("result must be an object")
        if row.get("case") == "vadd":
            _vadd_row(row, count)
            vadd.append(row)
        elif row.get("case") == "segmented-sum":
            _segmented_row(row, count)
            segmented.append(row)
        else:
            raise ValueError("unrecognized result case")
    if vadd and [row["n"] for row in vadd] != list(_VADD_SIZES):
        raise ValueError("vadd cases must be the unique declared size sequence")
    if segmented and [row["distribution"] for row in segmented] != list(
        _DISTRIBUTIONS
    ):
        raise ValueError(
            "segmented cases must be the unique declared distribution sequence"
        )
    if record["results"] != vadd + segmented:
        raise ValueError("result suite order must be vadd then segmented-sum")
    _compilation(record["compilation"], segmented)


def _raw_process_medians(child):
    medians = {}
    compilation = child["compilation"]
    if compilation["status"] == "measured":
        for name, measurement in compilation["timings"].items():
            medians[f"segmented-sum/{name}/compilation_us"] = statistics.median(
                measurement["samples_us"]
            )
    for row in child["results"]:
        case = (
            f"vadd/n={row['n']}"
            if row["case"] == "vadd"
            else f"segmented-sum/distribution={row['distribution']}"
        )
        for name, metrics in row["timings"].items():
            for metric, measurement in metrics.items():
                if measurement.get("available") is False:
                    continue
                medians[f"{case}/{name}/{metric}_us"] = statistics.median(
                    measurement["samples_us"]
                )
        if row["case"] == "segmented-sum":
            for name, measurement in row["planning"]["timings"].items():
                medians[f"{case}/{name}/planning_us"] = statistics.median(
                    measurement["samples_us"]
                )
            for mode in ("warm_preparation", "changing_geometry"):
                for name, measurement in row["end_to_end"][mode][
                    "timings"
                ].items():
                    medians[f"{case}/{name}/end_to_end_{mode}_us"] = (
                        statistics.median(measurement["samples_us"])
                    )
    return medians


def process_medians(child):
    """Validate a child and flatten medians computed only from raw samples."""
    validate_child(child)
    return _raw_process_medians(child)


def _metadata(value):
    if type(value) is dict:
        return {
            key: _metadata(item)
            for key, item in value.items()
            if key not in {"samples_us", "summary_us"}
        }
    if type(value) is list:
        return [_metadata(item) for item in value]
    return value


def aggregate_children(children):
    """Require exact-source agreement and aggregate independent raw medians."""
    _list(children, "children")
    for child in children:
        validate_child(child)
        if not child["source"]["worktree_clean"]:
            raise ValueError(
                "campaign children must report clean source metadata"
            )
    reference = children[0]
    flattened = []
    for index, child in enumerate(children):
        for key in ("source", "environment", "methodology"):
            if child[key] != reference[key]:
                raise ValueError(f"child {index} {key} metadata differs")
        for key in ("results", "compilation"):
            if _metadata(child[key]) != _metadata(reference[key]):
                raise ValueError(f"child {index} {key} metadata differs")
        flattened.append(_raw_process_medians(child))
    aggregates = [
        {
            "measurement": metric,
            "unit": "microseconds",
            "child_process_medians_us": [child[metric] for child in flattened],
            "median_of_process_medians_us": statistics.median(
                child[metric] for child in flattened
            ),
        }
        for metric in sorted(flattened[0])
    ]
    return {
        "source": reference["source"],
        "environment": reference["environment"],
        "methodology": reference["methodology"],
        "agreement": {
            "source": True,
            "environment": True,
            "methodology": True,
            "result_metadata": True,
            "compilation_metadata": True,
        },
        "process_count": len(children),
        "process_level_aggregates": aggregates,
    }


def _unavailable(value, path):
    _object(value, {"available", "reason"}, path)
    _equal(value["available"], False, path)
    _string(value["reason"], f"{path}.reason")


def _telemetry(value):
    if type(value) is not dict or type(value.get("available")) is not bool:
        raise ValueError("NVIDIA telemetry requires boolean available")
    if not value["available"]:
        _unavailable(value, "NVIDIA telemetry")
        return
    _object(
        value,
        {"available", "recorded_at", "fields", "gpus", "compute_processes"},
        "NVIDIA telemetry",
    )
    _timestamp(value["recorded_at"], "NVIDIA telemetry.recorded_at")
    fields = {
        "sm_clock_mhz": "MHz",
        "memory_clock_mhz": "MHz",
        "power_draw_watts": "W",
        "temperature_celsius": "degrees Celsius",
    }
    _object(value["fields"], fields, "NVIDIA telemetry.fields")
    _equal(value["fields"], fields, "NVIDIA telemetry.fields")
    _list(value["gpus"], "NVIDIA telemetry.gpus")
    indices, uuids = set(), set()
    for gpu in value["gpus"]:
        _object(gpu, {"index", "uuid", *fields}, "NVIDIA GPU")
        for key in ("index", "sm_clock_mhz", "memory_clock_mhz"):
            if gpu[key] is not None:
                _integer(gpu[key], f"NVIDIA GPU.{key}")
        for key in ("power_draw_watts", "temperature_celsius"):
            if gpu[key] is not None:
                _number(gpu[key], f"NVIDIA GPU.{key}")
                if gpu[key] < 0:
                    raise ValueError(f"NVIDIA GPU.{key} must be nonnegative")
        if gpu["uuid"] is not None:
            _string(gpu["uuid"], "NVIDIA GPU.uuid")
            if gpu["uuid"] in uuids:
                raise ValueError("duplicate NVIDIA GPU uuid")
            uuids.add(gpu["uuid"])
        if gpu["index"] is not None:
            if gpu["index"] in indices:
                raise ValueError("duplicate NVIDIA GPU index")
            indices.add(gpu["index"])
    processes = value["compute_processes"]
    if (
        type(processes) is not dict
        or type(processes.get("available")) is not bool
    ):
        raise ValueError("compute-process telemetry requires boolean available")
    if not processes["available"]:
        _unavailable(processes, "compute-process telemetry")
        return
    _object(
        processes,
        {"available", "fields", "processes"},
        "compute-process telemetry",
    )
    _equal(
        processes["fields"],
        {"used_memory_mib": "MiB"},
        "compute-process fields",
    )
    _list(processes["processes"], "compute processes", nonempty=False)
    seen = set()
    for process in processes["processes"]:
        _object(
            process,
            {"gpu_uuid", "pid", "process_name", "used_memory_mib"},
            "compute process",
        )
        for key in ("gpu_uuid", "process_name"):
            _string(process[key], f"compute process.{key}")
        _integer(process["pid"], "compute process.pid", minimum=1)
        if process["used_memory_mib"] is not None:
            _integer(
                process["used_memory_mib"], "compute process.used_memory_mib"
            )
        identity = (process["gpu_uuid"], process["pid"])
        if identity in seen:
            raise ValueError("duplicate NVIDIA compute process")
        seen.add(identity)


def compute_process_reasons(telemetry, boundary):
    """Validate stored NVIDIA telemetry and report contention evidence."""
    _string(boundary, "boundary")
    _telemetry(telemetry)
    processes = telemetry.get("compute_processes", telemetry)
    if not processes["available"]:
        return [
            f"{boundary}: compute-process telemetry unavailable "
            f"({processes['reason']})"
        ]
    return [
        f"{boundary}: competing compute process "
        f"gpu_uuid={process['gpu_uuid']!r} pid={process['pid']!r} "
        f"process_name={process['process_name']!r} "
        f"used_memory_mib={process['used_memory_mib']!r}"
        for process in processes["processes"]
    ]


def _stored_equal(actual, expected, path):
    """Compare stored aggregates recursively, rejecting scalar type tricks."""
    if type(expected) is dict:
        _object(actual, expected, path)
        for key in expected:
            _stored_equal(actual[key], expected[key], f"{path}.{key}")
    elif type(expected) is list:
        _list(actual, path, nonempty=False)
        if len(actual) != len(expected):
            raise ValueError(f"{path} stored aggregate drift")
        for index, (item, reference) in enumerate(zip(actual, expected)):
            _stored_equal(item, reference, f"{path}[{index}]")
    elif type(expected) in (int, float) and type(expected) is not bool:
        if type(expected) is int:
            _integer(actual, path)
        else:
            _number(actual, path)
        if actual != expected:
            raise ValueError(f"{path} stored aggregate drift")
    else:
        _equal(actual, expected, path)


def load_campaign(manifest_path, *, require_archival=True):
    """Load a strict immutable campaign and return recomputed raw evidence."""
    if type(require_archival) is not bool:
        raise ValueError("require_archival must be boolean")
    manifest_path = pathlib.Path(manifest_path)
    manifest = load_unique_json(manifest_path)
    _object(
        manifest,
        {
            "schema_version",
            "benchmark",
            "recorded_at",
            "independent_processes",
            "archival_eligible",
            "archival_ineligibility_reasons",
            "controls",
            "children",
            *_AGGREGATE_KEYS,
        },
        "manifest",
    )
    _equal(manifest["schema_version"], 1, "manifest.schema_version")
    _equal(
        manifest["benchmark"],
        "swage-triton-comparison-process-campaign",
        "manifest.benchmark",
    )
    _timestamp(manifest["recorded_at"], "manifest.recorded_at")
    _equal(manifest["independent_processes"], True, "independent_processes")
    _strings(
        manifest["archival_ineligibility_reasons"],
        "archival reasons",
        nonempty=False,
    )
    controls = manifest["controls"]
    _object(
        controls,
        {
            "suite",
            "repetitions",
            "samples_per_process",
            "warmups_per_process",
            "gpu_execution_mode",
            "exclusive_gpu_allocated",
            "allow_shared_gpu_engineering",
            "archival_source",
            "compute_process_observation_scope",
        },
        "controls",
    )
    if type(controls["suite"]) is not str or controls["suite"] not in {
        "all",
        "vadd",
        "segmented-sum",
    }:
        raise ValueError("controls.suite is invalid")
    for key in ("repetitions", "samples_per_process"):
        _integer(controls[key], f"controls.{key}", minimum=1)
    _integer(controls["warmups_per_process"], "controls.warmups_per_process")
    for key in (
        "exclusive_gpu_allocated",
        "allow_shared_gpu_engineering",
        "archival_source",
    ):
        if type(controls[key]) is not bool:
            raise ValueError(f"controls.{key} must be boolean")
    exclusive = controls["exclusive_gpu_allocated"]
    _equal(controls["allow_shared_gpu_engineering"], not exclusive, "GPU mode")
    _equal(
        controls["gpu_execution_mode"],
        "exclusive-asserted" if exclusive else "shared-gpu-engineering",
        "GPU execution mode",
    )
    _equal(
        controls["compute_process_observation_scope"],
        _OBSERVATION_SCOPE,
        "observation scope",
    )
    _list(manifest["children"], "manifest.children")
    _integer(manifest["process_count"], "manifest.process_count", minimum=1)
    count = controls["repetitions"]
    if len(manifest["children"]) != count or manifest["process_count"] != count:
        raise ValueError("campaign child/control/process count mismatch")
    expected_names = {f"process-{index:03d}.json" for index in range(count)}
    found = {path.name for path in manifest_path.parent.glob("process-*.json")}
    if found != expected_names:
        raise ValueError("campaign missing or extra process-*.json files")
    reasons = []
    if not exclusive:
        reasons.append(
            "shared-GPU engineering mode selected; "
            "exclusive GPU allocation was not asserted"
        )
    if not controls["archival_source"]:
        reasons.append("archival source revision was not asserted")
    children, identities = [], set()
    for index, entry in enumerate(manifest["children"]):
        _object(
            entry,
            {
                "process_index",
                "path",
                "sha256",
                "recorded_at",
                "nvidia_telemetry",
            },
            "child entry",
        )
        _equal(entry["process_index"], index, "child process_index")
        _equal(entry["path"], f"process-{index:03d}.json", "child path")
        path = manifest_path.parent / entry["path"]
        if path.is_symlink() or not path.is_file():
            raise ValueError("child path must be a regular non-symlink file")
        identity = (path.stat().st_dev, path.stat().st_ino)
        if identity in identities:
            raise ValueError("child paths must name independent files")
        identities.add(identity)
        _string(entry["sha256"], "child sha256")
        if not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]):
            raise ValueError("child sha256 is invalid")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"child {index} SHA-256 mismatch")
        _timestamp(entry["recorded_at"], "child entry.recorded_at")
        child = _parse_unique_json(data, path)
        validate_child(child)
        _equal(entry["recorded_at"], child["recorded_at"], "child recorded_at")
        method = child["methodology"]
        for control, field in (
            ("samples_per_process", "samples_per_candidate_per_measurement"),
            ("warmups_per_process", "warmups_per_candidate_per_measurement"),
        ):
            _equal(controls[control], method[field], f"controls.{control}")
        suites = {row["case"] for row in child["results"]}
        expected = (
            {"vadd", "segmented-sum"}
            if controls["suite"] == "all"
            else {controls["suite"]}
        )
        if suites != expected:
            raise ValueError("control suite does not match child results")
        telemetry = entry["nvidia_telemetry"]
        _object(telemetry, {"pre_process", "post_process"}, "child telemetry")
        for mode in ("pre", "post"):
            reasons.extend(
                compute_process_reasons(
                    telemetry[f"{mode}_process"],
                    f"process {index} {mode}-process boundary",
                )
            )
        children.append(child)
    aggregate = aggregate_children(children)
    for key in _AGGREGATE_KEYS:
        _stored_equal(manifest[key], aggregate[key], f"manifest.{key}")
    if aggregate["environment"]["compiler"]["build_type"] != "Release":
        reasons.append("compiler build type is not Release")
    if manifest["archival_ineligibility_reasons"] != reasons:
        raise ValueError(
            "stored archival reasons do not match controls and telemetry"
        )
    _equal(manifest["archival_eligible"], not reasons, "archival_eligible")
    if require_archival and reasons:
        raise ValueError(
            f"campaign is not archival eligible: {'; '.join(reasons)}"
        )
    return {
        "manifest": manifest,
        "children": children,
        "child_process_medians": [
            _raw_process_medians(child) for child in children
        ],
        "recomputed_aggregate": aggregate,
    }
