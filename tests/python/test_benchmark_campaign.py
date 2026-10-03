# tests/python/test_benchmark_campaign.py
"""Regression coverage for raw comparison evidence and archival integrity."""

import copy
import hashlib
import importlib.util
import json
import pathlib
import statistics

import pytest
from benchmark_campaign_fixtures import (
    make_child,
    make_telemetry,
    write_campaign,
)


@pytest.fixture(scope="module")
def campaign():
    """Load only the standard-library evidence parser."""
    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "benchmark_campaign.py"
    )
    spec = importlib.util.spec_from_file_location("campaign_schema_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def _measurement(child):
    return child["results"][0]["timings"]["swage"]["call"]


def _replace_samples(measurement, samples):
    q1, _, q3 = statistics.quantiles(samples, n=4, method="inclusive")
    measurement.update(
        {
            "samples_us": samples,
            "summary_us": {
                "median": statistics.median(samples),
                "q1": q1,
                "q3": q3,
            },
        }
    )


def test_inclusive_quartiles_and_single_sample(campaign):
    """Inclusive quartiles distinguish the declared estimator."""
    assert campaign.summarize_us([9.0]) == {"median": 9.0, "q1": 9.0, "q3": 9.0}
    assert campaign.summarize_us([9, 1, 3, 5]) == {
        "median": 4.0,
        "q1": 2.5,
        "q3": 6.0,
    }


@pytest.mark.parametrize(
    "samples", [[], [0], [-1], [True], [float("nan")], [float("inf")]]
)
def test_summaries_reject_invalid_raw_evidence(campaign, samples):
    """Timing evidence excludes empty, nonnumeric, and nonpositive samples."""
    with pytest.raises(ValueError):
        campaign.summarize_us(samples)


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version":1,"schema_version":1}',
        '{"outer":{"samples_us":[1],"samples_us":[2]}}',
        '{"sample":NaN}',
        '{"sample":Infinity}',
    ],
)
def test_json_rejects_duplicate_keys_and_nonfinite_constants(
    campaign, tmp_path, text
):
    """Ambiguous or nonfinite JSON must not enter the evidence model."""
    path = tmp_path / "record.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        campaign.load_unique_json(path)


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("unexpected",), 1),
        (("recorded_at",), "2026-09-08T12:00:00"),
        (("source", "revision"), "A" * 40),
        (("environment", "compiler", "source_revision"), "b" * 40),
        (("environment", "compiler", "source_clean"), 1),
        (("environment", "compiler", "llvm_pin"), "llvmorg-other"),
        (("environment", "compiler", "package_version"), None),
        (("environment", "multiprocessors"), True),
        (("methodology", "samples_per_candidate_per_measurement"), 3),
        (("methodology", "planning_compilation_excluded"), False),
        (("results", 0, "swage_block"), True),
        (("results", 0, "launch_contract", "extra"), "unexpected"),
        (("results", 0, "timings", "swage", "call", "summary_us", "q1"), 9.0),
        (
            ("results", 0, "timings", "swage", "call", "samples_us"),
            [True, 10.0],
        ),
        (("results", 0, "timings", "swage", "call", "samples_us"), [0.0, 20.0]),
        (("results", 0, "timings", "swage", "call", "extra"), 1),
        (("results", 0, "timings", "swage", "unexpected_metric"), {}),
        (
            ("results", 0, "timing_method", "order_position_counts", "swage"),
            [True, 0, 0, 0, 0, 1],
        ),
    ],
)
def test_child_schema_rejects_type_shape_and_summary_corruption(
    campaign, path, value
):
    """Exact schemas and raw summaries reject independently forged fields."""
    child = make_child()
    target = child
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        campaign.validate_child(child)


def test_graph_variants_and_missing_candidate_fail_closed(campaign):
    """Unavailable graphs omit metrics but never hide required candidates."""
    child = make_child()
    metrics = child["results"][0]["timings"]["swage"]
    metrics["graph"] = {"available": False, "error": "capture unsupported"}
    medians = campaign.process_medians(child)
    assert "vadd/n=1024/swage/graph_us" not in medians
    assert medians["vadd/n=1024/swage/batched_event_us"] == 10.0
    metrics["graph"]["samples_us"] = [1.0, 1.0]
    with pytest.raises(ValueError):
        campaign.validate_child(child)
    del metrics["graph"]["samples_us"]
    del child["results"][0]["timings"]["torch"]
    with pytest.raises(ValueError):
        campaign.validate_child(child)


def test_duplicate_cases_and_configurations_are_rejected(campaign):
    """Repeated cases and configs cannot silently replace measurements."""
    child = make_child()
    child["results"][1] = copy.deepcopy(child["results"][0])
    with pytest.raises(ValueError, match="unique declared"):
        campaign.validate_child(child)
    child = make_child(suite="segmented-sum")
    child["results"][0]["triton_sweep_configs"].append(
        copy.deepcopy(child["results"][0]["triton_sweep_configs"][0])
    )
    with pytest.raises(ValueError, match="configurations"):
        campaign.validate_child(child)


def test_process_aggregation_uses_median_of_raw_process_medians(campaign):
    """Process aggregation must not pool samples across independent runs."""
    children = [make_child(samples=3) for _ in range(3)]
    for child, values in zip(
        children, ([1, 1, 100], [2, 2, 200], [3, 300, 300])
    ):
        _replace_samples(_measurement(child), values)
    aggregate = campaign.aggregate_children(children)
    metric = next(
        metric
        for metric in aggregate["process_level_aggregates"]
        if metric["measurement"] == "vadd/n=1024/swage/call_us"
    )
    assert metric["child_process_medians_us"] == [1, 2, 300]
    assert metric["median_of_process_medians_us"] == 2
    assert statistics.median([1, 1, 100, 2, 2, 200, 3, 300, 300]) == 3


def test_dirty_exact_source_child_is_not_a_campaign(campaign):
    """A truthful dirty child is inspectable but cannot form a campaign."""
    child = make_child()
    child["source"].update(worktree_clean=False, dirty=[" M source.py"])
    child["environment"]["compiler"]["source_clean"] = False
    campaign.validate_child(child)
    with pytest.raises(ValueError, match="clean source"):
        campaign.aggregate_children([child])


@pytest.mark.parametrize(
    "section",
    ["source", "environment", "methodology", "results", "compilation"],
)
def test_child_agreement_preserves_all_nontiming_metadata(campaign, section):
    """Only timing payloads may differ among independent child records."""
    children = [make_child(suite="segmented-sum") for _ in range(2)]
    changed = children[1]
    if section == "source":
        changed["source"]["revision"] = "b" * 40
        changed["environment"]["compiler"]["source_revision"] = "b" * 40
    elif section == "environment":
        changed["environment"]["cuda_driver"] = "13.1"
    elif section == "methodology":
        changed["methodology"]["warmups_per_candidate_per_measurement"] = 2
    elif section == "results":
        changed["results"][0]["planning"]["geometry"] = (
            "different preparation scope"
        )
    else:
        changed["compilation"]["scope"]["swage"]["excluded"].append(
            "additional excluded phase"
        )
    with pytest.raises(ValueError, match="metadata differs"):
        campaign.aggregate_children(children)


def test_compilation_planning_and_kernel_metrics_remain_separate(campaign):
    """Consumers can distinguish compile, planning, and dispatch costs."""
    child = make_child(suite="segmented-sum")
    medians = campaign.process_medians(child)
    prefix = "segmented-sum/distribution=many-tiny/swage_mixed"
    assert medians["segmented-sum/swage_total/compilation_us"] == 30.0
    assert medians[f"{prefix}/planning_us"] == 10.0
    assert medians[f"{prefix}/batched_event_us"] == 10.0
    assert medians[f"{prefix}/end_to_end_warm_preparation_us"] == 10.0
    assert not any("preparation_only" in name for name in medians)


def test_compilation_total_must_equal_raw_component_sum(campaign):
    """A self-consistent summary cannot disguise an incorrect total."""
    child = make_child(suite="segmented-sum")
    total = child["compilation"]["timings"]["swage_total"]
    total.update(
        samples_us=[31.0], summary_us={"median": 31.0, "q1": 31.0, "q3": 31.0}
    )
    with pytest.raises(ValueError, match="exact component sum"):
        campaign.validate_child(child)


@pytest.mark.parametrize(
    "path,value",
    [
        (
            (
                "compilation",
                "configurations",
                "triton_case_signatures",
                0,
                "value_count",
            ),
            1,
        ),
        (
            ("compilation", "configurations", "triton_fused_warp_programs"),
            [True],
        ),
        (("compilation", "cache_policy", "swage_initially_empty"), False),
        (("compilation", "timings", "swage_warp", "samples_us"), [10.0, 10.0]),
        (
            (
                "results",
                0,
                "planning",
                "compilation_excluded_after_explicit_warmup",
            ),
            True,
        ),
        (("results", 0, "padded_layout", "storage_bytes"), 0),
        (("results", 0, "padded_layout", "padding_fraction"), True),
        (("results", 0, "matched_task_partition_triton", "cta_tasks"), 1),
        (("results", 0, "triton_fused_contract", "grid_programs"), 1),
        (
            ("results", 7, "trace_provenance", "license", "identifier"),
            "unknown",
        ),
        (("results", 7, "trace_provenance", "source", "unexpected"), "unknown"),
    ],
)
def test_segmented_schema_rejects_phase_geometry_and_provenance_drift(
    campaign, path, value
):
    """Geometry, phase boundaries, and pinned provenance are integrity data."""
    child = make_child(suite="segmented-sum")
    target = child
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        campaign.validate_child(child)


def test_zero_column_padding_has_zero_storage_and_fraction(campaign):
    """Every-empty storage is representable without dividing by zero."""
    child = make_child(suite="segmented-sum")
    row = child["results"][0]
    row["statistics"].update(total=0, min=0, median=0.0, p95=0, max=0)
    row["padded_layout"].update(
        columns=0,
        packed_elements=0,
        padded_elements=0,
        padding_elements=0,
        padding_fraction=0.0,
        storage_bytes=0,
    )
    child["compilation"]["configurations"]["triton_case_signatures"][0][
        "value_count"
    ] = 0
    campaign.validate_child(child)
    row["padded_layout"]["padding_fraction"] = 1.0
    with pytest.raises(ValueError, match="padding_fraction"):
        campaign.validate_child(child)


def test_complete_campaign_exposes_raw_medians_and_recomputed_aggregate(
    campaign, tmp_path
):
    """A complete archive exposes independently recomputable chart inputs."""
    path = write_campaign(tmp_path)
    loaded = campaign.load_campaign(path)
    assert set(loaded) == {
        "manifest",
        "children",
        "child_process_medians",
        "recomputed_aggregate",
    }
    metric = (
        "segmented-sum/distribution=many-tiny/torch_padded/batched_event_us"
    )
    raw_medians = [
        statistics.median(
            child["results"][0]["timings"]["torch_padded"]["batched_event"][
                "samples_us"
            ]
        )
        for child in loaded["children"]
    ]
    assert [
        medians[metric] for medians in loaded["child_process_medians"]
    ] == raw_medians
    assert statistics.median(raw_medians) == 30.0
    assert loaded["recomputed_aggregate"] == {
        key: loaded["manifest"][key] for key in loaded["recomputed_aggregate"]
    }


def test_changed_raw_child_requires_hash_and_recomputed_aggregate(
    campaign, tmp_path
):
    """Raw mutation requires both byte integrity and aggregate consistency."""
    path = write_campaign(tmp_path, children=[make_child()])
    child_path = tmp_path / "process-000.json"
    child = campaign.load_unique_json(child_path)
    _replace_samples(_measurement(child), [19.0, 21.0])
    _save(child_path, child)
    with pytest.raises(ValueError, match="SHA-256"):
        campaign.load_campaign(path)
    manifest = campaign.load_unique_json(path)
    manifest["children"][0]["sha256"] = hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    _save(path, manifest)
    with pytest.raises(ValueError, match="aggregate drift"):
        campaign.load_campaign(path)
    manifest.update(campaign.aggregate_children([child]))
    _save(path, manifest)
    loaded = campaign.load_campaign(path)
    assert (
        loaded["child_process_medians"][0]["vadd/n=1024/swage/call_us"] == 20.0
    )


@pytest.mark.parametrize(
    "path,value",
    [
        (("schema_version",), True),
        (("process_count",), True),
        (("process_count",), 2),
        (("controls", "repetitions"), 2),
        (("controls", "samples_per_process"), 3),
        (("controls", "suite"), "segmented-sum"),
        (("controls", "exclusive_gpu_allocated"), 1),
        (("controls", "allow_shared_gpu_engineering"), True),
        (("children", 0, "process_index"), True),
        (("children", 0, "path"), "../process-000.json"),
        (("children", 0, "path"), "/tmp/process-000.json"),
        (("children", 0, "path"), "process-001.json"),
        (("children", 0, "recorded_at"), "2026-09-08T12:01:00+00:00"),
        (("children", 0, "sha256"), "0" * 64),
        (("agreement", "compilation_metadata"), 1),
        (("archival_eligible",), False),
        (("archival_ineligibility_reasons",), ["invented reason"]),
    ],
)
def test_manifest_rejects_path_control_timestamp_and_flag_corruption(
    campaign, tmp_path, path, value
):
    """Manifest declarations must agree with files and observed controls."""
    manifest_path = write_campaign(tmp_path, children=[make_child()])
    manifest = campaign.load_unique_json(manifest_path)
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _save(manifest_path, manifest)
    with pytest.raises(ValueError):
        campaign.load_campaign(manifest_path, require_archival=False)


@pytest.mark.parametrize(
    "kind", ["missing", "extra", "symlink", "duplicate_path", "hardlink"]
)
def test_campaign_rejects_missing_extra_and_aliased_child_files(
    campaign, tmp_path, kind
):
    """Each recorded process must map to exactly one distinct local file."""
    path = write_campaign(tmp_path, children=[make_child(), make_child()])
    first, second = tmp_path / "process-000.json", tmp_path / "process-001.json"
    if kind == "missing":
        first.unlink()
    elif kind == "extra":
        (tmp_path / "process-002.json").write_bytes(first.read_bytes())
    elif kind == "symlink":
        second.unlink()
        second.symlink_to(first.name)
    elif kind == "hardlink":
        second.unlink()
        second.hardlink_to(first)
    else:
        manifest = campaign.load_unique_json(path)
        manifest["children"][1]["path"] = first.name
        _save(path, manifest)
    with pytest.raises(ValueError):
        campaign.load_campaign(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "subset",
        "boolean_pid",
        "negative_memory",
        "missing_unit",
        "naive_timestamp",
    ],
)
def test_stored_telemetry_schema_is_not_a_truthy_subset(campaign, mutation):
    """Stored policy rejects incomplete or scalar-corrupted observations."""
    process = {
        "gpu_uuid": "GPU-fixture",
        "pid": 123,
        "process_name": "competitor",
        "used_memory_mib": 5,
    }
    telemetry = make_telemetry(processes=[process])
    if mutation == "subset":
        telemetry = {"compute_processes": telemetry["compute_processes"]}
    elif mutation == "boolean_pid":
        telemetry["compute_processes"]["processes"][0]["pid"] = True
    elif mutation == "negative_memory":
        telemetry["compute_processes"]["processes"][0]["used_memory_mib"] = -1
    elif mutation == "missing_unit":
        del telemetry["fields"]["power_draw_watts"]
    else:
        telemetry["recorded_at"] = "2026-09-08T12:00:00"
    with pytest.raises(ValueError):
        campaign.compute_process_reasons(telemetry, "boundary")


def test_stored_contention_cannot_be_hidden_by_archival_flags(
    campaign, tmp_path
):
    """Contention observations override an unsupported archival claim."""
    path = write_campaign(tmp_path, children=[make_child()])
    manifest = campaign.load_unique_json(path)
    observation = make_telemetry(
        processes=[
            {
                "gpu_uuid": "GPU-fixture",
                "pid": 123,
                "process_name": "competitor",
                "used_memory_mib": None,
            }
        ]
    )
    manifest["children"][0]["nvidia_telemetry"]["post_process"] = observation
    _save(path, manifest)
    with pytest.raises(ValueError, match="archival reasons"):
        campaign.load_campaign(path)
    reasons = campaign.compute_process_reasons(
        observation, "process 0 post-process boundary"
    )
    assert "competing compute process" in reasons[0]
    manifest.update(
        archival_eligible=False, archival_ineligibility_reasons=reasons
    )
    _save(path, manifest)
    loaded = campaign.load_campaign(path, require_archival=False)
    assert loaded["manifest"]["archival_eligible"] is False
    with pytest.raises(ValueError, match="not archival eligible"):
        campaign.load_campaign(path)


def test_unavailable_observations_retain_their_reason(campaign):
    """Unavailable telemetry preserves the concrete observation failure."""
    assert campaign.compute_process_reasons(
        {"available": False, "reason": "nvidia-smi not found"},
        "pre",
    ) == ["pre: compute-process telemetry unavailable (nvidia-smi not found)"]
    telemetry = make_telemetry()
    telemetry["compute_processes"] = {
        "available": False,
        "reason": "query failed",
    }
    assert campaign.compute_process_reasons(telemetry, "post") == [
        "post: compute-process telemetry unavailable (query failed)"
    ]


def test_shared_and_nonrelease_campaigns_are_not_archival(campaign, tmp_path):
    """Engineering and Debug records remain readable but not archival."""
    shared = write_campaign(
        tmp_path / "shared", children=[make_child()], archival=False
    )
    loaded = campaign.load_campaign(shared, require_archival=False)
    assert loaded["manifest"]["archival_ineligibility_reasons"] == [
        "shared-GPU engineering mode selected; "
        "exclusive GPU allocation was not asserted",
        "archival source revision was not asserted",
    ]
    with pytest.raises(ValueError, match="not archival eligible"):
        campaign.load_campaign(shared)
    child = make_child()
    child["environment"]["compiler"]["build_type"] = "Debug"
    debug = write_campaign(tmp_path / "debug", children=[child])
    loaded = campaign.load_campaign(debug, require_archival=False)
    assert loaded["manifest"]["archival_ineligibility_reasons"] == [
        "compiler build type is not Release"
    ]
    with pytest.raises(ValueError, match="not archival eligible"):
        campaign.load_campaign(debug)
