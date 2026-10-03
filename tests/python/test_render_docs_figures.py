# tests/python/test_render_docs_figures.py
"""Validate the TikZ figure atlas without requiring a TeX toolchain."""

import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from benchmark_campaign_fixtures import make_child, write_campaign

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_docs_figures.py"


def _load_renderer():
    """Load the figure renderer module from its script path."""
    spec = importlib.util.spec_from_file_location("render_docs_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_matches_the_committed_tex_sources():
    """Every manifest figure has a source and no source is unlisted."""
    module = _load_renderer()
    names = [spec.name for spec in module.FIGURES]
    assert sorted(names) == sorted(set(names))
    sources = {
        path.stem
        for path in module.SOURCE_DIR.glob("*.tex")
        if path.name != module.PREAMBLE_NAME
    }
    assert sources == set(names)
    assert (module.SOURCE_DIR / module.PREAMBLE_NAME).is_file()


def test_figure_digest_covers_titles_and_descriptions():
    """Injected SVG metadata participates in the freshness digest."""
    module = _load_renderer()
    spec = module.FIGURES[0]
    base = module.figure_digest(spec)
    assert module.figure_digest(spec._replace(title="Other")) != base
    assert module.figure_digest(spec._replace(description="Other")) != base


def test_committed_svgs_are_current_stamped_and_accessible():
    """Committed outputs are fresh, well formed, and self contained."""
    module = _load_renderer()
    assert module.render_figures(module.OUTPUT_DIR, check=True) == []
    for spec in module.FIGURES:
        path = module.OUTPUT_DIR / f"{spec.name}.svg"
        text = path.read_text()
        lines = text.splitlines()
        assert lines[0] == f"<!-- docs/assets/figures/{spec.name}.svg -->"
        digest = module.figure_digest(spec)
        assert lines[1] == f"<!-- source-sha256: {digest} -->"
        assert text.endswith("\n") and not text.endswith("\n\n")
        root = ET.fromstring(text)
        assert root.tag.endswith("svg")
        assert root.get("role") == "img"
        labelled = f"{spec.name}-title {spec.name}-description"
        assert root.get("aria-labelledby") == labelled
        title = root.find("{http://www.w3.org/2000/svg}title")
        desc = root.find("{http://www.w3.org/2000/svg}desc")
        assert title is not None and title.get("id") == f"{spec.name}-title"
        assert title.text == spec.title
        assert desc is not None
        assert desc.get("id") == f"{spec.name}-description"
        assert desc.text == spec.description
        for element in root.iter():
            assert not element.tag.endswith("script"), path
            for key, value in element.attrib.items():
                assert not key.startswith("on"), (path, key)
                forbidden = ("http:", "https:", "file:", "data:")
                assert not value.startswith(forbidden), (path, key, value)


def test_check_reports_missing_stale_and_orphaned_outputs(tmp_path):
    """Check mode diagnoses every drift class and never writes."""
    module = _load_renderer()
    missing = module.render_figures(tmp_path, check=True)
    assert missing == sorted(
        f"missing generated figure: {tmp_path / (spec.name + '.svg')}"
        for spec in module.FIGURES
    )

    for spec in module.FIGURES:
        committed = module.OUTPUT_DIR / f"{spec.name}.svg"
        (tmp_path / f"{spec.name}.svg").write_bytes(committed.read_bytes())
    stale_path = tmp_path / f"{module.FIGURES[0].name}.svg"
    lines = stale_path.read_text().splitlines(keepends=True)
    lines[1] = f"<!-- source-sha256: {'0' * 64} -->\n"
    stale_path.write_text("".join(lines))
    unstamped_path = tmp_path / f"{module.FIGURES[1].name}.svg"
    stamped_lines = unstamped_path.read_text().splitlines(keepends=True)
    unstamped_path.write_text("".join(stamped_lines[2:]))
    orphan = tmp_path / "stray.svg"
    orphan.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
    (tmp_path / "not-a-file.svg").mkdir()
    errors = module.render_figures(tmp_path, check=True)
    assert errors == sorted(
        [
            f"orphaned generated figure: {orphan}",
            f"stale generated figure: {stale_path}",
            f"stale generated figure: {unstamped_path}",
        ]
    )
    assert orphan.exists()
    assert stale_path.read_text() == "".join(lines)


def test_chart_data_includes_match_the_snapshot():
    """Generated chart data reproduces the committed snapshot values."""
    module = _load_renderer()
    snapshot = json.loads(module.SNAPSHOT_PATH.read_text())
    by_name = {spec.name: spec for spec in module.FIGURES}
    segsum = by_name["segsum-graph-comparison"]
    assert segsum.data == ("benchmarks/results/perf-5090-sm120.json",)
    include = module.chart_include(segsum)
    assert include == module.chart_include(segsum)
    for row in snapshot["segsum_graph_us"]:
        for impl in ("swage", "triton", "torch"):
            median = row[impl]["median"]
            coordinate = f"({row['distribution']},{median:.1f})"
            assert coordinate in include, coordinate

    ladder = by_name["dispatch-ladder"]
    assert ladder.data == ("benchmarks/results/perf-5090-sm120.json",)
    stages = snapshot["dispatch_call_us"]
    assert [stage["impl"] for stage in stages] == [
        "swage",
        "swage",
        "swage",
        "triton",
        "torch",
    ]
    assert [stage["stage"] for stage in stages] == [
        "baseline per-launch host path",
        "cached identity and emit-on-miss (O1)",
        "compiled nanobind launcher",
        "compiled-C launcher",
        "torch.add dispatch",
    ]
    ladder_include = module.chart_include(ladder)
    for stage in stages:
        assert f"({stage['median']:.1f}," in ladder_include, stage["stage"]
    cold = snapshot["cold_start_ms"]
    assert f"\\swagecoldms{{{cold['swage']}}}" in ladder_include
    assert f"\\tritoncoldms{{{cold['triton']}}}" in ladder_include


def test_perf_snapshot_is_wellformed_and_sourced():
    """The committed campaign snapshot is complete and auditable."""
    module = _load_renderer()
    snapshot = json.loads(module.SNAPSHOT_PATH.read_text())
    assert snapshot["environment"]["compute_capability"] == "sm_120"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", snapshot["recorded_at"])
    rows = snapshot["segsum_graph_us"]
    assert [row["distribution"] for row in rows] == [
        "uniform-8",
        "uniform-24",
        "uniform-512",
        "uniform-4k",
        "zipf",
        "bimodal",
        "few-huge",
    ]
    for row in rows:
        for impl in ("swage", "triton", "torch"):
            cell = row[impl]
            assert cell["median"] > 0, (row["distribution"], impl)
            sourced = "provenance" in cell
            spread = (
                "q1" in cell and 0 < cell["q1"] <= cell["median"] <= cell["q3"]
            )
            assert sourced or spread, (row["distribution"], impl)
    for stage in snapshot["dispatch_call_us"]:
        assert stage["median"] > 0
        assert "provenance" in stage or (
            0 < stage["q1"] <= stage["median"] <= stage["q3"]
        )
    assert "provenance" in snapshot["cold_start_ms"]
    sizes = [row["n"] for row in snapshot["vadd_graph_us"]]
    assert sizes == [2**e for e in range(14, 27, 2)]
    for row in snapshot["vadd_graph_us"]:
        for impl in ("swage", "triton", "torch"):
            cell = row[impl]
            assert 0 < cell["q1"] <= cell["median"] <= cell["q3"]


def _toolchain_missing():
    """Report whether the local render toolchain is unavailable."""
    module = _load_renderer()
    if module.tectonic_binary() is None:
        return True
    return importlib.util.find_spec("pymupdf") is None


@pytest.mark.skipif(
    _toolchain_missing(), reason="tectonic or pymupdf unavailable"
)
def test_render_mode_writes_stamped_svgs_and_removes_orphans(tmp_path):
    """Write mode renders fresh outputs and deletes strays."""
    module = _load_renderer()
    module.FIGURES = module.FIGURES[:1]
    spec = module.FIGURES[0]
    orphan = tmp_path / "stray.svg"
    orphan.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
    assert module.render_figures(tmp_path, check=False) == []
    assert not orphan.exists()
    rendered = tmp_path / f"{spec.name}.svg"
    digest = module.figure_digest(spec)
    assert f"<!-- source-sha256: {digest} -->" in rendered.read_text()
    assert module.render_figures(tmp_path, check=True) == []


def _coordinate_series(include):
    """Read numerical chart coordinates independently of the renderer."""
    return [
        [float(value) for _, value in re.findall(r"\((\d+),([^)]+)\)", body)]
        for body in re.findall(r"coordinates \{([^}]*)\}", include)
    ]


def test_campaign_coordinates_come_from_raw_child_samples(tmp_path):
    """Both chart includes use medians of five raw process medians."""
    renderer = _load_renderer()
    manifest = write_campaign(tmp_path)
    children = [
        json.loads((tmp_path / f"process-{index:03d}.json").read_text())
        for index in range(5)
    ]
    candidates = (
        "torch_padded",
        "swage_warp",
        "swage_cta",
        "triton_matched_task_partition",
        "swage_mixed",
    )

    def raw_median(row_index, candidate, phase):
        measurements = []
        for child in children:
            row = child["results"][row_index]
            measurement = (
                row["planning"]["timings"][candidate]
                if phase == "planning"
                else row["timings"][candidate][phase]
            )
            measurements.append(statistics.median(measurement["samples_us"]))
        return statistics.median(measurements)

    assert _coordinate_series(
        renderer._segmented_baselines_include(manifest)
    ) == [
        [raw_median(index, candidate, "batched_event") for index in range(8)]
        for candidate in candidates
    ]
    compilation = statistics.median(
        statistics.median(
            child["compilation"]["timings"]["swage_total"]["samples_us"]
        )
        for child in children
    )
    assert _coordinate_series(renderer._phase_breakdown_include(manifest)) == [
        [compilation],
        [raw_median(index, "swage_mixed", "planning") for index in range(8)],
        [
            raw_median(index, "swage_mixed", "batched_event")
            for index in range(8)
        ],
    ]


def test_campaign_chart_requires_archival_five_processes(tmp_path):
    """A shared-mode or undersized campaign cannot supply archival charts."""
    renderer = _load_renderer()
    shared = write_campaign(tmp_path / "shared", archival=False)
    with pytest.raises(ValueError, match="archival"):
        renderer._segmented_baselines_include(shared)
    single = write_campaign(
        tmp_path / "single", children=[make_child(suite="segmented-sum")]
    )
    with pytest.raises(ValueError, match="five independent processes"):
        renderer._phase_breakdown_include(single)


def test_campaign_tampering_and_digest_closure(tmp_path, monkeypatch):
    """Raw edits need hashes and aggregates, then change the figure digest."""
    monkeypatch.syspath_prepend(str(REPO_ROOT / "benchmarks"))
    renderer = _load_renderer()
    directory = tmp_path / "campaign"
    manifest_path = write_campaign(
        directory,
        children=[
            make_child(
                median=10.0 * (index + 1), samples=3, suite="segmented-sum"
            )
            for index in range(5)
        ],
    )
    # The digest fixture includes exactly the planned six evidence inputs
    # plus both Python sources, without registering an unmeasured figure.
    digest_renderer = _load_renderer()
    digest_renderer.REPO_ROOT = tmp_path
    digest_renderer.SOURCE_DIR = tmp_path / "figures"
    digest_renderer.SOURCE_DIR.mkdir()
    for source in (
        "scripts/render_docs_figures.py",
        "benchmarks/benchmark_campaign.py",
    ):
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / source, destination)
    (digest_renderer.SOURCE_DIR / "common-preamble.tex").write_text(
        "% preamble\n"
    )
    (digest_renderer.SOURCE_DIR / "campaign-smoke.tex").write_text("% chart\n")
    spec = digest_renderer.FigureSpec(
        "campaign-smoke",
        "Raw campaign",
        "Independent process medians",
        (
            "campaign/manifest.json",
            *(f"campaign/process-{index:03d}.json" for index in range(5)),
            "benchmarks/benchmark_campaign.py",
            "scripts/render_docs_figures.py",
        ),
    )

    def digest():
        include = renderer._segmented_baselines_include(manifest_path)
        return digest_renderer.figure_digest(spec, include=include)

    before = digest()
    child_path = directory / "process-002.json"
    child = json.loads(child_path.read_text())
    measurement = child["results"][0]["timings"]["swage_mixed"]["batched_event"]
    measurement["samples_us"] = [1.0, 31.0, 1000.0]
    quartiles = statistics.quantiles(
        measurement["samples_us"], method="inclusive"
    )
    measurement["summary_us"] = {
        "median": 31.0,
        "q1": quartiles[0],
        "q3": quartiles[2],
    }
    child_path.write_text(json.dumps(child))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        digest()
    manifest = json.loads(manifest_path.read_text())
    manifest["children"][2]["sha256"] = hashlib.sha256(
        child_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="stored aggregate drift"):
        digest()
    # Recompute only after proving that changing the hash is insufficient.
    from benchmark_campaign import aggregate_children

    children = [
        json.loads((directory / f"process-{index:03d}.json").read_text())
        for index in range(5)
    ]
    manifest.update(aggregate_children(children))
    manifest_path.write_text(json.dumps(manifest))
    assert digest() != before
