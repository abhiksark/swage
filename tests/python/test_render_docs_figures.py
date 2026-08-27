# tests/python/test_render_docs_figures.py
"""Validate the TikZ figure atlas without requiring a TeX toolchain."""

import importlib.util
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "render_docs_figures.py"

REQUIRED_LABELS = {
    "fixed-block-thread-map": (
        "grid = ceil(n / BLOCK) blocks",
        "one block = BLOCK threads",
        "gid = program_id(0) * BLOCK + arange(0, BLOCK)",
        "mask = gid < n",
        "masked",
        "public M3",
    ),
    "warp-vs-cta-tiles": (
        "32-thread warp tile",
        "128-thread CTA tile",
        "shuffle reduction tree",
        "block-stride chunks",
        "up to 32 elements",
        "33 to 4096 elements",
    ),
    "fused-mixed-schedule": (
        "one 128-thread block",
        "four independent warp slots",
        "task_ids",
        "warp tasks: four per block",
        "CTA tasks: one per block",
        "warp_task_count",
        "cta_task_count",
    ),
    "plan-classification": (
        "swage_plan.classify",
        "empty",
        "warp",
        "cta",
        "split",
        "four task lists",
        "INT32_MAX >= cta chunk >= warp max > 0",
    ),
    "specialization-key-cache": (
        "specialization key",
        "normalized source",
        "ordered ABI descriptors",
        "exact compute capability",
        "Swage revision",
        "LLVM version",
        "verified before module load",
        "rejected",
    ),
    "dispatch-path": (
        "_launch_kernel",
        "nanobind",
        "dlopen libcuda.so.1",
        "GIL held across the enqueue",
        "ctypes fallback",
        "compiled bindings are absent",
        "current PyTorch stream",
    ),
    "timing-methods": (
        "call_us",
        "kernel_us",
        "graph_us",
        "synchronized wall clock",
        "32 back-to-back launches",
        "launcher still visible",
        "CUDA-graph replay",
        "host removed",
    ),
}


def _load_renderer():
    """Load the figure renderer module from its script path."""
    spec = importlib.util.spec_from_file_location(
        "render_docs_figures", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_manifest_matches_the_committed_tex_sources():
    """Every manifest figure has a source and no source is unlisted."""
    module = _load_renderer()
    names = [spec.name for spec in module.FIGURES]
    assert sorted(names) == sorted(set(names))
    assert set(names) == set(REQUIRED_LABELS)
    sources = {
        path.stem
        for path in module.SOURCE_DIR.glob("*.tex")
        if path.name != module.PREAMBLE_NAME
    }
    assert sources == set(names)
    assert (module.SOURCE_DIR / module.PREAMBLE_NAME).is_file()


def test_figures_declare_the_required_source_labels():
    """Each TeX source carries its figure's content contract."""
    module = _load_renderer()
    for spec in module.FIGURES:
        source = (module.SOURCE_DIR / f"{spec.name}.tex").read_text()
        for label in REQUIRED_LABELS[spec.name]:
            assert label in source, (spec.name, label)


def test_tex_sources_open_with_their_repo_path_comment():
    """Sources and the preamble name their repo-relative location."""
    module = _load_renderer()
    paths = [module.SOURCE_DIR / module.PREAMBLE_NAME]
    paths += [
        module.SOURCE_DIR / f"{spec.name}.tex" for spec in module.FIGURES
    ]
    for path in paths:
        first_line = path.read_text().splitlines()[0]
        assert first_line == f"% figures/{path.name}", path


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
        f"missing generated figure: {tmp_path}/{spec.name}.svg"
        for spec in module.FIGURES
    )

    for spec in module.FIGURES:
        committed = module.OUTPUT_DIR / f"{spec.name}.svg"
        (tmp_path / f"{spec.name}.svg").write_bytes(committed.read_bytes())
    stale_path = tmp_path / f"{module.FIGURES[0].name}.svg"
    lines = stale_path.read_text().splitlines(keepends=True)
    lines[1] = f"<!-- source-sha256: {'0' * 64} -->\n"
    stale_path.write_text("".join(lines))
    orphan = tmp_path / "stray.svg"
    orphan.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
    errors = module.render_figures(tmp_path, check=True)
    assert errors == sorted(
        [
            f"orphaned generated figure: {orphan}",
            f"stale generated figure: {stale_path}",
        ]
    )
    assert orphan.exists()
    assert stale_path.read_text() == "".join(lines)


def test_perf_snapshot_is_wellformed_and_sourced():
    """The committed campaign snapshot is complete and auditable."""
    module = _load_renderer()
    snapshot = json.loads(module.SNAPSHOT_PATH.read_text())
    assert snapshot["environment"]["compute_capability"] == "sm_120"
    assert snapshot["recorded_at"] == "2026-08-27"
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
            spread = "q1" in cell and cell["q1"] <= cell["q3"]
            assert sourced or spread, (row["distribution"], impl)
    for stage in snapshot["dispatch_call_us"]:
        assert stage["median"] > 0
        assert "provenance" in stage or stage["q1"] <= stage["q3"]
    assert "provenance" in snapshot["cold_start_ms"]


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
    spec = module.FIGURES[0]
    orphan = tmp_path / "stray.svg"
    orphan.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>\n")
    assert module.render_figures(tmp_path, check=False) == []
    assert not orphan.exists()
    rendered = tmp_path / f"{spec.name}.svg"
    digest = module.figure_digest(spec)
    assert f"<!-- source-sha256: {digest} -->" in rendered.read_text()
    assert module.render_figures(tmp_path, check=True) == []
