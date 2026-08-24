# tests/python/test_render_docs_diagrams.py
"""Tests for the deterministic accessible documentation diagrams."""

import importlib.util
import xml.etree.ElementTree as ET
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "render_docs_diagrams.py"
EXPECTED_OUTPUTS = {
    "capability-boundary.svg",
    "compiler-pipeline.svg",
    "frontend-boundary.svg",
    "m8-split-lifecycle.svg",
    "ragged-storage.svg",
    "runtime-lifecycle.svg",
    "segments-tasks-tiles.svg",
}
REQUIRED_LABELS = {
    "capability-boundary.svg": (
        "PUBLIC TODAY",
        "PRIVATE QUALIFICATION",
        "not a public API",
        "PLANNED",
        "public segmented API",
    ),
    "frontend-boundary.svg": (
        "restricted AST",
        "verified semantic MLIR",
        "emit_mlir()",
        "compile-only",
        "launch()",
        "sm_80+",
    ),
    "ragged-storage.svg": (
        "offsets = [0, 2, 2, 5, 6]",
        "[2, 2)",
        "empty segment",
    ),
    "segments-tasks-tiles.svg": (
        "runtime-length segment",
        "policy-bearing work unit",
        "32-thread warp step",
        "128-thread CTA step",
    ),
    "compiler-pipeline.svg": (
        "verified semantic MLIR",
        "M3 fixed block",
        "M4 / M5 segmented direct",
        "M6-M8 SwagePlan",
        "GPU / SCF / NVVM / LLVM",
        "CUDA Driver API",
        "sequential CPU oracle",
        "SCF / memref stop",
        "GPU: one CTA / segment",
    ),
    "m8-split-lifecycle.svg": (
        "identity sum only",
        "absolute input ranges",
        "unique scratch writer",
        "compact scratch range",
        "one final writer",
        "direct fused > partial CTAs > merge CTAs",
        "empty phases skipped",
    ),
    "runtime-lifecycle.svg": (
        "VALIDATE",
        "FAIL CLOSED",
        "zero-work no-op",
        "specialize / cache",
        "current PyTorch stream",
        "record_stream()",
        "no synchronization or fallback",
    ),
}


def _load_renderer():
    assert SCRIPT.is_file(), f"missing renderer: {SCRIPT}"
    spec = importlib.util.spec_from_file_location(
        "render_docs_diagrams", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contrast_ratio(first: str, second: str) -> float:
    def luminance(color: str) -> float:
        channels = [
            int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)
        ]
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    light, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (light + 0.05) / (dark + 0.05)


def test_rendered_bytes_are_deterministic_and_exactly_seven():
    """Catch nondeterministic content and additions outside the fixed atlas."""
    renderer = _load_renderer()

    first = renderer.render_all()
    second = renderer.render_all()

    assert set(first) == EXPECTED_OUTPUTS
    assert first == second
    for name, content in first.items():
        assert content.startswith(
            f"<!-- docs/assets/diagrams/{name} -->\n".encode()
        )
        assert content.endswith(b"\n")


def test_svg_markup_is_accessible_self_contained_and_parseable():
    """Catch inaccessible wiring, unsafe resources, and undersized text."""
    renderer = _load_renderer()
    seen_ids = set()

    for name, content in renderer.render_all().items():
        root = ET.fromstring(content)
        assert root.tag == "{http://www.w3.org/2000/svg}svg"
        assert root.attrib["viewBox"].startswith("0 0 ")
        assert root.attrib["role"] == "img"

        labelled = root.attrib["aria-labelledby"].split()
        assert len(labelled) == 2
        ids = {
            element.attrib["id"]
            for element in root.iter()
            if "id" in element.attrib
        }
        assert set(labelled) <= ids
        assert seen_ids.isdisjoint(ids), f"duplicate ID across atlas: {name}"
        seen_ids.update(ids)

        canvas = next(
            element
            for element in root
            if element.attrib.get("data-role") == "canvas"
        )
        assert canvas.attrib["fill"].startswith("#")
        assert "opacity" not in canvas.attrib

        for element in root.iter():
            local_name = element.tag.rsplit("}", 1)[-1]
            assert local_name not in {"script", "foreignObject"}
            if local_name == "text":
                assert float(element.attrib["font-size"]) >= 16
            for attribute, value in element.attrib.items():
                assert not attribute.lower().startswith("on")
                assert not value.lower().startswith(
                    ("data:", "file:", "http:", "https:")
                )


def test_diagrams_include_the_required_semantic_labels():
    """Catch visual output that drops a required boundary or lifecycle fact."""
    renderer = _load_renderer()

    for name, labels in REQUIRED_LABELS.items():
        content = renderer.render_all()[name].decode()
        for label in labels:
            assert label in content, f"{name} is missing {label!r}"


def test_runtime_validation_labels_fit_panel():
    """Catch centered validation labels that visibly exceed their panel."""
    renderer = _load_renderer()
    root = ET.fromstring(renderer.render_all()["runtime-lifecycle.svg"])
    labels = [
        element
        for element in root.iter("{http://www.w3.org/2000/svg}text")
        if 172 <= float(element.attrib["y"]) <= 310
        and 55 <= float(element.attrib["x"]) <= 275
    ]

    assert labels
    for label in labels:
        font_size = float(label.attrib["font-size"])
        estimated_width = len("".join(label.itertext())) * font_size * 0.62
        center = float(label.attrib["x"])
        assert center - estimated_width / 2 >= 67
        assert center + estimated_width / 2 <= 263


def test_palette_pairs_meet_accessible_contrast_thresholds():
    """Catch palette changes that make text or essential edges illegible."""
    renderer = _load_renderer()

    for foreground, background, minimum in renderer.CONTRAST_PAIRS:
        ratio = _contrast_ratio(
            renderer.PALETTE[foreground], renderer.PALETTE[background]
        )
        assert ratio >= minimum, (
            f"{foreground} on {background} is {ratio:.2f}:1, below {minimum}:1"
        )


def test_write_and_check_modes_manage_only_the_expected_set(tmp_path):
    """Catch incomplete writes and freshness checks that mutate output."""
    renderer = _load_renderer()
    output_dir = tmp_path / "diagrams"

    assert renderer.render_diagrams(output_dir) == []
    assert {path.name for path in output_dir.iterdir()} == EXPECTED_OUTPUTS
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}
    assert renderer.render_diagrams(output_dir, check=True) == []
    assert before == {
        path.name: path.read_bytes() for path in output_dir.iterdir()
    }


def test_check_reports_missing_stale_and_orphaned_outputs_without_writes(
    tmp_path,
):
    """Catch incomplete diagnostics or writes performed by check mode."""
    renderer = _load_renderer()
    output_dir = tmp_path / "diagrams"
    assert renderer.render_diagrams(output_dir) == []

    missing = output_dir / "capability-boundary.svg"
    stale = output_dir / "frontend-boundary.svg"
    orphan = output_dir / "old-diagram.svg"
    missing.unlink()
    stale.write_text("stale\n", encoding="utf-8")
    orphan.write_text("orphan\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    assert renderer.render_diagrams(output_dir, check=True) == [
        f"missing generated diagram: {missing}",
        f"stale generated diagram: {stale}",
        f"orphaned generated diagram: {orphan}",
    ]
    assert before == {
        path.name: path.read_bytes() for path in output_dir.iterdir()
    }
