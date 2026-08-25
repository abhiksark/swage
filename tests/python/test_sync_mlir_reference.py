# tests/python/test_sync_mlir_reference.py
"""Tests for deterministic MLIR reference synchronization."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "sync_mlir_reference.py"
GENERATED_INPUTS = {
    "docs/swage/SwageDialect.md": "swage-dialect.inc",
    "docs/swage/SwageOps.md": "swage-ops.inc",
    "docs/swage_plan/SwagePlanDialect.md": "swage-plan-dialect.inc",
    "docs/swage_plan/SwagePlanOps.md": "swage-plan-ops.inc",
}


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_mlir_reference", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_inputs(build_dir: Path, contents: dict[str, str] | None = None):
    contents = contents or {}
    for relative_path in GENERATED_INPUTS:
        path = build_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            contents.get(relative_path, "# Reference\n"), encoding="utf-8"
        )


def test_normalize_removes_toc_and_nests_headings():
    """Remove TOC markers and demote headings outside code fences."""
    sync = _load_sync_module()
    source = """<!-- generated -->
[TOC]
# Dialect

## Operation

```mlir
# Not a Markdown heading
```

Trailing text.

"""

    assert sync.normalize_markdown(source) == """<!-- generated -->
## Dialect

### Operation

```mlir
# Not a Markdown heading
```

Trailing text.
"""


def test_normalize_preserves_content_in_longer_fences():
    """Keep shorter delimiter runs inside a longer fenced block."""
    sync = _load_sync_module()
    source = (
        "````markdown\n"
        "literal delimiter:\n"
        "```\n"
        "# Still code, not a heading\n"
        "````   \n"
        "# Outside heading\n"
    )

    assert sync.normalize_markdown(source) == (
        "````markdown\n"
        "literal delimiter:\n"
        "```\n"
        "# Still code, not a heading\n"
        "````   \n"
        "## Outside heading\n"
    )


def test_write_mode_creates_all_four_deterministic_fragments(tmp_path):
    """Write the exact four stable fragments from generated inputs."""
    sync = _load_sync_module()
    build_dir = tmp_path / "build"
    output_dir = tmp_path / "docs" / "reference" / "_generated"
    _write_inputs(
        build_dir,
        {
            relative_path: f"[TOC]\n# {output_name}\n"
            for relative_path, output_name in GENERATED_INPUTS.items()
        },
    )

    assert sync.sync_references(build_dir, output_dir) == []
    first_bytes = {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }
    assert set(first_bytes) == set(GENERATED_INPUTS.values())
    assert first_bytes == {
        output_name: f"## {output_name}\n".encode()
        for output_name in GENERATED_INPUTS.values()
    }

    assert sync.sync_references(build_dir, output_dir) == []
    assert first_bytes == {
        path.name: path.read_bytes() for path in sorted(output_dir.iterdir())
    }


def test_missing_generated_inputs_fail_without_writes(tmp_path):
    """Report every missing input before creating the output directory."""
    sync = _load_sync_module()
    build_dir = tmp_path / "build"
    output_dir = tmp_path / "output"
    one_input = next(iter(GENERATED_INPUTS))
    _write_inputs(build_dir, {one_input: "# Present\n"})
    for relative_path in GENERATED_INPUTS:
        if relative_path != one_input:
            (build_dir / relative_path).unlink()

    errors = sync.sync_references(build_dir, output_dir)

    assert len(errors) == 3
    assert all(
        error.startswith("missing generated input: ") for error in errors
    )
    assert not output_dir.exists()


def test_check_reports_every_missing_committed_fragment(tmp_path):
    """Report every absent committed fragment in check mode."""
    sync = _load_sync_module()
    build_dir = tmp_path / "build"
    output_dir = tmp_path / "output"
    _write_inputs(build_dir)

    errors = sync.sync_references(build_dir, output_dir, check=True)

    assert errors == [
        f"missing committed output: {output_dir / output_name}"
        for output_name in GENERATED_INPUTS.values()
    ]
    assert not output_dir.exists()


def test_check_reports_stale_fragments_without_writes(tmp_path):
    """Report all stale fragments without changing their bytes."""
    sync = _load_sync_module()
    build_dir = tmp_path / "build"
    output_dir = tmp_path / "output"
    _write_inputs(build_dir)
    assert sync.sync_references(build_dir, output_dir) == []
    stale_names = ["swage-dialect.inc", "swage-plan-ops.inc"]
    for name in stale_names:
        (output_dir / name).write_text("stale\n", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in output_dir.iterdir()}

    errors = sync.sync_references(build_dir, output_dir, check=True)

    assert errors == [
        f"stale committed output: {output_dir / name}" for name in stale_names
    ]
    assert before == {
        path.name: path.read_bytes() for path in output_dir.iterdir()
    }


def test_check_rejects_orphans_and_write_removes_only_fragments(tmp_path):
    """Reject orphaned fragments without touching unrelated entries."""
    sync = _load_sync_module()
    build_dir = tmp_path / "build"
    output_dir = tmp_path / "output"
    _write_inputs(build_dir)
    assert sync.sync_references(build_dir, output_dir) == []
    orphan = output_dir / "old-reference.inc"
    unrelated = output_dir / "README.md"
    non_file = output_dir / "cache.inc"
    orphan.write_text("orphan\n", encoding="utf-8")
    unrelated.write_text("keep\n", encoding="utf-8")
    non_file.mkdir()
    before = {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
        if path.is_file()
    }

    assert sync.sync_references(build_dir, output_dir, check=True) == [
        f"orphaned committed output: {orphan}"
    ]
    assert before == {
        path.name: path.read_bytes()
        for path in output_dir.iterdir()
        if path.is_file()
    }

    assert sync.sync_references(build_dir, output_dir) == []
    assert not orphan.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep\n"
    assert non_file.is_dir()


def test_check_accepts_hand_written_fixtures(tmp_path):
    """Accept committed fragments that match hand-written fixtures."""
    sync = _load_sync_module()
    build_dir = tmp_path / "build"
    output_dir = tmp_path / "output"
    _write_inputs(
        build_dir,
        {
            relative_path: f"[TOC]\n# {output_name}\nBody.\n"
            for relative_path, output_name in GENERATED_INPUTS.items()
        },
    )
    output_dir.mkdir()
    for output_name in GENERATED_INPUTS.values():
        (output_dir / output_name).write_text(
            f"## {output_name}\nBody.\n", encoding="utf-8"
        )

    assert sync.sync_references(build_dir, output_dir, check=True) == []
