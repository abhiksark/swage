# scripts/sync_mlir_reference.py
"""Synchronize checked-in reference fragments with MLIR documentation."""

import argparse
from pathlib import Path

REFERENCE_FILES = (
    (Path("docs/swage/SwageDialect.md"), "swage-dialect.inc"),
    (Path("docs/swage/SwageOps.md"), "swage-ops.inc"),
    (Path("docs/swage_plan/SwagePlanDialect.md"), "swage-plan-dialect.inc"),
    (Path("docs/swage_plan/SwagePlanOps.md"), "swage-plan-ops.inc"),
)
DEFAULT_OUTPUT_DIR = Path(__file__).parents[1] / "docs/reference/_generated"


def normalize_markdown(text: str) -> str:
    """Nest generated Markdown below a wrapper page heading."""
    output = []
    fence = None
    for line in text.splitlines():
        candidate = line.lstrip(" ")
        if candidate.startswith(("```", "~~~")):
            marker = candidate[:3]
            if fence is None:
                fence = marker
            elif marker == fence:
                fence = None
        elif fence is None and candidate == "[TOC]":
            continue
        elif fence is None:
            heading_length = len(candidate) - len(candidate.lstrip("#"))
            has_heading_space = candidate[heading_length:].startswith(" ")
            if 0 < heading_length <= 6 and has_heading_space:
                line = line[: len(line) - len(candidate)] + "#" + candidate
        output.append(line)
    return "\n".join(output).rstrip("\n") + "\n"


def sync_references(
    build_dir: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    check: bool = False,
) -> list[str]:
    """Write or check committed fragments against MLIR documentation output."""
    rendered = {}
    errors = []
    for input_path, output_name in REFERENCE_FILES:
        source = build_dir / input_path
        if not source.is_file():
            errors.append(f"missing generated input: {source}")
        else:
            rendered[output_name] = normalize_markdown(
                source.read_text(encoding="utf-8")
            ).encode("utf-8")

    if check:
        for _, output_name in REFERENCE_FILES:
            output = output_dir / output_name
            if not output.is_file():
                errors.append(f"missing committed output: {output}")
            elif (
                output_name in rendered
                and output.read_bytes() != rendered[output_name]
            ):
                errors.append(f"stale committed output: {output}")
        return errors

    if errors:
        return errors
    output_dir.mkdir(parents=True, exist_ok=True)
    for _, output_name in REFERENCE_FILES:
        (output_dir / output_name).write_bytes(rendered[output_name])
    return []


def main() -> int:
    """Synchronize generated MLIR reference fragments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    errors = sync_references(args.build_dir, check=args.check)
    for error in errors:
        print(error)
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
