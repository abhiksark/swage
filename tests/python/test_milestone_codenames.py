# tests/python/test_milestone_codenames.py
"""Keep internal planning codenames out of product-facing surfaces."""

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]
_SKIP_PARTS = {
    ".benchmarks",
    ".code-review-graph",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".superpowers",
    ".tox",
    ".venv",
    "CMakeFiles",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "site",
    "venv",
}
_EXEMPT_FILES = {
    Path("ROADMAP.md"),
    Path("scripts/mkdocs_redirect_pages.py"),
    Path("tests/python/test_mkdocs_redirect_pages.py"),
}
_CODENAME = re.compile(
    r"(?<![A-Za-z0-9])(?:M(?:10|[0-9])|P[0])(?![0-9])",
    re.IGNORECASE,
)


def _source_files():
    """Yield source files without traversing build or planning trees."""
    for root, directories, filenames in os.walk(REPO_ROOT):
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in _SKIP_PARTS
            and not (Path(root) == REPO_ROOT and directory == "maintainers")
        )
        for filename in sorted(filenames):
            yield Path(root) / filename


def test_milestone_codenames_stay_in_planning_history():
    """Use capability names in code, docs, tests, and artifact paths."""
    violations = []
    for path in _source_files():
        relative = path.relative_to(REPO_ROOT)
        if relative in _EXEMPT_FILES:
            continue
        path_match = _CODENAME.search(relative.as_posix())
        if path_match:
            violations.append(f"{relative}: codename in path")

        # Generated SVG path data can resemble planning identifiers. Its
        # source generator or TeX file is scanned instead.
        if path.suffix.lower() == ".svg":
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for match in _CODENAME.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            violations.append(
                f"{relative}:{line}: unexpected {match.group(0)!r}"
            )

    assert not violations, "\n" + "\n".join(violations)
