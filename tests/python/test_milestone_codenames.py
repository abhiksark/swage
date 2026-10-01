# tests/python/test_milestone_codenames.py
"""Keep internal planning codenames out of product-facing surfaces."""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

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
_PLANNING_DIRECTORY = "maintainers"
_EXEMPT_FILES = {
    Path("ROADMAP.md"),
    Path("scripts/mkdocs_redirect_pages.py"),
    Path("tests/python/test_mkdocs_redirect_pages.py"),
}
_CODENAME = re.compile(
    r"(?<![A-Za-z0-9])(?:M(?:10|[0-9])|P[0])(?![0-9])",
    re.IGNORECASE,
)
# Assembled at run time so this tracked file never spells a codename.
_SAMPLE_CODENAME = "M" + "7"


def _git(root, *arguments):
    """Run git in a tree, or return None when git cannot answer."""
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def _tracked_files(root):
    """Return the files git tracks in a checkout rooted at root.

    Args:
        root: Directory expected to be the top level of a git work tree.

    Returns:
        Sorted paths relative to root, or None when root is not the top
        level of a git checkout or git is unavailable. A tree unpacked
        inside another repository is not that repository's checkout, so
        it also yields None.
    """
    top_level = _git(root, "rev-parse", "--show-toplevel")
    if top_level is None:
        return None
    if Path(os.fsdecode(top_level).rstrip("\n")).resolve() != root.resolve():
        return None
    listing = _git(root, "ls-files", "-z")
    if listing is None:
        return None
    return sorted(
        Path(os.fsdecode(name)) for name in listing.split(b"\0") if name
    )


def _walked_files(root):
    """Yield files under root without traversing build or planning trees."""
    for directory, directories, filenames in os.walk(root):
        directories[:] = sorted(
            name
            for name in directories
            if name not in _SKIP_PARTS
            and not (Path(directory) == root and name == _PLANNING_DIRECTORY)
        )
        for filename in sorted(filenames):
            yield (Path(directory) / filename).relative_to(root)


def _source_files(root):
    """Yield the repository-relative files that must stay codename free.

    A git checkout is scanned through its tracked files, so untracked
    tool output cannot change the result. Any other tree, such as an
    unpacked source archive, is scanned by walking the filesystem.

    Args:
        root: Repository root to scan.

    Yields:
        Paths relative to root, excluding maintainer planning files.
    """
    tracked = _tracked_files(root)
    if tracked is None:
        yield from _walked_files(root)
        return
    for relative in tracked:
        if relative.parts[0] != _PLANNING_DIRECTORY:
            yield relative


def _violations(root):
    """Return one message per codename found in a path or in file text."""
    violations = []
    for relative in _source_files(root):
        if relative in _EXEMPT_FILES:
            continue
        path_match = _CODENAME.search(relative.as_posix())
        if path_match:
            violations.append(f"{relative}: codename in path")

        # Generated SVG path data can resemble planning identifiers. Its
        # source generator or TeX file is scanned instead.
        if relative.suffix.lower() == ".svg":
            continue
        try:
            text = (root / relative).read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for match in _CODENAME.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            violations.append(
                f"{relative}:{line}: unexpected {match.group(0)!r}"
            )
    return violations


def _write(root, relative, text):
    """Write one file under root, creating its parent directories."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_milestone_codenames_stay_in_planning_history():
    """Use capability names in code, docs, tests, and artifact paths."""
    this_file = Path(__file__).resolve().relative_to(REPO_ROOT.resolve())
    assert this_file in set(_source_files(REPO_ROOT)), "the scan is empty"

    violations = _violations(REPO_ROOT)

    assert not violations, "\n" + "\n".join(violations)


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_git_checkout_scans_tracked_files_only(tmp_path):
    """Untracked files in a checkout never reach the scan."""
    assert _git(tmp_path, "init", "--quiet") is not None
    _write(tmp_path, "docs/guide.md", f"shipped in {_SAMPLE_CODENAME}\n")
    _write(tmp_path, "docs/clean.md", "capability names only\n")
    _write(tmp_path, "maintainers/plan.md", f"{_SAMPLE_CODENAME} plan\n")
    assert _git(tmp_path, "add", "docs", "maintainers") is not None
    _write(tmp_path, "scratch.md", f"local {_SAMPLE_CODENAME} note\n")
    _write(tmp_path, "tool-cache/graph.json", f"{_SAMPLE_CODENAME}\n")

    assert _tracked_files(tmp_path) == [
        Path("docs/clean.md"),
        Path("docs/guide.md"),
        Path("maintainers/plan.md"),
    ]
    assert _violations(tmp_path) == [
        f"docs/guide.md:1: unexpected {_SAMPLE_CODENAME!r}"
    ]


def test_tree_without_git_metadata_falls_back_to_a_filesystem_walk(tmp_path):
    """A source archive is scanned by walking it, minus generated trees."""
    _write(tmp_path, "docs/guide.md", f"shipped in {_SAMPLE_CODENAME}\n")
    _write(tmp_path, "docs/clean.md", "capability names only\n")
    _write(tmp_path, "build/generated.md", f"{_SAMPLE_CODENAME}\n")
    _write(tmp_path, "maintainers/plan.md", f"{_SAMPLE_CODENAME} plan\n")

    assert _tracked_files(tmp_path) is None
    assert sorted(_source_files(tmp_path)) == [
        Path("docs/clean.md"),
        Path("docs/guide.md"),
    ]
    assert _violations(tmp_path) == [
        f"docs/guide.md:1: unexpected {_SAMPLE_CODENAME!r}"
    ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git unavailable")
def test_tree_nested_in_another_checkout_is_not_that_checkout(tmp_path):
    """An archive unpacked inside a repository is still walked."""
    assert _git(tmp_path, "init", "--quiet") is not None
    unpacked = tmp_path / "swage-source"
    _write(unpacked, "docs/guide.md", f"shipped in {_SAMPLE_CODENAME}\n")

    assert _tracked_files(unpacked) is None
    assert _violations(unpacked) == [
        f"docs/guide.md:1: unexpected {_SAMPLE_CODENAME!r}"
    ]
