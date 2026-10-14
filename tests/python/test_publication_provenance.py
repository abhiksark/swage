# tests/python/test_publication_provenance.py
"""Exercise the workflow's first-parent publication guard on real Git DAGs."""

import ast
import pathlib
import re
import subprocess

import pytest
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _guard():
    """Compile the actual first-parent assertion from the release workflow."""
    workflow = yaml.safe_load(
        (_ROOT / ".github/workflows/publish-pypi.yml").read_text()
    )
    source = next(
        step["run"]
        for step in workflow["jobs"]["source"]["steps"]
        if step.get("id") == "source"
    )
    python = re.search(r"python - <<'PY'\n(.*?)\nPY", source, re.DOTALL)
    assert python is not None
    body = ast.parse(python.group(1)).body
    guards = [
        statement
        for statement in body
        if isinstance(statement, ast.Assert)
        and any(
            isinstance(node, ast.Constant) and node.value == "--first-parent"
            for node in ast.walk(statement)
        )
    ]
    assert len(guards) == 1, (
        "publication must check main's first-parent history"
    )
    return compile(
        ast.Module(body=guards, type_ignores=[]), "release-guard", "exec"
    )


@pytest.fixture
def release_history(tmp_path):
    """Create first-parent, merged-side, and outside-main commits."""

    def git(*arguments):
        return subprocess.check_output(
            [
                "git",
                "-C",
                str(tmp_path),
                "-c",
                "commit.gpgsign=false",
                "-c",
                "user.name=Policy Fixture",
                "-c",
                "user.email=policy@example.invalid",
                *arguments,
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    git("init", "--initial-branch=main")
    git("commit", "--allow-empty", "-m", "root")
    root = git("rev-parse", "HEAD")
    git("commit", "--allow-empty", "-m", "main change")
    direct = git("rev-parse", "HEAD")
    git("checkout", "-b", "side", root)
    git("commit", "--allow-empty", "-m", "side change")
    side = git("rev-parse", "HEAD")
    git("checkout", "main")
    git("merge", "--no-ff", "-m", "reviewed integration", "side")
    merged = git("rev-parse", "HEAD")
    git("update-ref", "refs/remotes/origin/main", merged)
    git("checkout", "-b", "outside")
    git("commit", "--allow-empty", "-m", "unmerged change")
    outside = git("rev-parse", "HEAD")
    return git, {
        "root": root,
        "direct": direct,
        "merged": merged,
        "side": side,
        "outside": outside,
    }


@pytest.mark.parametrize("position", ["root", "direct", "merged"])
def test_publication_accepts_main_first_parent_commits(
    release_history, position
):
    """Accept the root, a direct commit, and the reviewed merge commit."""
    git, commits = release_history
    exec(_guard(), {"git": git, "revision": commits[position]})


@pytest.mark.parametrize("position", ["side", "outside"])
def test_publication_refuses_side_or_unmerged_commits(
    release_history, position
):
    """Ancestry through a side parent is insufficient publication evidence."""
    git, commits = release_history
    guard = _guard()
    if position == "side":
        git(
            "merge-base",
            "--is-ancestor",
            commits[position],
            "refs/remotes/origin/main",
        )
    with pytest.raises(AssertionError, match="first-parent"):
        exec(guard, {"git": git, "revision": commits[position]})
