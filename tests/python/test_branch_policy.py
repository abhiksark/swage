# tests/python/test_branch_policy.py
"""Exercise branch routing through the metadata-only command boundary."""

import copy
import json
import os
import pathlib
import subprocess

import pytest
import yaml

_CHECKER = (
    pathlib.Path(__file__).resolve().parents[2]
    / ".github/scripts/check-pr-branch-policy.sh"
)
_REPOSITORY = {"id": 123, "full_name": "owner/swage"}


def _pull_request():
    """Return authenticated API metadata for an ordinary pull request."""
    return {
        "number": 42,
        "state": "open",
        "base": {"ref": "dev", "repo": _REPOSITORY},
        "head": {
            "ref": "feature/example",
            "sha": "a" * 40,
            "repo": {"id": 456, "full_name": "contributor/swage"},
        },
        "user": {"id": 7, "login": "contributor", "type": "User"},
    }


def _replace(record, changes):
    """Copy a record and replace dotted paths with the supplied values."""
    result = copy.deepcopy(record)
    for path, value in changes.items():
        target = result
        parts = path.split(".")
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value
    return result


def _run(tmp_path, pull_request, *, current=None, final=None, actor="alice"):
    """Evaluate an event against two independent current API records."""
    event = {"repository": _REPOSITORY, "pull_request": pull_request}
    current = pull_request if current is None else current
    final = current if final is None else final
    records = (event, current, final)
    paths = []
    for name, record in zip(("event", "current", "final"), records):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths.append(str(path))
    environment = {
        **os.environ,
        "GITHUB_REPOSITORY": "owner/swage",
        "GITHUB_REPOSITORY_ID": "123",
        "GITHUB_ACTOR": actor,
    }
    return subprocess.run(
        ["bash", str(_CHECKER), *paths],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


@pytest.mark.parametrize(
    ("changes", "allowed"),
    [
        ({}, True),
        ({"head.repo": _REPOSITORY}, True),
        (
            {"base.ref": "main", "head.ref": "dev", "head.repo": _REPOSITORY},
            True,
        ),
        (
            {
                "base.ref": "main",
                "head.ref": "hotfix/urgent",
                "head.repo": _REPOSITORY,
            },
            True,
        ),
        ({"base.ref": "main", "head.ref": "dev"}, False),
        ({"base.ref": "main", "head.ref": "hotfix/spoof"}, False),
        (
            {
                "base.ref": "main",
                "head.ref": "hotfix/",
                "head.repo": _REPOSITORY,
            },
            False,
        ),
        (
            {
                "base.ref": "main",
                "head.ref": "feature/example",
                "head.repo": _REPOSITORY,
            },
            False,
        ),
        (
            {
                "base.ref": "main",
                "head.ref": "docs/agents-comprehensive",
                "head.repo": _REPOSITORY,
            },
            False,
        ),
        ({"base.ref": "unknown"}, False),
        ({"head.repo": None}, False),
        ({"base.repo.id": 999}, False),
        ({"head.repo.id": 0}, False),
        ({"head.repo.full_name": "malformed"}, False),
        ({"head.ref": ""}, False),
        ({"head.ref": "feature/../spoof"}, False),
        ({"head.sha": "not-a-commit"}, False),
        ({"user.id": True}, False),
        ({"user.login": ""}, False),
        ({"state": "closed"}, False),
        ({"number": "42"}, False),
    ],
)
def test_branch_routing_matrix(tmp_path, changes, allowed):
    """Apply the reviewed matrix and reject incomplete identities."""
    result = _run(tmp_path, _replace(_pull_request(), changes))
    assert result.returncode == (0 if allowed else 1), result.stderr
    assert json.loads(result.stdout)["allowed"] is allowed


@pytest.mark.parametrize("actor", ["maintainer", "dependabot[bot]"])
@pytest.mark.parametrize(
    ("author", "same_repository", "allowed"),
    [
        (
            {"id": 49699333, "login": "dependabot[bot]", "type": "Bot"},
            True,
            True,
        ),
        (
            {"id": 49699333, "login": "dependabot[bot]", "type": "Bot"},
            False,
            False,
        ),
        ({"id": 7, "login": "dependabot[bot]", "type": "User"}, True, False),
        ({"id": 7, "login": "dependabot[bot]", "type": "Bot"}, True, False),
    ],
)
def test_dependabot_uses_authenticated_author_not_rerun_actor(
    tmp_path, actor, author, same_repository, allowed
):
    """Require the bot account and repository, independent of the actor."""
    changes = {
        "base.ref": "main",
        "user": author,
        "head.ref": "dependabot/pip/example",
    }
    if same_repository:
        changes["head.repo"] = _REPOSITORY
    result = _run(tmp_path, _replace(_pull_request(), changes), actor=actor)
    assert result.returncode == (0 if allowed else 1), result.stderr
    assert json.loads(result.stdout)["allowed"] is allowed


@pytest.mark.parametrize(
    "changes",
    [
        {"base.ref": "main"},
        {"head.sha": "b" * 40},
        {"head.repo.id": 789},
        {"number": 43},
    ],
)
@pytest.mark.parametrize("stage", ["current", "final"])
def test_stale_metadata_cannot_authorize_another_pr_state(
    tmp_path, changes, stage
):
    """Reject changes in head, base, repository, or PR association."""
    original = _pull_request()
    result = _run(tmp_path, original, **{stage: _replace(original, changes)})
    assert result.returncode == 1
    assert json.loads(result.stdout)["allowed"] is False


def test_branch_metacharacters_remain_data(tmp_path):
    """Pass a valid branch containing shell syntax without executing it."""
    marker = tmp_path / "executed"
    branch = f"feature/$(touch${{IFS}}{marker})"
    result = _run(tmp_path, _replace(_pull_request(), {"head.ref": branch}))
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


def test_an_api_error_object_cannot_authorize_a_pull_request(tmp_path):
    """Fail closed when a lookup supplies an error instead of PR metadata."""
    result = _run(tmp_path, _pull_request(), current={"message": "Not Found"})
    assert result.returncode == 1
    assert json.loads(result.stdout)["allowed"] is False


def _workflow(name):
    """Read workflow values without YAML's implicit boolean conversion."""
    path = _CHECKER.parents[1] / "workflows" / name
    return yaml.load(path.read_text(), Loader=yaml.BaseLoader)


def test_policy_workflow_uses_trusted_code_with_read_only_permissions():
    """Prevent a privileged policy job from checking out or executing a PR."""
    workflow = _workflow("branch-policy.yml")
    assert set(workflow["on"]) == {"pull_request_target"}
    assert set(workflow["on"]["pull_request_target"]["types"]) == {
        "opened",
        "synchronize",
        "reopened",
        "edited",
    }
    assert all(value == "read" for value in workflow["permissions"].values())
    for job in workflow["jobs"].values():
        assert all(
            value == "read" for value in job.get("permissions", {}).values()
        )
        checkouts = [
            step
            for step in job["steps"]
            if step.get("uses", "").startswith("actions/checkout@")
        ]
        assert len(checkouts) == 1
        assert checkouts[0]["with"]["ref"] == "main"
        assert checkouts[0]["with"]["persist-credentials"] == "false"
        commands = "\n".join(step.get("run", "") for step in job["steps"])
        assert "${{" not in commands, "untrusted expressions belong in env"
        assert '"$GITHUB_EVENT_PATH"' in commands
        assert "download-artifact" not in str(job["steps"])
        assert "pip install" not in commands


def test_development_ci_keeps_main_and_pull_request_triggers():
    """Cover dev pushes without replacing main or pull-request CI."""
    for name in ("ci-cpp.yml", "ci-python.yml"):
        triggers = _workflow(name)["on"]
        assert {"dev", "main"}.issubset(triggers["push"]["branches"])
        assert "pull_request" in triggers


def test_dependabot_version_updates_target_development():
    """Route ordinary dependency updates to the integration branch."""
    path = _CHECKER.parents[1] / "dependabot.yml"
    updates = yaml.safe_load(path.read_text())["updates"]
    assert all(update["target-branch"] == "dev" for update in updates)
