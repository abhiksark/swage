#!/usr/bin/env bash
# .github/scripts/check-pr-branch-policy.sh
# Read event and API JSON as data; execute no pull-request source.
set -eu
exec python3 - "$@" <<'PY'
"""Evaluate branch routing with complete, current GitHub identities."""

import json
import os
import pathlib
import re
import sys


def positive_integer(value):
    """Return whether a GitHub numeric identity is a positive integer."""
    return type(value) is int and value > 0


def repository(record):
    """Return the validated repository identity, rejecting deleted heads."""
    if not isinstance(record, dict) or not positive_integer(record.get("id")):
        raise ValueError("missing or invalid repository identity")
    name = record.get("full_name")
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", name
    ):
        raise ValueError("missing or invalid repository name")
    return record["id"], name.casefold()


def branch(value):
    """Validate a branch ref without interpreting it as shell code."""
    if (
        not isinstance(value, str)
        or not value
        or value == "@"
        or value.startswith(("-", "/"))
        or value.endswith(("/", "."))
        or any(part in value for part in ("..", "@{", "//"))
        or any(ord(char) <= 32 or ord(char) == 127 for char in value)
        or any(char in value for char in "~^:?*[\\")
        or any(
            part.startswith(".") or part.endswith(".lock")
            for part in value.split("/")
        )
    ):
        raise ValueError("missing or invalid branch ref")
    return value


def identity(record, trusted_repository):
    """Validate a current PR and return its routing and association key."""
    if (
        not positive_integer(record.get("number"))
        or record.get("state") != "open"
    ):
        raise ValueError("pull request must be open with a numeric identity")
    base, head, author = record["base"], record["head"], record["user"]
    base_repository = repository(base["repo"])
    if base_repository != trusted_repository:
        raise ValueError(
            "base repository differs from trusted workflow context"
        )
    head_repository = repository(head["repo"])
    base_ref, head_ref = branch(base["ref"]), branch(head["ref"])
    sha = head.get("sha")
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("missing or invalid head commit")
    login = author.get("login")
    if (
        not positive_integer(author.get("id"))
        or author.get("type") not in ("User", "Bot")
        or not isinstance(login, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[bot\])?", login)
    ):
        raise ValueError("missing or invalid pull-request author")
    return (
        record["number"],
        base_ref,
        base_repository,
        head_ref,
        sha,
        head_repository,
        author["id"],
        login,
        author["type"],
    )


def evaluate(event, current, final):
    """Allow only the reviewed routing matrix on one unchanged PR state."""
    trusted_repository = repository(
        {
            "id": int(os.environ["GITHUB_REPOSITORY_ID"]),
            "full_name": os.environ["GITHUB_REPOSITORY"],
        }
    )
    if repository(event["repository"]) != trusted_repository:
        raise ValueError(
            "event repository differs from trusted workflow context"
        )
    original = identity(event["pull_request"], trusted_repository)
    if original != identity(
        current, trusted_repository
    ) or original != identity(final, trusted_repository):
        raise ValueError(
            "pull-request state changed; a fresh event is required"
        )
    _, base_ref, _, head_ref, _, head_repository, author_id, login, kind = (
        original
    )
    if base_ref == "dev":
        return True, "ordinary pull request targets dev"
    if base_ref != "main":
        return False, "unknown base branch"
    if head_repository != trusted_repository:
        return False, "main requires a head from this repository"
    if head_ref == "dev" or head_ref.startswith("hotfix/"):
        return True, "repository dev or hotfix branch targets main"
    # GitHub's immutable account ID distinguishes the bot from name spoofs.
    # The PR author remains authoritative when a maintainer reruns a job.
    if (author_id, login, kind) == (49699333, "dependabot[bot]", "Bot"):
        return True, "authenticated repository Dependabot pull request"
    return False, "ordinary pull requests must target dev"


def main():
    """Read three JSON snapshots and report an advisory routing decision."""
    try:
        if len(sys.argv) != 4:
            raise ValueError("expected EVENT_JSON CURRENT_JSON FINAL_JSON")
        records = [
            json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
            for path in sys.argv[1:]
        ]
        if not all(isinstance(record, dict) for record in records):
            raise ValueError("metadata must hold JSON objects")
        allowed, reason = evaluate(*records)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        allowed, reason = False, f"invalid or stale metadata: {error}"
    print(json.dumps({"allowed": allowed, "reason": reason}))
    return 0 if allowed else 1


sys.exit(main())
PY
