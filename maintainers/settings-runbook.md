<!-- maintainers/settings-runbook.md -->

# Repository settings runbook

Repository settings that a review of the project raised. These are account
settings, not code, so nothing in the repository applies them. Run each
command yourself with a GitHub CLI session that has admin rights on
`abhiksark/swage`, then run the check under it.

The commands use `gh api` only, so they work with an old GitHub CLI.

## 1. Require approval for every outside pull request

The self-hosted GPU runner is registered at repository level on a public
repository. `ci-gpu.yml` only runs on `main` by dispatch or schedule, but
`ci-python.yml` and `ci-cpp.yml` run on pull requests, and a pull request
from a fork can change `runs-on` in those files. With the default policy only
first-time contributors need approval before workflows run.

```bash
gh api -X PUT repos/abhiksark/swage/actions/permissions/fork-pr-contributor-approval \
  -f approval_policy=all_external_contributors
```

Check:

```bash
gh api repos/abhiksark/swage/actions/permissions/fork-pr-contributor-approval
```

Before approving a run on an outside pull request, read its diff under
`.github/workflows/`.

## 2. Isolate the self-hosted runner

This step has no API call. The runner process currently runs under a personal
workstation login. Move it to one of:

- a dedicated unprivileged user with no access to the maintainer's home
  directory, SSH keys, or GitHub CLI session, and with only the GPU device
  and the runner directory available to it; or
- an ephemeral container or virtual machine that is recreated for each job
  (`./config.sh --ephemeral`).

Check, on the runner host:

```bash
ps -o user=,cmd= -C Runner.Listener
```

The user shown must not be a personal login.

## 3. Protect `main`

List the check names that currently report on `main`, then require them.

```bash
gh api repos/abhiksark/swage/commits/main/check-runs --jq '.check_runs[].name'
```

Put the names from that output into `contexts` below. Pull request reviews
are left off because the project has one maintainer.

```bash
gh api -X PUT repos/abhiksark/swage/branches/main/protection --input - <<'JSON'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test (3.10)", "test (3.13)", "docs", "build-and-test"]
  },
  "enforce_admins": false,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false
}
JSON
```

Check:

```bash
gh api repos/abhiksark/swage/branches/main/protection \
  --jq '{checks: .required_status_checks.contexts, force: .allow_force_pushes.enabled}'
```

## 4. Enable private vulnerability reporting

`SECURITY.md` and the issue template send reporters to the private advisory
form. The form only accepts reports when this setting is on.

```bash
gh api -X PUT repos/abhiksark/swage/private-vulnerability-reporting
```

Check:

```bash
gh api repos/abhiksark/swage/private-vulnerability-reporting
```

## 5. Enable secret scanning, push protection, and Dependabot security updates

```bash
gh api -X PATCH repos/abhiksark/swage --input - <<'JSON'
{
  "security_and_analysis": {
    "secret_scanning": {"status": "enabled"},
    "secret_scanning_push_protection": {"status": "enabled"}
  }
}
JSON
gh api -X PUT repos/abhiksark/swage/automated-security-fixes
```

Check:

```bash
gh api repos/abhiksark/swage --jq '.security_and_analysis'
gh api repos/abhiksark/swage/automated-security-fixes
```

## After the settings are applied

Add one sentence to the threat model in `SECURITY.md` stating how the
self-hosted runner is isolated. Do not add it before step 2 is done.
