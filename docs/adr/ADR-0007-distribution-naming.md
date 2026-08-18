# ADR-0007: Distribution and package naming

- Status: accepted
- Date: 2026-08-18

## Context

The brand is **Swage**, the import is `swage`, the repository is
`abhiksark/swage`, and the tools are `swagec` / `swage-opt`. PyPI already
has an unrelated project named `swage` (checked 2026-08-18, HTTP 200 on
`pypi.org/pypi/swage/json`), so the distribution cannot use the bare name.

## Decision

- PyPI distribution name: **`swage-compiler`** (verified available,
  HTTP 404 at decision time).
- Python import stays `swage`; a distribution name and import name need
  not match.
- Command-line names remain `swagec` (compiler, future) and `swage-opt`
  (optimizer driver, exists).

## Consequences

- `pip install swage-compiler` / `import swage`; docs must show both to
  avoid confusion with the unrelated `swage` project.
- The name is claimed on PyPI at first release, not before there is
  something installable worth publishing.
