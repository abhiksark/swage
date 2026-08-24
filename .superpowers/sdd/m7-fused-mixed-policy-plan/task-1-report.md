# Task 1 report: predeclare the revised fused schedule

## What changed

Added ADR-0016, which supersedes only ADR-0015's two-launch mixed rule. It
records the existing `1.238806` mixed-to-best-pure failure and predeclares one
128-thread block, four one-segment warp slots per block, and one kernel launch.
The ADR preserves ADR-0015's benchmark constants and acceptance gate and does
not claim M7 completion. Added ADR-0016 to the MkDocs navigation.

## Verification

- `mkdocs build --strict`: passed. MkDocs reported the existing informational
  notice that `github-project.md` is not in navigation and the installed
  Material for MkDocs compatibility warning; there were no build errors.
- `git diff --check`: passed.
- No timing benchmark was run, as required.

## Files changed

- `docs/adr/ADR-0016-m7-fused-mixed-policy-schedule.md`
- `mkdocs.yml`
- `.superpowers/sdd/m7-fused-mixed-policy-plan/task-1-report.md`

## Semantic behavior impact

None in executable code. This is a prose/configuration-only change that
predeclares a revised benchmark schedule.

## Self-review

Confirmed the new ADR has the required repository-relative file header,
contains the exact revised schedule values, records `1.238806`, repeats all
ADR-0015 benchmark constants and the `1.05` gate, and does not add completion
documentation. The diff is whitespace-clean.

## Concerns

None. The only verification notices are the pre-existing MkDocs informational
notice and the installed Material for MkDocs warning described above.
