<!-- docs/adr/ADR-0003-two-dialects.md -->
# ADR-0003: Two dialects instead of one

- Status: accepted; `swage_plan` was introduced by
  [ADR-0014](ADR-0014-minimal-swage-plan-gate.md); see
  [Later decisions](#later-decisions)
- Date: 2026-08-18

## Context

Segment semantics ("what happens to one segment") and scheduling ("how
segments become tasks and tiles") could share one dialect, but their
invariants differ: semantic IR must stay schedule-free, while planning IR
is all about schedules, policies, and dependencies.

## Decision

Two dialects. `swage` carries segment-local semantics with region-based
map/reduce structure and no hardware notions. `swage_plan` will carry
tasks, policies (`warp`, `packed_warp`, `cta`, `split_cta`, `merge`,
`persistent`), queues, and dependencies. `swage_plan` is not introduced
until the semantic dialect and one fixed GPU lowering work end to end,
with no empty scaffolding.

## Consequences

- The semantic level stays analyzable and fusible without schedule noise.
- One extra conversion layer (`SwageToPlan`) once planning lands.
- Until `swage_plan` exists, simple lowerings (one CTA per segment) go
  directly from `swage` to standard dialects.

## Later decisions

The decision stands. The dialect that exists today is narrower than the
surface this record anticipated:

- [ADR-0014](ADR-0014-minimal-swage-plan-gate.md) introduced `swage_plan`
  with two policies, `warp` and `cta`, one task-range type, and one
  classification operation.
- [ADR-0017](ADR-0017-private-split-cta-reductions.md) added split
  execution as task decomposition under the `cta` policy. `split_cta` and
  `merge` did not become policies.
- [ADR-0018](ADR-0018-private-persistent-task-queue.md) proposes a
  persistent queue as a private experiment. `persistent` did not become a
  policy, and that record remains proposed.
- No record has added `packed_warp`, queues, or dependencies to the
  dialect.

Until [ADR-0020](ADR-0020-planned-per-function-lowering.md), the
`--swage-to-plan` conversion added a private companion function that held
the classification operation, and no lowering consumed `swage_plan` IR.
ADR-0020 made `swage_plan` a pipeline stage and removed the companion and
the classification operation. `--swage-to-plan` now replaces each segment
function by plan functions, one per schedule, with five task operations;
`--swage-plan-to-gpu` lowers every segmented GPU kernel from them, and
`--swage-plan-to-scf` lowers the sequential CPU oracle. The decision of
this record stands: the plan dialect carries the schedule, and the
semantic dialect stays free of it.
