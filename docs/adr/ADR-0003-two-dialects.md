# ADR-0003: Two dialects instead of one

- Status: accepted (swage_plan deliberately deferred)
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
until the semantic dialect and one fixed GPU lowering work end to end —
no empty scaffolding.

## Consequences

- The semantic level stays analyzable and fusible without schedule noise.
- One extra conversion layer (`SwageToPlan`) once planning lands.
- Until `swage_plan` exists, simple lowerings (one CTA per segment) go
  directly from `swage` to standard dialects.
