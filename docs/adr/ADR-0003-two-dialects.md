# ADR-0003: Two dialects instead of one

- Status: accepted
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
- Before `swage_plan` existed, simple lowerings such as one CTA per segment
  went directly from `swage` to standard dialects.

## Implementation note

The narrow `swage_plan` dialect now exists for one admitted identity segmented
sum. It intentionally contains only warp/CTA policy attributes, one opaque
task-range type, and one classification operation. The broader policy, queue,
and dependency surface described above remains deferred.
