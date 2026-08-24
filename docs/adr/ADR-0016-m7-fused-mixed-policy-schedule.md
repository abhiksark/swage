<!-- docs/adr/ADR-0016-m7-fused-mixed-policy-schedule.md -->
# ADR-0016: M7 fused mixed-policy schedule

- Status: accepted
- Date: 2026-08-24

## Context

ADR-0015 predeclared a two-launch mixed policy: warp tasks launch first and
CTA tasks launch second. The existing mixed-policy timing record has a
mixed-to-best-pure ratio of `1.238806`, which fails the predeclared `1.05`
gate. The failure is recorded here before any new benchmark timing run.

## Decision

ADR-0016 supersedes only ADR-0015's two-launch mixed rule. The revised fused
mixed schedule is:

- one 128-thread block;
- four one-segment warp slots per block;
- one kernel launch.

The semantic program, task-ID ABI, pure-policy schedules, validation rules,
and public API boundary remain unchanged. This ADR predeclares the revised
schedule; it does not establish a performance result or claim M7 completion.

## Unchanged benchmark contract

The timing run uses exactly:

- NVIDIA RTX A6000 at `sm_86`;
- the `bimodal` generator with seed 7 and 32,768 segments;
- a warp threshold of 32;
- 32 threads for the warp policy and 128 threads for the CTA policy;
- 25 warmups for each pure or mixed policy;
- 100 interleaved CUDA-event samples per policy;
- all-one f32 input values, making each exact expected sum equal to its
  segment length.

Compilation, module loading, classification, and allocation occur before
timing and are excluded. The committed record at
`benchmarks/results/m7-a6000-sm86.json` contains every raw sample, environment
metadata, generated distribution statistics, policy medians, and the mixed
to best-pure ratio.

M7 passes only when:

```text
mixed_median <= 1.05 * min(pure_warp_median, pure_cta_median)
```

If this gate fails, the distribution and policy constants above do not
change. The implementation must improve or M7 remains open.

## Consequences

- Mixed execution has one launch with one 128-thread block and four
  one-segment warp slots per block.
- The existing two-launch mixed schedule is no longer the predeclared M7
  schedule.
- This decision does not add completion documentation or establish a new
  benchmark result.
