<!-- docs/internals/task-execution.md -->

# Task Execution

Classified warp and CTA tasks execute through fixed tiles, either
as pure launches or one fused mixed launch. This page records the
exact internal contracts; none of them is a public API.

*Qualified under the M7 gate on NVIDIA RTX A6000 (`sm_86`);
see [`ROADMAP.md`](https://github.com/abhiksark/swage/blob/main/ROADMAP.md) and
[ADR-0015](../adr/ADR-0015-m7-minimal-mixed-policy-execution.md) and
[ADR-0016](../adr/ADR-0016-m7-fused-mixed-policy-schedule.md).*

Pure warp and pure CTA qualification use this task-ID ABI:

```text
values*, offsets*, output*, task_ids*, value_count:i32, task_count:i32
```

<div class="doc-figure" tabindex="0" markdown="1">

![A 32-thread warp tile with an xor shuffle butterfly, a 128-thread CTA tile striding passes, and a 512-thread split tile covering one chunk](../assets/figures/warp-vs-cta-tiles.svg)

</div>

*Fixed physical tile shapes for warp, CTA, and split tasks under the default M6 and M8 limits. [Open the full-size figure](../assets/figures/warp-vs-cta-tiles.svg).*

The warp kernel uses 32 threads. The CTA kernel uses 128 threads. Fused mixed
execution uses one 128-thread kernel and this ABI:

```text
values*, offsets*, output*, task_ids*, value_count:i32,
warp_task_count:i32, cta_task_count:i32
```

Each initial block contains four independent one-segment warp slots. CTA
tasks follow at one segment per block. An empty task set enqueues no kernel.
The M7 qualification is limited to canonical identity sum.

<div class="doc-figure" tabindex="0" markdown="1">

![One fused launch covering four-per-block warp tasks then one-per-block CTA tasks](../assets/figures/fused-mixed-schedule.svg)

</div>

*The one-launch fused schedule and its task-ID indirection. [Open the full-size figure](../assets/figures/fused-mixed-schedule.svg).*

The frozen NVIDIA RTX A6000 `sm_86` benchmark reports medians of
`0.067584 ms` for pure warp, `0.070656 ms` for pure CTA, and `0.063488 ms`
for fused mixed execution. The fused schedule it measures is the one drawn
above. The mixed-to-best-pure ratio is `0.939394`, which passes the
predeclared maximum of `1.05`. The committed raw record is
[`benchmarks/results/m7-a6000-sm86.json`](https://github.com/abhiksark/swage/blob/main/benchmarks/results/m7-a6000-sm86.json).

Continue with [Split Execution](split-execution.md) for oversized
segments, or [Benchmarks](benchmarks.md) for the recorded campaign.
