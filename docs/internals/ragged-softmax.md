<!-- docs/internals/ragged-softmax.md -->

# Ragged Softmax

Stable ragged softmax executes all phases in one CTA per segment
with fused maps. This page records the exact internal contracts;
none of them is a public API.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) and
[ADR-0013](../adr/ADR-0013-fusion-and-map-store-abi.md).*

The softmax path retains the five-argument ABI defined in
[Segmented Reductions](segmented-reductions.md). It admits ordered f32
reduction captures,
single-consumer map chains, and exactly one scalar store or map-store
terminal. The stable softmax module performs maximum, shifted exponential
sum, and normalize/store phases. Maps are fused into their consumer.

The CPU path executes the phases sequentially. The GPU path executes all
phases in one CTA per segment. Exact-target GPU compilation requires `sm_80`
or newer because every Swage compile request must name a processor from the
admitted list in [Runtime and Environment](../reference/runtime-environment.md).
The floor is not specific to this path: native `exp2` on f32 is legal for
every processor in the pinned NVPTX backend. The runner validates host-visible
offset metadata, capacity, and output disjointness. Empty segments perform no
map-store writes.

<div class="doc-figure" tabindex="0" markdown="1">

![Three block-stride passes per segment separated by uniform all-reduce operations](../assets/figures/ragged-softmax-phases.svg)

</div>

*The one-CTA softmax schedule, with all-reduce as broadcast and phase barrier. [Open the full-size figure](../assets/figures/ragged-softmax-phases.svg).*

Continue with [Task Planning](planning.md) for how segments become
schedulable tasks.
