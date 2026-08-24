<!-- docs/concepts/swage-visual-guide.md -->

# Swage, Visually

Swage begins with a mismatch. Applications often store variable-sized
segments, while GPUs run most efficiently when work has a fixed shape.
Swage keeps those concerns separate:

- A segment-local program describes **what** one logical segment means.
- A planner describes **which work** must run for the actual segment lengths.
- GPU lowering describes **how** fixed warps and CTAs execute that work.

The result is one stable semantic description with room for different GPU
schedules.

!!! note "Read the status labels literally"

    **Public today** means an application can use the supported API.
    **Private qualification** means tested compiler and runtime machinery that
    is not a public API. **Planned** means design direction, not implemented
    behavior.

    This guide explains all three levels without presenting private segmented
    execution as public functionality.

## Begin with one ragged buffer

Variable-sized segments can share one dense values buffer. An offsets array
marks where each segment begins and ends:

    values  = [ 4.0, 1.0 | | 3.0, 8.0, 2.0 | ... ]
    offsets = [ 0,        2, 2,            5, ... ]

Segment `i` is the half-open slice
`values[offsets[i] : offsets[i + 1]]`. Its elements are dense, but its
length is known only at runtime. A segment may also be empty, as shown by the
repeated offset `2`.

This storage gives the compiler the logical problem. It does not force every
segment to use the same amount of GPU work.

## Keep segments, tasks, and tiles separate

- A **segment** answers: what logical data does this program operate on?
- A **task** answers: what schedulable unit must execute for this segment?
- A **tile** answers: what fixed physical shape does a warp or CTA process?

This separation is the central Swage idea. A runtime-sized segment never
becomes a runtime-sized register array, and thread or block IDs never enter
semantic Swage IR. The planner is therefore free to change the task plan
without changing the segment-local meaning.

## Follow one batch from meaning to execution

For the current private identity-sum qualification, the default warp limit is
32 elements and the default CTA chunk limit is 4096 elements:

- A length from 0 through 32 produces one direct warp task.
- A length from 33 through 4096 produces one direct CTA task.
- A length above 4096 produces ordered partial CTA tasks and one merge task.

Partial tasks cover the original values with absolute, non-overlapping ranges.
The 4096-element limit describes the range a CTA traverses, not its thread
count. Each 128-thread partial CTA writes one unique value to global scratch.
A later 128-thread merge CTA reduces the corresponding compact scratch range
and writes the segment output once.

<div
  role="region"
  aria-label="Segments, tasks, and tiles diagram"
  tabindex="0"
  style="overflow-x: auto;"
>
  <img
    src="../../assets/diagrams/segments-tasks-tiles.svg"
    alt="Four variable-sized segments mapped through direct and split tasks to fixed 32-value and 128-value tiles."
    loading="lazy"
    style="width: 100%; min-width: 60rem; height: auto;"
  >
</div>

*Segments carry meaning, tasks carry scheduling decisions, and tiles carry
fixed hardware execution. [Open the full-size
diagram](../assets/diagrams/segments-tasks-tiles.svg).*

Mixed execution submits direct fused work, then all partial CTAs, then all
merge CTAs on the same current PyTorch stream. If no segment needs splitting,
the existing M7 one-launch path stays unchanged.

Split-CTA is task decomposition under the existing CTA policy, not a third
policy. This path is a private correctness qualification for canonical
identity sum. It is not a public segmented API, and it does not cover split
max or split softmax.

## Follow the compiler journey

<div
  role="region"
  aria-label="Swage compiler journey diagram"
  tabindex="0"
  style="overflow-x: auto;"
>
  <img
    src="../../assets/diagrams/compiler-journey.svg"
    alt="The public fixed vector-add path and the separate M4 to M8 private qualification paths through Swage MLIR, LLVM NVPTX, and the current PyTorch stream."
    loading="lazy"
    style="width: 100%; min-width: 60rem; height: auto;"
  >
</div>

*[Open the compiler journey at full
size](../assets/diagrams/compiler-journey.svg).*

The two solid lanes have different entry points:

1. **Public today:** the restricted `@sw.jit` frontend captures the canonical
   fixed vector-add subset and builds verified Swage MLIR. `emit_mlir()`
   stops at that compile-only boundary. Keyword-only `launch()` lowers and
   executes that same fixed subset.
2. **Private qualification:** canonical segmented modules exercise sum, max,
   ragged softmax, host classification, mixed warp and CTA work, and split-CTA
   identity sum. These helpers do not widen the public frontend or launch API.

After admission, GPU branches use standard MLIR, LLVM, and NVPTX
infrastructure to produce PTX for the exact active target. The M4 and M5 CPU
branch instead lowers to a sequential correctness oracle. The GPU runtime
validates arguments and launches asynchronously through the CUDA Driver API on
the current PyTorch stream. It does not silently copy, cast, synchronize, or
fall back to another backend.

## Know the boundary today

| Surface | Status | Current meaning |
|---|---|---|
| `@sw.jit` and `emit_mlir()` | Public today | Capture and compile the supported fixed vector-add subset through native bindings |
| Keyword-only `launch()` | Public today | Execute the canonical fixed vector add on CUDA tensors |
| Segmented sum, max, and ragged softmax | Private qualification | Compare canonical internal CPU and GPU paths against correctness oracles |
| Warp, CTA, and split-CTA identity sum | Private qualification | Qualify M6 to M8 classification, direct work, partial reductions, and merges |
| Public segment syntax and segmented launch | Planned | Let applications express and execute segment-local programs |
| Packed warps and persistent scheduling | Planned | Extend scheduling choices after their own correctness and performance gates |

Swage owns segment semantics, task planning, and the boundary between them.
Upstream MLIR and LLVM own general arithmetic, control flow, GPU dialect
lowering, and PTX generation. PyTorch supplies tensors, the CUDA context, and
the current stream.

## Carry one mental model forward

> Swage keeps variable-sized segment meaning separate from fixed-size GPU
> work, so the task schedule can change without rewriting the program's
> meaning.

With that picture in place, continue through the documentation in this order:

1. [Quickstart](../quickstart.md) runs the supported public fixed vector-add
   path.
2. [Segments, Tasks, and Tiles](segments-tiles-tasks.md) defines the three
   concepts and their invariants.
3. [Compiler pipeline](../architecture/compiler-pipeline.md) traces each
   implemented milestone through the compiler and runtime.
4. [DESIGN.md](https://github.com/abhiksark/swage/blob/main/DESIGN.md)
   records the architecture and the boundaries that future work must preserve.
