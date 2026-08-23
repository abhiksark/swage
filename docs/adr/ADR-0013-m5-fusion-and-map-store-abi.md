# ADR-0013: M5 fusion and map-store ABI

- Status: accepted
- Date: 2026-08-23

## Context

M5 must qualify stable ragged softmax without introducing the planned public
segment frontend, a second production IR, or a general scheduler. The existing
internal segmented runner and five-argument ABI are sufficient if the compiler
can execute ordered reductions, fuse segment maps into their consumers, and
terminate with either one scalar per segment or one value per segment element.

## Decision

The internal segmented lowering admits one fail-closed program shape. A module
contains exactly one single-block function with Swage segment operations,
using the existing five-argument ABI: rank-one f32 values, rank-one i32
offsets, rank-one f32 output, i32 value count, and i32 segment count. It
contains one axis-zero `segment_id`, one `make_segment`, at least one `reduce`,
one return, and exactly one output terminal. Segment regions use only the
admitted f32 arithmetic and `math.exp2` operations. Captures are ordered f32
results of reductions in the same function. Any other shape, operation, type,
capture, or terminal fails before lowering.

Each `swage.map` result has exactly one segment consumer: another map, a
reduction, or `swage.map_store`. The compiler walks map chains in application
order and clones their scalar regions into the consumer's element loop. Mapped
segments are never materialized. Reduction results remain in program order so
later regions receive captures in operand order.

The terminal determines only the output interpretation, not the five-argument
ABI:

- A scalar `memref.store` writes one reduction result to
  `output[segment_id]`, so output has at least one f32 element per segment.
- `swage.map_store` writes one transformed value to `output[element_index]`,
  so output has at least `offsets[-1]` f32 elements. Values beyond the final
  offset are outside the covered range and are not written.

For the map-store path, output must be disjoint from both values and offsets.
Offsets start at zero, are nonnegative and nondecreasing, and end no later than
the values count. Empty segments run the reduction identities, negative
infinity for max and zero for sum, but execute no terminal stores. An all-empty
input therefore has a zero-length covered output.

The CPU lowering executes each segment sequentially, performing reductions in
source order and then the terminal phase. The GPU lowering uses one CTA per
segment. Threads make block-stride passes for maximum, exponential sum, and
normalization/store. Uniform `gpu.all_reduce` operations broadcast each
reduction and provide the phase synchronization; the emitter adds no separate
barrier or scratch allocation.

Stable softmax subtracts the segment maximum before exponentiation. The mapped
exponentials are fused into the sum and recomputed in the terminal store rather
than saved in an intermediate buffer. The semantic program expresses `exp(x)`
as `math.exp2((x - max) * log2(e))`. GPU code generation replaces the
unlinked libdevice call with LLVM's native `exp2` intrinsic, which lowers to
the NVPTX native approximation. NVPTX compilation fails closed for targets
older than `sm_80`; M5 was qualified on an RTX A6000 at `sm_86`.

## Consequences

- The internal runner qualifies ragged softmax against PyTorch and the
  sequential CPU oracle without widening the public M3 launch contract.
- The one-CTA schedule is a qualification schedule, not schedule selection.
  Tiny-segment packing, multi-CTA execution, and persistent scheduling remain
  later work.
- Public segment primitives and public ragged-softmax emission or launch remain
  planned.
- Device-side offset validation, pre-Ampere support, and intermediate-storage
  policies are outside this decision.
