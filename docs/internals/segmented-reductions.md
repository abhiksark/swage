<!-- docs/internals/segmented-reductions.md -->

# Segmented Reductions

Canonical segmented sum and max execute through a sequential CPU
oracle and a one-CTA GPU path. This page records the exact internal
contracts; none of them is a public API.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) for the executable evidence.*

The admitted semantic module has exactly three user buffers: rank-one f32
values, rank-one i32 offsets, and rank-one f32 output. It has one axis-zero
segment ID, one segment, and one capture-free reduction of kind `sum` or
`max`, with an optional element expression and single-consumer map chains.
Buffer roles come from types, effects, and operation dataflow, not parameter
names or positions; ambiguous programs fail during read-only admission.

The semantic ABI is:

```text
values*, offsets*, output*
```

CPU and planning lowerings derive value and segment counts from memref
dimensions. GPU lowerings retain concrete i32 count parameters in their
physical PTX signatures, where compiler-generated launch contracts mark them
as derived runtime bindings.

The CPU path lowers to sequential SCF and memref operations and executes with
upstream `mlir-runner`. The GPU path uses one CTA per segment and block-stride
loads. Empty sums produce zero; empty maxima produce negative infinity. Max
uses NaN-propagating semantics. The same single-stage programs also support
private warp, fused mixed, and split execution; see [Task Planning](planning.md)
and [Split Execution](split-execution.md).

Continue with [Ragged Softmax](ragged-softmax.md) for the fused
multi-phase case, or [Verification](verification.md) for the oracle
topology behind these claims.
