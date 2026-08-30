<!-- docs/internals/segmented-reductions.md -->

# Segmented Reductions

Canonical segmented sum and max execute through a sequential CPU
oracle and a one-CTA GPU path. This page records the exact internal
contracts; none of them is a public API.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) for the executable evidence.*

The admitted semantic module has one axis-zero segment ID, one segment over
rank-one f32 values and rank-one i32 offsets, one capture-free identity
reduction of kind `sum` or `max`, one rank-one f32 output, and explicit i32
value and segment counts.

The internal ABI is:

```text
values*, offsets*, output*, value_count:i32, segment_count:i32
```

The CPU path lowers to sequential SCF and memref operations and executes with
upstream `mlir-runner`. The GPU path uses one CTA per segment and block-stride
loads. Empty sums produce zero; empty maxima produce negative infinity. Max
uses NaN-propagating semantics.

Continue with [Ragged Softmax](ragged-softmax.md) for the fused
multi-phase case, or [Verification](verification.md) for the oracle
topology behind these claims.
