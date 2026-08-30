<!-- docs/internals/index.md -->

# Internals

Internals documents the compiler and runtime machinery behind the public
surface. None of it is public API. The segmented pages record exact
internal compiler and runtime contracts that exist to qualify semantics,
lowering, planning, and execution before any public surface is designed.

Readers arriving from an older link to the private qualification page can
find its content in the topic pages below.

| Page | Covers |
|---|---|
| [Compiler Pipeline](compiler-pipeline.md) | The spine, the admitted branches, and ownership |
| [Swage Dialect](swage-dialect.md) | The semantic operations and types |
| [Segmented Reductions](segmented-reductions.md) | Direct segmented sum and max, CPU oracle, one CTA per segment |
| [Ragged Softmax](ragged-softmax.md) | Fused multi-phase softmax in one CTA |
| [Task Planning](planning.md) | Classification of segments into tasks |
| [Task Execution](task-execution.md) | Warp, CTA, and fused mixed launches |
| [Split Execution](split-execution.md) | Oversized segments, partials, and merges |
| [SwagePlan Dialect](swage-plan-dialect.md) | The private planning IR surface |
| [Compiler Tools and Passes](compiler-tools.md) | `swage-opt` and the registered passes |
| [Verification](verification.md) | The claim-to-test evidence matrix |
| [Benchmarks](benchmarks.md) | The recorded performance campaign |

Use [Decisions](../decisions/index.md) for the rationale behind each
boundary and [Verification](verification.md) for its executable evidence.

Continue with [Compiler Pipeline](compiler-pipeline.md).
