<!-- docs/internals/index.md -->

# Internals

Internals documents the compiler and runtime machinery behind the public
surface. None of it is public API. The segmented pages record exact
internal compiler and runtime contracts that exist to qualify semantics,
lowering, planning, and execution. The two public segmented calls run fixed
programs through this machinery with its default limits;
[swage](../reference/swage.md) states their contract, and nothing on these
pages widens it.

Readers arriving from an older link to the private qualification page can
find its content in the topic pages below.

| Page | Covers |
|---|---|
| [Compiler Pipeline](compiler-pipeline.md) | The spine, the admitted branches, and ownership |
| [Swage Dialect](swage-dialect.md) | The semantic operations and types |
| [Textual Swage IR](../language/swage-ir.md) | Textual syntax, SSA data flow, and one complete module |
| [Segmented Reductions](segmented-reductions.md) | Direct segmented sum and max, CPU oracle, one CTA per segment, and sum rounding |
| [Ragged Softmax](ragged-softmax.md) | Fused multi-phase softmax in one CTA, and its accuracy |
| [SwagePlan Dialect](swage-plan-dialect.md) | The private planning IR surface |
| [Task Planning](planning.md) | Classification of segments into tasks |
| [Task Execution](task-execution.md) | Warp, CTA, and fused mixed launches |
| [Split Execution](split-execution.md) | Oversized segments, partials, and merges |
| [Persistent Execution](persistent-execution.md) | The experimental resident-queue kernel, whose performance gate failed |
| [Compiler Tools and Passes](compiler-tools.md) | `swage-opt`, the registered passes, the CMake package, and lit features |
| [Verification](verification.md) | The claim-to-test evidence matrix |
| [Benchmarks](benchmarks.md) | The benchmark harnesses and the recorded performance campaign |
| [A6000 Comparison Study](a6000-comparison.md) | One exploratory Swage and Triton campaign on skewed distributions |

Use [Decisions](../decisions/index.md) for the rationale behind each
boundary and [Verification](verification.md) for its executable evidence.

Continue with [Compiler Pipeline](compiler-pipeline.md).
