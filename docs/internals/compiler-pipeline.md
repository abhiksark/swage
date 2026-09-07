<!-- docs/internals/compiler-pipeline.md -->

# Compiler Pipeline

Swage has one canonical compiler spine. Python source or native test IR
becomes verified semantic MLIR, then an admitted lowering branch uses upstream
MLIR and LLVM infrastructure. There is no second production IR between Python
and MLIR.

Verified semantic MLIR enters one of three admitted branches. The public
fixed-block branch uses one shared canonical vector elementwise admission,
then emits either a CUDA GPU function or a sequential Native host function. Private
direct segmented branches lower to the sequential CPU oracle or the one-CTA
GPU path. The private SwagePlan branch adds the narrow classification
companion for direct or split identity-sum lowering. GPU branches rejoin
upstream GPU, SCF, NVVM, and LLVM lowering before LLVM NVPTX emits PTX for the
CUDA Driver API. The Native fixed branch lowers SCF, control flow, arithmetic,
index, and function operations to LLVM dialect before creating an
`ExecutionEngine`. No branch introduces a second production IR or a silent
backend fallback.

<div class="doc-figure" tabindex="0" markdown="1">

![Verified semantic MLIR entering three admitted compiler branches](../assets/diagrams/compiler-pipeline.svg)

</div>

*The implemented compiler spine and its public and private branches. [Open the full-size figure](../assets/diagrams/compiler-pipeline.svg).*

## Frontend boundary

`@swage.jit` captures source without executing the body. The frontend parses
a restricted Python AST and constructs a live `mlir_swage.ir.Module` directly
through MLIR Python bindings. The module preserves source locations and must
verify before it crosses the frontend boundary.

The exact accepted source forms live in [Kernel Language](../reference/kernel-language.md).
The compile-only and execution call contracts live in
[swage](../reference/swage.md).

## Semantic MLIR boundary

The `swage` dialect represents logical fixed-block and segment semantics.
Runtime segment identity is carried by SSA values, not types. Region-based
maps and reductions remain symbolic until an admitted lowering handles them.
Ordinary arithmetic, loops, buffers, and backend operations use upstream
dialects.

[Swage Dialect](../internals/swage-dialect.md) owns the current operation and
type surface. [SwagePlan Dialect](../internals/swage-plan-dialect.md) owns the
small private planning surface.

## Public fixed-block branch

The fixed-block conversion admits only canonical vector add or multiply. The
CUDA pass maps each vector lane to one GPU x-thread, lowers through upstream
GPU, SCF, NVVM, and LLVM infrastructure, and emits PTX in process. The Native
host pass uses the same admission and emits a sequential pointer loop that is
lowered to an eagerly initialized process-local LLVM JIT executable. The
public runtime selects exactly one branch from `backend="cuda"` or
`backend="cpu"`.

The three pointer elements must all be `f32`, `f16`, `f8E4M3FN`, or `f8E5M2`.
Shared scalar emission keeps native FP32 operations, widens FP16 loads for
FP32 arithmetic before rounding the store, and implements FP8 conversion using
byte loads/stores and scalar arithmetic. FP8 never reaches LLVM as an
unsupported floating type and does not require newer FP8 hardware.
The semantic module still contains a same-element-type vector add or multiply;
these are physical realizations of its rounding contract.

No `nvgpu` conversion is part of the CUDA branch. Runtime specialization,
cache, executable ownership, module loading, stream, and retention behavior
live in [Runtime and Environment](../reference/runtime-environment.md).

## Lowered launch-contract boundary

Each physical branch creates one concrete typed function and a canonical
version-2 compiler-generated launch contract from an ordered argument
specification. The contract identifies the backend, entry, argument kinds,
user/derived/plan/scratch bindings, access, and a launch union: CUDA uses
`spmd-grid` with three-axis block geometry; CPU uses `host-call` with no
block. The C API validates the contract against the physical function and,
for CUDA, `nvvm.reqntid`; removes its temporary MLIR attribute; and returns
canonical JSON beside the lowered module and executable image. Runtime
binding uses this metadata and does not infer an ABI from parameter names,
PTX, or LLVM text.

The contract describes one lowered entry. It is not semantic IR, does not
widen admission, and does not replace concrete physical parameter types.

## Private direct segmented branches

Canonical segmented sum, max, and stable ragged-softmax modules enter through
native qualification, not the public Python frontend. One conversion creates
a sequential CPU correctness oracle. Another creates one CTA per segment and
continues through upstream GPU, NVVM, LLVM, and NVPTX stages.

Exact admitted module shapes and internal ABIs live in
[Segmented Reductions](segmented-reductions.md) and
[Ragged Softmax](ragged-softmax.md).

## Private SwagePlan branch

For one canonical identity segmented sum, admission can add a private planning
companion without mutating the semantic function. Validated host metadata is
then classified and materialized into direct IDs or split records. Private
lowering factories produce the direct, partial, and merge kernels used by the
qualification runtime.

This branch implements narrow rule-based classification and split task
decomposition. One private experimental lowering consumes those materialized
descriptors through persistent device claim counters and split completion
publication; its predeclared performance gate failed. The branch does not
implement general cost inference, general schedule selection, packing, or a
public reusable queue.

## Ownership boundary

Swage currently owns semantic operations, fail-closed admission, the narrow
planning record, host descriptor materialization, the dedicated conversions
required by qualified paths, and the launch runtime's validation, cache, and
dispatch. Upstream MLIR and LLVM own ordinary
arithmetic, control flow, memory operations, GPU lowering infrastructure,
LLVM IR, and NVPTX emission. PyTorch owns tensors, the active CUDA context,
and the current stream.

<div class="doc-figure" tabindex="0" markdown="1">

![Three ownership lanes with one CUDA launch traced across the domains](../assets/figures/ownership-map.svg)

</div>

*What Swage owns against upstream MLIR and LLVM, and PyTorch. [Open the full-size figure](../assets/figures/ownership-map.svg).*

Continue with [Swage Dialect](swage-dialect.md) for the semantic
surface, [Compiler Tools and Passes](compiler-tools.md) for the
command-line surface, or [Verification](verification.md) to audit the
executable gates for each branch.
