<!-- docs/adr/ADR-0019-compiler-generated-kernel-contracts.md -->
# ADR-0019: Compiler-generated kernel launch contracts

- Status: accepted
- Date: 2026-09-04

## Context

Every admitted physical lowering emits a concrete entry point. Runtime code
must bind user tensors, metadata-derived counts, prepared-plan buffers, and
runtime-owned scratch without restating the compiler's physical argument
order. That description must cover explicit CUDA and Native CPU execution
without becoming a second program IR, a public kernel API, or a reason to
weaken lowering admission.

## Decision

Every admitted physical lowering constructs one versioned `KernelContract`
from the same ordered argument specification used to create its concrete
function. Version 2 records:

- backend, exactly `cuda` or `cpu`;
- the exact entry symbol;
- a launch union: CUDA uses `spmd-grid` with required three-dimensional block
  geometry, while CPU uses `host-call` with no block;
- each physical argument's exact kind: `ptr`, `i1`, `i8`, `i16`, `i32`, `i64`,
  `f16`, `bf16`, `f32`, or `f64`;
- its `user`, `derived`, `plan`, or `scratch` origin;
- a user source index or stable non-user binding key; and
- pointer access as `read`, `write`, or `readwrite`.

The lowering temporarily attaches the contract as the discardable built-in
`swage.kernel_contract` dictionary attribute. The C API requires exactly one
contract-bearing entry, validates the contract against the concrete function
type, and erases the attribute before downstream lowering. CUDA additionally
validates `nvvm.reqntid`; Native host lowering validates the `host-call`
contract before lowering to LLVM dialect and creating an `ExecutionEngine`.
Both paths return deterministic canonical JSON beside their lowered module and
executable image.

Parsing is strict. Unknown versions, fields, backends, launch models, kinds,
origins, access modes, duplicate bindings, invalid source indexes, malformed
geometry, and type or entry mismatches fail before executable loading or
invocation. Context-inapplicable fields also fail: CUDA requires an
`spmd-grid` block and CPU `host-call` forbids one.

The compiled artifact stores the canonical JSON and its SHA-256 digest. The
digest participates in artifact identity and CUDA persistent-cache
verification. Runtime code binds values by origin and materializes one ordered
raw-bit sequence. CUDA submits that sequence through equivalent nanobind
`_launch_cuda_kernel` and ctypes lanes with full three-axis grid/block
geometry. CPU invokes the packed entry owned by its process-local
`ExecutionEngine`. Lowering selection remains explicit; a generic contract
does not imply generic semantic admission or backend fallback.

## Rejected alternatives

### One runtime method per kernel ABI

This keeps compiler and runtime definitions independent. Every new physical
signature would add another positional slice and another opportunity for
silent drift.

### Infer the ABI from Python parameter names

Names are diagnostic labels, not durable type or ownership information. They
cannot describe compiler-derived counts, prepared plans, scratch, or generated
entry points.

### Parse executable text or symbols at runtime

PTX and lowered LLVM are backend artifacts, not Swage's ownership boundary.
Inspection could recover some physical kinds but not semantic origins, access,
binding keys, or the launch model.

### Pass one universal argument blob

A blob would erase typed entry signatures, move layout authority back to host
code, and weaken validation and backend diagnostics. CUDA's internal
`void **kernelParams` and LLVM's packed host-call array remain implementation
details populated from ordered, type-correct storage.

### Add a Python launch-program object IR

MLIR remains the only production IR between Python and LLVM. The contract is
compiler metadata for one lowered entry, not another representation of
program meaning.

## Consequences

- The compiler is the sole authority for physical CUDA and Native CPU entry
  signatures.
- User semantic arguments stay separate from derived, plan, and scratch
  values.
- New admitted entries extend compiler contract construction and binding maps
  instead of adding positional runtime launch methods.
- Contract changes are versioned cache changes and fail closed rather than
  receiving compatibility guesses.
- Explicit lowering, concrete physical parameter types, CUDA current-stream
  ownership, synchronous Native host invocation, and no-fallback behavior
  remain distinct backend contracts.
- Public segmented syntax, additional policies, dtypes, and operations remain
  separate capability decisions with independent gates.
