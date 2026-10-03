<!-- DESIGN.md -->

# Swage design

This document records stable architecture and invariants. Current interfaces
live under `docs/reference/`, implemented data flow lives in the compiler
pipeline page, milestone gates live in `ROADMAP.md`, and alternatives live in
the ADRs.

## Problem

Variable-sized, internally dense segments appear in ragged softmax rows,
jagged batches, and graph neighborhoods. Fixed GPU work shapes handle regular
data well, but padding and one-shape scheduling can waste work or create load
imbalance. Swage studies whether one segment-local program can retain its
meaning while task derivation changes with runtime length distributions.

## Vocabulary

Three levels remain distinct:

- A **segment** is a logical runtime-sized dense slice described by values,
  offsets, and a segment index.
- A **task** is a schedulable unit derived for one or more stages of segment
  work.
- A **tile** is a fixed physical warp or CTA step used to execute a task.

Some ADRs use `tile<...>` as conceptual notation. There is no current Swage
tile type. Current qualified warp and CTA paths use 32-thread and 128-thread
steps respectively.

The logical grid identifies semantic program instances. The physical grid
contains launched GPU work. See
[`docs/user-guide/execution-model.md`](docs/user-guide/execution-model.md)
for the current and planned boundary.

## Compiler architecture

```text
Python source or native test IR
        |
        v
verified Swage semantic MLIR
        |
        +-- public canonical fixed vector add or multiply
        |      +-- Native host lowering -> LLVM ExecutionEngine
        |      `-- GPU lowering -> LLVM NVPTX -> PTX -> CUDA Driver API
        +-- private direct segmented CPU/GPU qualification
        `-- private identity-sum planning and GPU split execution
```

The backend-neutral boundary is the verified semantic module plus a
compiler-generated physical launch contract. Backend choice selects one
explicit admitted lowering; it does not introduce fallback or another IR.

MLIR is the only production IR between Python and LLVM. The Python frontend
constructs native operations directly through the pinned MLIR bindings.
Textual MLIR is for tests, debugging, and reproducers, not the JIT
construction path.

Each admitted physical lowering constructs a versioned launch contract from
the same ordered specification as its concrete function signature. The
contract records backend, launch model, physical types, semantic or runtime
ownership, access, and entry symbol. CUDA contracts additionally record block
geometry. The contract is compiler metadata rather than another program IR;
it is validated and removed before downstream lowering.

The current fixed-block frontend and public execution subset are deliberately
narrow. Native segmented modules exercise a separate private qualification
surface. The canonical pipeline and links to exact references live in
[`docs/internals/compiler-pipeline.md`](docs/internals/compiler-pipeline.md).

## Semantic invariants

- A runtime-length segment is symbolic and is never represented as a
  runtime-sized register array.
- A segment type carries element type only. Values, offsets, and runtime
  identity remain SSA operands.
- GPU thread and block IDs do not appear in semantic Swage IR.
- Ordinary scalar arithmetic uses upstream `arith` and `math` operations.
- Region captures are explicit and ordered.
- Cross-segment effects must be explicit. A map-store writes only the
  corresponding segment range.
- Unsupported syntax, module shapes, types, policies, and ABIs fail before
  mutation or launch.

## Planning invariants

`swage_plan` is a distinct private dialect because scheduling and semantic
meaning have different invariants. Its current surface records only warp and
CTA policies, one opaque task-range result, and one classification operation
for an admitted identity segmented sum.

Compiler passes do not inspect runtime offset contents. Host classification
validates that metadata before producing stable direct or split records.
Split-CTA execution is task decomposition under the CTA policy, not a new
policy.
The fused mixed executor privately packs four warp task records into each
128-thread block. This is a qualified physical execution strategy, not a
public or general planner policy.

One private experimental identity-sum path now consumes the existing host
classification through device claim counters and publishes split completion
before a unique merge. Its clean A6000 run failed the predeclared performance
gate, so the path remains experimental.
General cost inference, public/general packed-warp planner policy, reusable
queues, and public segmented execution remain planned. They are not current
Swage ownership claims.

## Runtime invariants

- PyTorch owns tensor storage. On CUDA it also owns the active device,
  context, and current stream.
- Swage reads raw pointers only after validation and dispatches to exactly one
  selected backend.
- CUDA PTX is emitted in process through LLVM NVPTX and launched through the
  CUDA Driver API. NVRTC is not a production dependency.
- CUDA launch is asynchronous and records submitted tensors on the stream.
  Native CPU launch invokes a process-local LLVM JIT entry synchronously.
- No path silently copies, casts, synchronizes CUDA work, changes devices,
  creates a CUDA context, or falls back to another backend or policy.
- Compiler-produced launch contracts are validated against concrete physical
  functions before executable loading or invocation. CUDA nanobind and ctypes
  lanes marshal the same ordered typed values.
- Process artifact state is bounded for both backends. CUDA loaded-entry
  lookup state is separately bounded; lookup eviction does not imply module
  unload because leases, per-stream completion events, current context, and
  graph-capture lifetime are separate gates.
- CUDA deferred unload only polls in the matching current context and never
  synchronizes or changes contexts. Graph-captured modules remain pinned until
  context destruction without an explicit graph lifetime hook.

Runtime and cache requirements live only in
[`docs/reference/runtime-environment.md`](docs/reference/runtime-environment.md).

## Package boundary

The PyPI distribution is `swage-compiler`; its public import package is
`swage`. The v0.5.2 native-wheel contract bundles self-contained private
`mlir_swage`, fixed-contract stubs, native runtime libraries, licenses, and
validated build provenance. Private segmented Python modules remain in source
distributions and checkouts but are excluded from wheels.

`mlir_swage` embeds the exact pinned MLIR Python core and generated Swage
bindings. It never layers onto an unrelated external `mlir` package. Normal
source builds retain `build/python_packages/mlir_swage`; wheel installs use
the site-packages root. Asking CMake to enable bindings against an MLIR install
without Python bindings, or against a release other than the exact pin, is an
error. Native packaging does not expand the admitted public kernel subset.

## Verification strategy

- Python tests cover source capture, diagnostics, package boundaries, launch
  validation, specialization, cache integrity, backend selection, and CUDA
  Driver marshalling.
- Lit and FileCheck cover dialect parsing, verification, and admitted CPU/GPU
  lowering shapes.
- Native integration tests construct live MLIR through the build-tree package;
  installed-wheel gates also exercise the public CPU path on every wheel ABI.
- C++ tests cover host task classification, descriptor invariants, and
  physical launch contracts.
- Sequential CPU lowering and PyTorch serve as correctness oracles for
  private segmented qualification.
- The trusted GPU workflow covers public fixed vector add or multiply plus
  private segmented runtime qualification on a real NVIDIA device.
- Frozen performance evidence separates preparation from timed launches and
  is not retuned after a failed gate.
- Release checks inspect native wheel tags, ELF dependencies and relocatability,
  immutable build identity, licenses, size, and source/sdist byte reproducibility.
  Installed-artifact A6000 SLO gates are separate from private research gates.

The claim-to-test mapping lives in
[`docs/internals/verification.md`](docs/internals/verification.md).

## Dependency policy

Swage uses one exact LLVM release from `cmake/llvm-version.txt`, built
out-of-tree through `MLIR_DIR` and `LLVM_DIR`. LLVM pin changes require a
dedicated compatibility change. LLVM is not vendored, Triton is not a
dependency, and the documentation and helper tooling do not introduce a
second compiler stack.

For rationale, continue with the
[`docs/decisions/` index](docs/decisions/index.md).
