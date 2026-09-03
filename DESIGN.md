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
        +-- public canonical fixed vector add
        +-- private direct segmented qualification
        +-- private identity-sum planning and split execution
        |
        v
upstream MLIR GPU, SCF, NVVM, and LLVM infrastructure
        |
        v
LLVM NVPTX -> PTX -> CUDA Driver API
```

MLIR is the only production IR between Python and LLVM. The Python frontend
constructs native operations directly through the pinned MLIR bindings.
Textual MLIR is for tests, debugging, and reproducers, not the JIT
construction path.

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

One private experimental identity-sum path now consumes the existing host
classification through device claim counters and publishes split completion
before a unique merge. Its clean A6000 run failed the predeclared performance
gate, so the path remains experimental.
General cost inference, schedule selection, packed warps, reusable queues, and
public segmented execution remain planned. They are not current Swage
ownership claims.

## Runtime invariants

- PyTorch owns tensor storage, the active CUDA device and context, and the
  current stream.
- Swage reads raw pointers only after validation and launches through the
  CUDA Driver API.
- PTX is emitted in process through LLVM NVPTX. NVRTC is not a production
  dependency.
- Launch is asynchronous. Submitted tensors are recorded on the stream.
- No path silently copies, casts, synchronizes, changes devices, creates a
  context, or falls back to another backend or policy.

Runtime and cache requirements live only in
[`docs/reference/runtime-environment.md`](docs/reference/runtime-environment.md).

## Package boundary

The PyPI distribution is `swage-compiler`; its import package is `swage`.
The wheel contains pure Python package files only. It excludes compiler build
output and the native `mlir_swage` package.

`mlir_swage` embeds the pinned MLIR Python core and generated Swage bindings
as a build-tree artifact. It never layers onto an unrelated external `mlir`
package. Native wheel packaging remains deferred. Asking CMake to enable the
bindings against an MLIR install without Python bindings is an error.

## Verification strategy

- Python tests cover source capture, diagnostics, package boundaries, launch
  validation, specialization, cache integrity, and CUDA Driver marshalling.
- Lit and FileCheck cover dialect parsing, verification, and admitted
  lowering shapes.
- Native integration tests construct live MLIR through the build-tree package.
- C++ tests cover host task classification and descriptor invariants.
- Sequential CPU lowering and PyTorch serve as correctness oracles for
  private segmented qualification.
- The trusted GPU workflow covers public fixed vector add plus private
  segmented runtime qualification on a real NVIDIA device.
- Frozen performance evidence separates preparation from timed launches and
  is not retuned after a failed gate.

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
