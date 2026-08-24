# Swage design

This document records the architecture and the invariants that hold across
the codebase. Deep dives live in `docs/`; decisions with alternatives live
in `docs/adr/`.

## Problem

Fixed GPU tiles work extremely well for regular workloads. Variable-sized,
internally dense segments (ragged softmax rows, jagged batches, graph
neighborhoods) create padding waste and load imbalance. Existing systems
expose pieces of the solution, including manual bucketing, hand-written
persistent kernels, and per-shape specialization. Swage's research question
is whether one segment-local program can automatically produce competitive
warp, CTA, split-CTA, and persistent schedules as the runtime segment-length
distribution changes.

## Vocabulary

Three terms, never conflated (full treatment in
[docs/concepts/segments-tiles-tasks.md](docs/concepts/segments-tiles-tasks.md)):

- **Segment**: a logical, runtime-sized, internally dense data object:
  `segment i = values[offsets[i] : offsets[i+1]]`. Part of the semantic
  model.
- **Task**: a schedulable unit of execution: one short segment, several
  packed short segments, a chunk of a long segment, a partial reduction, a
  merge, a store stage. Part of the planning model.
- **Tile**: a fixed-size physical unit (`tile<32xf32>`, `tile<128xf32>`)
  processed by a warp or CTA. Part of the hardware-lowering model.

The **logical grid** is the semantic domain written in Python. The
**physical grid** is the generated GPU task domain. The **planner** converts
segment programs plus runtime extents into tasks and fixed tiles.

## Pipeline

```text
Python @sw.jit kernel
        ↓  inspect.getsource → ast.parse (restricted subset, M2 emission)
Swage semantic MLIR            (dialect: swage)
        ├── M3 fixed vector add
        │     ↓  canonical validation and lane-to-thread conversion
        │   gpu + scf + nvvm + llvm
        │     ↓
        │   LLVM IR → LLVM NVPTX → PTX → CUDA Driver API
        │     ↓
        │   current PyTorch stream
        │
        ├── M4 canonical segmented sum/max
        │     ├── sequential scf + memref → upstream mlir-runner
        │     └── one CTA per segment → gpu + nvvm + llvm → PTX
        │
        ├── M5 canonical ragged softmax
        │     ├── maximum → exponential sum → normalize/store
        │     ├── sequential scf + memref → upstream mlir-runner
        │     └── one CTA per segment → gpu + nvvm + llvm → PTX
        │
        ├── M6 identity segmented sum
        │     ↓  read-only admission and private planning companion
        │   SwagePlan classify (dialect: swage_plan)
        └── M7 private materialization → stable warp/CTA task IDs
                                       → pure task-ID kernels
                                       → one fused mixed GPU kernel
```

There is exactly one production IR between Python and LLVM: MLIR. The
Python frontend constructs Swage MLIR directly through the MLIR Python
bindings (ADR-0001). Textual MLIR is a debug, test, and reproducer format,
not the JIT construction path.

## Current Python, M3 execution, and M4–M7 qualification boundary

M2 is complete. The fixed-block vector-add subset captures a `@sw.jit`
function and emits a verified live `mlir_swage.ir.Module` directly through
the native bindings. Callers must provide exactly one of `signature=` or
`arguments=`; providing both or neither is invalid. From `arguments=`, the
frontend infers rank-one f32 pointers and i32 scalars. PyTorch stays optional
for explicit descriptors and is imported lazily for inference or launch.

Inference reads metadata only. `emit_mlir()` does not inspect data pointers or
contents, retain arguments, infer lengths, validate cross-tensor
relationships, or cache specializations. Direct kernel calls and direct
symbolic language operations do not execute.

M3 adds an explicit keyword-only `launch()` boundary for the canonical fixed
vector add. It accepts three contiguous rank-one f32 CUDA tensors, one
nonnegative i32 `n`, a device-legal `BLOCK`, axis-zero `program_id`, and the
exact one-dimensional grid. Validation completes before data pointers are
marshalled. The backend lowers through standard GPU, NVVM, and LLVM paths,
emits PTX for the active device's exact compute capability, and launches on
the current PyTorch stream (ADR-0011, ADR-0012). Unsupported kernels and ABI
shapes fail closed.

M4 adds native compiler qualification for one canonical f32 segmented
reduction shape. It accepts rank-one values, i32 offsets, rank-one output,
and explicit i32 value and segment counts. `segment_id(0)` becomes the
sequential CPU loop induction variable or the GPU x-block ID. GPU threads
perform block-stride loads and an upstream `gpu.all_reduce`; only thread zero
stores the result. Sum uses zero and max uses negative infinity for empty
segments. Floating max uses NaN-propagating `maximumf` semantics.

The qualification runner validates offsets on the host before launch and
uses the existing CUDA Driver wrapper and exact active `sm_*` target. This is
not a new public runtime contract: `swage.language` gains no segment
primitives, `emit_mlir()` remains compile-only, and public `launch()` remains
the M3 fixed vector-add boundary.

M5 extends only this internal qualification path (ADR-0013). It admits
ordered f32 reduction captures and single-consumer maps, then fuses each map
into its consumer. Stable ragged softmax runs a maximum pass, a sum of shifted
exponentials, and a normalization/store pass. The terminal
`swage.map_store` writes one output per covered input element, whereas the M4
scalar `memref.store` terminal writes one output per segment; both retain the
same five-argument internal ABI. Exponentials are recomputed for the terminal
pass instead of stored in an intermediate buffer.

The CPU path executes those phases sequentially. The GPU path uses one CTA
per segment, with uniform all-reduces broadcasting results and synchronizing
the phases. GPU `math.exp2` becomes the native NVPTX approximation, and exact
target compilation requires `sm_80` or newer. The internal runner requires
output to be disjoint from values and offsets and large enough for
`offsets[-1]`; empty segments write nothing.

CPU output matches PyTorch, and RTX A6000 `sm_86` output matches both PyTorch
and the CPU oracle across all-empty, all-singleton, many-tiny, few-huge,
one-outlier, and alternating-empty distributions. Public segment primitives,
public segmented launch, schedule selection, multi-CTA execution, and
device-side offset validation remain future runtime and compiler work.

M6 adds only the minimal planning boundary in ADR-0014. The
`--swage-to-plan` conversion admits one capture-free, map-free, single-stage
identity segmented sum, preserves the semantic function, and adds a private
planning companion. The companion records one `swage_plan.classify`
operation, warp then CTA as its legal policy order, and a configurable
nonnegative i32 threshold that defaults to 32. Unsupported inputs fail before
module mutation.

The internal host classifier is a separately tested descriptor generator. It
validates i32 runtime metadata before producing one absolute half-open range
per segment, including empty segments. Lengths at or below the threshold use
warp; longer lengths use CTA. M6 alone does not connect these descriptors to
the planning operation, dispatch work, execute either policy, or change the
public frontend, emission, or launch contracts.

M7 adds the smallest private execution connection defined by ADR-0015 and
ADR-0016. A native helper clones the semantic module, runs the M6 planning
pass, requires exactly one `swage_plan.classify`, reads its threshold, and
invokes the host classifier. Python materializes stable warp IDs followed by
CTA IDs only after tensor and offset validation. Unsupported shapes and
failures do not fall back to another policy or backend.

Pure qualification schedules run every segment through either a 32-thread
warp kernel or a 128-thread CTA kernel with the same task-ID ABI. The mixed
schedule submits one 128-thread kernel. Its warp blocks hold four independent
one-segment warp slots, followed by one block for each CTA task. Empty total
work enqueues no kernel. Launches use the current PyTorch CUDA stream and
retain all submitted tensors through stream recording.

Exact identity sums pass the internal GPU suite on NVIDIA RTX A6000 `sm_86`.
On the frozen 32,768-segment bimodal benchmark, the mixed median is
`0.063488 ms`, the best pure median is `0.067584 ms`, and their ratio is
`0.939394`, below the predeclared `1.05` maximum. This evidence does not widen
`swage.language`, `emit_mlir()`, or the public M3 `launch()` contract.

## The `swage` dialect (semantic level)

Implemented today: `!swage.segment<T>` plus `swage.segment_id`,
`swage.make_segment`, `swage.extent`, and the region-based `swage.map`,
`swage.reduce` (kinds `sum`/`max`/`min`), `swage.map_store`, and
`swage.yield`. Regions are isolated from above: outer values enter only
through explicit `captures(...)`; scalar math inside regions uses standard
`arith` / `math` operations. `swage.map_store` is the only effectful
operation (declared write on its output). Empty-segment behavior,
reduction identities, and the aliasing obligation are specified in
[ADR-0008](docs/adr/ADR-0008-region-ops-isolation-and-kinds.md).

Design rules (enforced by verifiers as the dialect grows):

- A segment value is symbolic. A runtime-length segment is never
  represented as a runtime-sized register array; consumers are tiled during
  lowering.
- Runtime segment identity lives in SSA values and operands, never in
  types. `!swage.segment<f32>` carries the element type only.
- No GPU thread or block indices in semantic IR. `swage.segment_id` is a
  logical-grid coordinate.
- No custom Swage operations for ordinary scalar arithmetic; regions use
  `arith` and `math`.
- Cross-segment effects must be explicit; a segment program may not
  silently touch another segment's output range.

## Python bindings

Python constructs Swage IR through the self-contained `mlir_swage`
package: the pinned MLIR core Python sources, the standard-dialect
wrappers regions compose with (`builtin`, `func`, `arith`, `math`), and
the generated `swage` op bindings, all built from the one pinned LLVM
(ADR-0009). The package never imports an external `mlir` distribution,
so bindings can never skew against the pin. During development it is
imported from the build tree; wheel packaging is deferred. Bindings are
optional at configure time (`SWAGE_PYTHON_BINDINGS`), and requesting
them against an MLIR install built without bindings is a configure
error, never a silent skip.

## The `swage_plan` dialect (minimal planning and execution level)

M6 implements only `#swage_plan.policy<warp|cta>`,
`!swage_plan.task_range`, and `swage_plan.classify` (ADR-0014). Compile time
verifies the one admitted semantic shape and records its legal policy order
and threshold. M7 privately consumes that record to classify actual offsets
and materialize stable task-ID lists without changing the semantic IR. It
executes only the canonical identity segmented sum through the fixed pure and
fused schedules above.

Packed warps, split CTAs, partial reductions, merges, queues, dependencies,
ragged-softmax planning, general task-range lowering, and cost-based selection
remain future work. Compiler passes do not pretend to know runtime offset
contents.

## Runtime

The runtime reuses PyTorch's CUDA context and stream: device pointers via
`tensor.data_ptr()`, launches on `torch.cuda.current_stream()`, module
loading through the CUDA Driver API, PTX emitted by LLVM NVPTX (no NVRTC on
the production path, ADR-0006). No silent copies, dtype changes, CPU
fallback, or backend fallback. Launch is asynchronous, records each tensor on
the stream after submission, and never creates a context, changes devices, or
synchronizes.

Persistent cache keys include normalized source, kernel name, ordered ABI
descriptors, constexpr values, exact compute capability, code-generation
options, clean Swage revision, dialect version, and LLVM version. Cache files
are digest-validated and written atomically with user-only permissions.
Symlinked, world-writable, incomplete, or corrupt entries are rejected before
module loading. Dirty or unidentified builds retain only process-local reuse.
Loaded modules are cached per CUDA context and never retain tensors or data
pointers.

## Testing strategy

- Python: pytest for frontend emission, source diagnostics, launch validation,
  specialization keys, cache integrity, and CUDA Driver ABI marshalling.
- MLIR: lit + FileCheck for parse/print round trips, verifier failures,
  and lowering; checks target semantic invariants, not incidental
  formatting.
- C++: GoogleTest for the host task classifier and later planner or cost-model
  units.
- Differential: every public correctness claim is backed by at least one
  oracle comparison. With no Python prototype in this repository
  (ADR-0005), PyTorch reference implementations are the initial oracle;
  a CPU reference lowering joins it with the first segment lowerings.
- GPU: a trusted main-only self-hosted workflow checks fixed vector-add
  correctness; segmented sum, max, and ragged-softmax distributions; empty
  and NaN behavior; repeated execution; non-default streams; argument
  lifetime; cache reuse; and the internal M7 pure and fused mixed identity-sum
  schedules on a real NVIDIA device. Pull requests never execute on that
  runner.
- Property-based: generated segment distributions (empty, tiny, uniform,
  log-normal, bimodal, Zipf-like, one-outlier) checking coverage and
  no-overlap invariants.

## Dependency policy

One exact LLVM release, pinned in `cmake/llvm-version.txt`, built
out-of-tree via `MLIR_DIR`/`LLVM_DIR` (ADR-0004). No vendored LLVM sources.
Pin updates are dedicated compatibility PRs.
