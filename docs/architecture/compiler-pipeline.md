# Compiler pipeline

The production pipeline, end to end. Stages marked ✅ exist today; ⏳ are
in progress; 🔬 are research targets. There is exactly one production IR
between Python and LLVM: MLIR (ADR-0001).

```text
Python @sw.jit kernel
        ↓   inspect.getsource → textwrap.dedent → ast.parse     ✅ (M2 subset)
Restricted Python AST
        ↓   AST visitor building ops via MLIR Python bindings   ✅
Swage semantic MLIR (dialect: swage)                            ✅ (initial ops)
        ↓   verified module boundary                             ✅
Live mlir_swage.ir.Module                                        (M2 fixed path)
        ├── M3 fixed vector-add path
        │     ↓   validate canonical kernel                      ✅
        │   gpu + scf + llvm dialect                             ✅
        │     ↓   GPU to NVVM/LLVM lowering                      ✅
        │   LLVM IR → LLVM NVPTX → PTX                           ✅
        │     ↓
        │   validated cache + CUDA Driver launch                  ✅
        │     ↓
        │   current PyTorch CUDA stream                           ✅
        │
        ├── M4 canonical segmented sum/max path                  ✅
        │     ├── sequential scf + memref → mlir-runner          ✅
        │     └── one CTA per segment → gpu + nvvm → PTX         ✅
        │         → host-validated internal CUDA qualification    ✅
        │
        ├── M5 canonical ragged-softmax path                     ✅
        │     ├── max → exponential sum → normalize/store         ✅
        │     ├── sequential scf + memref → mlir-runner           ✅
        │     └── one CTA per segment → native exp2 → PTX         ✅
        │         → internal PyTorch and CPU-oracle qualification ✅
        │
        ├── M6 canonical identity segmented-sum planning path
        │     ↓   read-only admission                             ✅
        │   private companion with swage_plan.classify           ✅ (ADR-0014)
        └── M7 private materialization → stable warp/CTA task IDs ✅
                                       → pure task-ID kernels      ✅
                                       → one fused mixed kernel    ✅ (ADR-0016)
```

## Frontend (M2 complete)

The shipped slice captures source with `@sw.jit`, parses a deliberately
restricted Python subset, and emits a live `mlir_swage.ir.Module` directly
through MLIR Python bindings. It accepts the README fixed-block vector-add
form with either explicit pointer and scalar types in `signature` or an
`arguments` mapping containing supported PyTorch tensors and Python integers.
Compile-time values remain in `constexprs`. It preserves Python source
locations and returns verifiable deterministic MLIR without textual
round-tripping.

The `emit_mlir()` boundary is compile-only. PyTorch inference reads layout,
dtype, rank, device type, and contiguity, then discards the arguments. It does
not read data pointers or contents, execute kernel calls or direct symbolic
language operations, launch a GPU kernel, or return a runtime result. The M3
runtime uses a separate explicit `launch()` boundary without changing
`emit_mlir()`.
The following constraints remain in force:

- `constexpr` arguments remain separate from runtime arguments.
- Arbitrary Python calls and unsupported control flow remain rejected with
  source-located errors.
- No Torch FX or MLIR text round-tripping appears on the emission path.

## Fixed vector-add execution (M3 complete)

`kernel.launch(arguments=..., constexprs=..., grid=...)` validates the exact
M3 subset before marshalling three f32 device pointers and one i32 length.
The supported kernel uses axis-zero `program_id`, one vector whose width
equals `BLOCK`, masked f32 loads and stores, and addition. Every vector lane
becomes one x-thread. Unsupported axes, shapes, operations, ABI types, blocks,
or grids fail closed.

The internal compiler clones the semantic module, lowers it through upstream
GPU, NVVM, and LLVM infrastructure, attaches the active device's exact `sm_*`
target, and emits PTX in-process through LLVM NVPTX. There is no NVRTC,
subprocess compiler, textual round-trip, or CUDA toolkit dependency.

The runtime verifies disk-cache metadata and content digests before loading
PTX through `libcuda`. Loaded modules are reused per CUDA context. Launch is
asynchronous on `torch.cuda.current_stream()` and calls `record_stream()` for
each tensor after submission. It never copies, casts, changes devices,
creates a CUDA context, synchronizes, or falls back. The trusted GPU workflow
qualifies correctness, non-default streams, argument lifetime, and cache
reuse on `main`.

## Segmented reduction qualification (M4 complete)

The native M4 path accepts one fail-closed semantic shape: axis-zero
`segment_id`, `make_segment` over rank-one f32 values and i32 offsets, and a
capture-free identity transform reduced with `kind<sum>` or `kind<max>`. The
ABI adds rank-one f32 output plus explicit i32 value and segment counts.
Unsupported axes, types, captures, region operations, reduction kinds, and
stores fail before lowering.

The CPU conversion replaces the logical segment grid with a sequential
`scf.for` and executes the fully lowered module with upstream `mlir-runner`.
The GPU conversion maps one segment to each x-block. Threads traverse the
segment with block-stride loads and use `gpu.all_reduce` before thread zero
stores one result. Empty sums return zero; empty maxima return negative
infinity. Max uses `maximumf` so a NaN input produces a NaN result.

An internal qualification runner copies offset metadata to the host for
validation, then launches raw CUDA pointers and the two explicit counts on
the current PyTorch stream. This runner is not public API and does not change
the M3 `kernel.launch()` contract. General CUDA-resident offset validation,
public segment transforms, and schedule selection remain outside M4.

## Ragged-softmax qualification (M5 complete)

M5 keeps the M4 five-argument internal ABI (ADR-0013) and admits ordered f32
reduction captures plus single-consumer `swage.map` chains. Maps are fused
into their sole map, reduce, or map-store consumer rather than materialized.
Admission is fail-closed: unsupported region operations, captures, consumers,
types, or multiple output terminals fail before either backend lowers the
module.

The canonical stable softmax performs three stages per segment:

1. Reduce the maximum.
2. Reduce the sum of `exp2((value - maximum) * log2(e))`.
3. Recompute that exponential, divide by the captured sum, and store each
   normalized value.

The scalar `memref.store` terminal used by reductions writes one value per
segment. The `swage.map_store` terminal writes one value per covered input
element and therefore requires output capacity of at least `offsets[-1]`.
Output must not overlap values or offsets. Empty segments perform no stores.

The CPU backend executes reductions in source order and then the store loop.
The GPU backend maps one segment to one CTA and makes one block-stride pass per
stage. Uniform `gpu.all_reduce` broadcasts each result and fences the next
phase, so the emitter adds no separate barrier or scratch buffer. The terminal
recomputes exponentials instead of retaining an intermediate allocation. GPU
code generation replaces the unavailable libdevice call with LLVM's native
`exp2` intrinsic; exact-target NVPTX compilation requires `sm_80` or newer.

The internal runner validates metadata on the host. Its CPU result matches
PyTorch, and its RTX A6000 `sm_86` result matches both PyTorch and the CPU
oracle across all-empty, all-singleton, many-tiny, few-huge, one-outlier, and
alternating-empty distributions. This does not add public segment primitives,
public segmented launch, schedule selection, or multi-CTA execution.

## Minimal planning gate (M6 complete)

The `--swage-to-plan` pass accepts only the canonical capture-free,
map-free, single-stage identity segmented sum. Admission is read-only. On
success, the pass preserves the semantic function and adds one private
planning companion containing `swage_plan.classify`, the semantic kernel
reference, the configured threshold, warp then CTA as the legal policy order,
and one `!swage_plan.task_range`. Unsupported modules fail before mutation.

The internal host classifier separately validates i32 offsets, counts, and
the threshold before emitting one stable descriptor per segment, including
empty segments. Each descriptor carries the segment ID, absolute half-open
range, stage zero, selected warp or CTA policy, and a dependency group equal
to the segment ID. M7 connects this generator to the planning operation only
through the private qualification path described next.

## Minimal mixed-policy execution (M7 complete)

The private native materialization helper clones the semantic module, runs
`--swage-to-plan`, requires exactly one `swage_plan.classify`, reads its
recorded threshold, and invokes the host classifier with validated i32
metadata. The source module remains unchanged. The Python preparation path
then materializes stable warp IDs followed by stable CTA IDs. Tensor, alias,
offset, and device validation completes before compilation, allocation, or
launch, and failures never select a fallback policy or backend.

Pure warp and pure CTA qualification use the same four-pointer, two-count
task-ID ABI with 32-thread and 128-thread blocks, respectively. Mixed
qualification uses one four-pointer, three-count kernel launch. Each initial
128-thread block contains four one-segment warp slots; the remaining blocks
each execute one CTA task. Empty total work enqueues no kernel. Successful
submission uses the current PyTorch stream and records values, offsets,
output, and task IDs on that stream.

The internal correctness suite covers empty, boundary, repeated-empty,
skewed, and large segment cases on NVIDIA RTX A6000 `sm_86`. The frozen
bimodal benchmark recorded medians of `0.067584 ms` for pure warp,
`0.070656 ms` for pure CTA, and `0.063488 ms` for mixed execution. The
mixed-to-best-pure ratio is `0.939394`, which passes the predeclared `1.05`
maximum.

This remains a canonical identity-sum qualification path, not public
segmented execution. Packing, splitting, partial reductions, merges, queues,
persistent scheduling, ragged-softmax planning, and general task-range
lowering remain deferred.

## Semantic level (M0–M1 complete)

The `swage` dialect models segment-local computation. Today:
`!swage.segment<T>`, `swage.segment_id`, `swage.make_segment`,
`swage.extent`, and the region-based `swage.map`, `swage.reduce`,
`swage.map_store`, and `swage.yield` operations. Ordinary arithmetic uses
`arith`/`math` ops *inside* regions; a dynamic-length segment is never
materialized as a runtime-sized SSA vector; reductions and maps stay
symbolic until tiling.

## Planning level (M6–M7 minimal boundary complete)

`swage_plan` currently represents only warp and CTA policy attributes, one
opaque task-range type, and one classification operation. Compile time records
the legal policy order and threshold for the admitted semantic kernel. The M7
private materialization path uses actual lengths and counts to produce stable
task IDs and dispatch the fixed pure or fused mixed schedules.

This does not define a general lowering for arbitrary task ranges. Later
milestones own packing, splitting, partial reductions, merges, queues,
dependencies, persistent scheduling, and broader policy selection. Passes
never pretend to know offset contents.

## Backend

Standard MLIR lowering: tiles into `vector`/`scf`/`memref`, GPU structure
via `gpu`/`nvgpu`, NVIDIA intrinsics via `nvvm`, then LLVM IR and the
NVPTX backend emit PTX. The runtime loads PTX with the CUDA Driver API
and launches on `torch.cuda.current_stream()`, sharing PyTorch's CUDA
context, never copying tensors, changing dtypes, or falling back silently.

## What Swage deliberately does not own

Generic arithmetic, loops, buffers, and LLVM IR belong to upstream
dialects and LLVM. Swage owns segment semantics, fusion, cost inference,
task decomposition, schedule selection, partial-reduction and merge
construction, and queue generation, and nothing else.
