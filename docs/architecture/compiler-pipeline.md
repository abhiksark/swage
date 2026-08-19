# Compiler pipeline

The production pipeline, end to end. Stages marked ✅ exist today; ⏳ are
in progress; 🔬 are research targets. There is exactly one production IR
between Python and LLVM: MLIR (ADR-0001).

```text
Python @sw.jit kernel
        ↓   inspect.getsource → textwrap.dedent → ast.parse     ✅ (M2 subset)
Restricted Python AST
        ↓   AST visitor building ops via MLIR Python bindings   ✅ (compile only)
Swage semantic MLIR (dialect: swage)                            ✅ (initial ops)
        ↓   verified module boundary                             ✅ (fixed vector add)
Live mlir_swage.ir.Module                                        (M2 endpoint)
        ├── M3 fixed vector-add path
        │     ↓   validate canonical kernel                      ✅
        │   gpu + scf + llvm dialect                             ✅
        │     ↓   GPU to NVVM/LLVM lowering                      ✅
        │   LLVM IR → LLVM NVPTX → PTX                           ✅
        │     ↓
        │   CUDA Driver launch on current PyTorch stream          ⏳
        │
        └── General segment path
              ↓   canonicalization and fusion                    ⏳
            SwagePlan task IR (dialect: swage_plan)              🔬 (ADR-0003)
              ↓   task decomposition, policy selection
            Fixed-size tile operations
              ↓
            arith + math + scf + memref + vector                 (standard MLIR)
              ↓
            gpu + nvgpu → nvvm + llvm → LLVM NVPTX → PTX         (ADR-0006)
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

The boundary is compile only. PyTorch inference reads layout, dtype, rank,
device type, and contiguity, then discards the arguments. It does not read
data pointers or contents, execute kernel calls or direct symbolic language
operations, launch a GPU kernel, or return a runtime result. An internal
native entry point now lowers the returned canonical module to PTX without
changing `emit_mlir()`.
The following constraints remain in force:

- `constexpr` arguments remain separate from runtime arguments.
- Arbitrary Python calls and unsupported control flow remain rejected with
  source-located errors.
- No Torch FX or MLIR text round-tripping appears on the emission path.

## Semantic level (M0–M1 complete)

The `swage` dialect models segment-local computation. Today:
`!swage.segment<T>`, `swage.segment_id`, `swage.make_segment`,
`swage.extent`, and the region-based `swage.map`, `swage.reduce`,
`swage.map_store`, and `swage.yield` operations. Ordinary arithmetic uses
`arith`/`math` ops *inside* regions — a dynamic-length segment is never
materialized as a runtime-sized SSA vector; reductions and maps stay
symbolic until tiling.

## Planning level (research, M6+)

`swage_plan` will represent tasks, policies (`warp`, `packed_warp`, `cta`,
`split_cta`, `merge`, `persistent`), queues, and dependencies. Compile
time decides *legal schedule variants, tile sizes, reduction structure,
cost estimates*; runtime decides *actual lengths, counts, skew, task
counts, selected policies*. Passes never pretend to know offset contents;
where runtime decisions are needed, the compiler emits planner code.

## Backend

Standard MLIR lowering: tiles into `vector`/`scf`/`memref`, GPU structure
via `gpu`/`nvgpu`, NVIDIA intrinsics via `nvvm`, then LLVM IR and the
NVPTX backend emit PTX. The runtime loads PTX with the CUDA Driver API
and launches on `torch.cuda.current_stream()` — sharing PyTorch's CUDA
context, never copying tensors, changing dtypes, or falling back silently.

## What Swage deliberately does not own

Generic arithmetic, loops, buffers, and LLVM IR belong to upstream
dialects and LLVM. Swage owns segment semantics, fusion, cost inference,
task decomposition, schedule selection, partial-reduction and merge
construction, and queue generation — nothing else.
