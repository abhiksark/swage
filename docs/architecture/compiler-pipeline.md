# Compiler pipeline

The production pipeline, end to end. Stages marked ✅ exist today; ⏳ are
in progress; 🔬 are research targets. There is exactly one production IR
between Python and LLVM: MLIR (ADR-0001).

```text
Python @sw.jit kernel
        ↓   inspect.getsource → textwrap.dedent → ast.parse     ⏳
Restricted Python AST
        ↓   AST visitor building ops via MLIR Python bindings   ⏳
Swage semantic MLIR (dialect: swage)                            ✅ (initial ops)
        ↓   canonicalization and fusion                          ⏳
SwagePlan task IR (dialect: swage_plan)                         🔬 (ADR-0003)
        ↓   task decomposition, policy selection
Fixed-size tile operations
        ↓
arith + math + scf + memref + vector                            (standard MLIR)
        ↓
gpu + nvgpu
        ↓
nvvm + llvm dialect
        ↓
LLVM IR → LLVM NVPTX → PTX                                      (ADR-0006)
        ↓
CUDA Driver API launch on the current PyTorch stream
```

## Frontend (planned, M2)

The `@sw.jit` decorator captures the function source, parses a
deliberately restricted Python subset, and drives an AST visitor that
constructs Swage MLIR directly. Requirements that shape it:

- Python source locations preserved into MLIR; diagnostics carry function
  name and line number.
- `constexpr` arguments separated from runtime arguments; pointer element
  types inferred from PyTorch tensors.
- Arbitrary Python calls and unsupported control flow rejected with clear
  errors; kernel bodies are never executed as normal Python.
- No Torch FX; no MLIR text round-tripping on the JIT path (textual MLIR
  is for debugging, tests, and reproducers).

## Semantic level (partially exists, M0–M1)

The `swage` dialect models segment-local computation. Today:
`!swage.segment<T>`, `swage.segment_id`, `swage.make_segment`,
`swage.extent`. Next (M1): region-based `swage.map`, `swage.reduce`,
`swage.map_store`, `swage.yield`, with ordinary arithmetic expressed by
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
