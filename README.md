# Swage

[![ci-python](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml)
[![ci-cpp](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml)
[![GPU runtime](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml)

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is an experimental Python-embedded GPU compiler built on MLIR and
LLVM. You write a Triton-like kernel that describes what happens to *one
logical segment*; Swage decides how that segment becomes fixed-size GPU
tile tasks and generates NVIDIA GPU code through MLIR, LLVM, and NVPTX.

> **Status: pre-alpha foundation.** The `swage` MLIR dialect, pinned
> LLVM/MLIR build, `swage-opt`, native bindings, and the fixed-block
> vector-add AST-emission and deterministic NVPTX code-generation slices work
> today. **No Python kernel executes** and no launch or runtime result exists.
> The PTX compiler is internal while M3 launch work remains in progress. See
> [Current status](#current-status) for the exact line.

## The programming model

The fixed-block vector-add subset is deliberately Triton-like and can infer
its narrow MLIR signature from supported PyTorch tensor metadata and Python
integers (`n` maps to `sl.int32`):

```python
import swage as sw
import swage.language as sl
import torch


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)


length = 1024
x = torch.empty(length, dtype=torch.float32)
y = torch.empty(length, dtype=torch.float32)
output = torch.empty(length, dtype=torch.float32)
module = add_kernel.emit_mlir(
    arguments={
        "x_ptr": x,
        "y_ptr": y,
        "output_ptr": output,
        "n": length,
    },
    constexprs={"BLOCK": 128},
)
assert module.operation.verify()
```

Install the optional dependency with `pip install "swage-compiler[pytorch]"`.
The explicit `signature=` form remains available without PyTorch.
`emit_mlir` requires the build-tree-only `mlir_swage` bindings. It produces
a live `mlir_swage.ir.Module`; it reads metadata only and does not execute or
retain the supplied arguments. Calling a decorated kernel, direct symbolic
language operations, launch, and runtime execution remain unavailable. Fixed
vector-add lowering and PTX emission are internal runtime building blocks,
not public APIs.

The research target is the segment API: one segment-local program, from
which the compiler derives packing, bucketing, partitioning, partial
reductions, and static or persistent scheduling as the runtime
segment-length distribution changes:

```python
@sw.jit
def segmented_softmax(values_ptr, offsets_ptr, output_ptr):
    sid = sl.segment_id(0)
    segment = sl.segment(values_ptr, offsets_ptr, sid)

    x = sl.load_segment(segment)
    maximum = sl.max(x)
    numerator = sl.exp(x - maximum)
    denominator = sl.sum(numerator)

    sl.store_segment(output_ptr, segment, numerator / denominator)
```

## Architecture

```text
Python @sw.jit kernel
        │  restricted Python AST
        ▼
Swage semantic MLIR          (segments: what happens to one segment)
        ▼
SwagePlan task IR            (tasks: how segments become schedulable work)
        ▼
Fixed-size tile operations   (tiles: warp/CTA-shaped physical units)
        ▼
gpu / nvvm / LLVM IR
        ▼
LLVM NVPTX  →  PTX  →  CUDA Driver API (M3 pending)  →  current PyTorch stream
```

The three-level vocabulary is load-bearing: a **segment** is a logical,
runtime-sized, internally dense data object; a **task** is a schedulable
unit of execution; a **tile** is a fixed-size physical unit processed by a
warp or CTA. See
[docs/concepts/segments-tiles-tasks.md](docs/concepts/segments-tiles-tasks.md).

## Current status

| Stage | State |
|---|---|
| `swage` MLIR dialect (`!swage.segment<T>`, `segment_id`, `make_segment`, `extent`) | **Works today** — parses, prints, verifies; lit-tested |
| Pinned out-of-tree LLVM/MLIR build (`llvmorg-22.1.8`) + `swage-opt` | **Works today** |
| `python -m swage.env` environment diagnostics | **Works today** |
| Native `mlir_swage` bindings package | **Works today** from the build tree; integration-tested |
| Python AST → verified live `mlir_swage.ir.Module` (fixed-block vector add, inferred or explicit signature) | **Works today, compile only** |
| Fixed vector add lowering through LLVM NVPTX to deterministic PTX | **Works today, internal**; native-tested for exact targets |
| CUDA Driver launch, cache, and real GPU result | **In progress**; no launch API or runtime result yet |
| Segment lowering (segmented sum/max, ragged softmax) | Planned |
| SwagePlan task dialect; warp/CTA/split-CTA/persistent policies | Research target |

The research question: *can one segment-local program automatically produce
competitive warp, CTA, split-CTA, and persistent schedules as the runtime
segment-length distribution changes?* Swage did not invent ragged tensors,
segmented reductions, persistent kernels, or tile programming — the intended
contribution is the automatic derivation of the schedule from one
segment-local kernel.

## Getting started

```bash
# CPU-only: build the pinned LLVM/MLIR (once, ~1 hour), then Swage
./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
./scripts/build_swage.sh        # builds swage-opt and runs the lit suite

# Python package and tests
make setup
make test
```

The native bindings require the pinned LLVM/MLIR install with its Python
bindings enabled. Run `ninja -C build check-swage-python` after the native
build; it sets the build-tree `PYTHONPATH` for `mlir_swage`. This M2 frontend
slice remains compile-only: no Python kernel executes and no GPU result is
produced. M3's internal codegen can lower that fixed vector-add module to
deterministic PTX; launch is still pending.

See [docs/quickstart.md](docs/quickstart.md). A GPU is not required to
build, test, or contribute.

## More

- [DESIGN.md](DESIGN.md) — architecture and design invariants
- [ROADMAP.md](ROADMAP.md) — phased plan and honest phase status
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor paths, CPU-only onboarding
- [docs/adr/](docs/adr/) — architecture decision records

## License

MIT — see [LICENSE](LICENSE).
