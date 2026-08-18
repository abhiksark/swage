# Swage

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is an experimental Python-embedded GPU compiler built on MLIR and
LLVM. You write a Triton-like kernel that describes what happens to *one
logical segment*; Swage decides how that segment becomes fixed-size GPU
tile tasks and generates NVIDIA GPU code through MLIR, LLVM, and NVPTX.

> **Status: pre-alpha foundation.** The `swage` MLIR dialect, pinned
> LLVM/MLIR build, `swage-opt`, and the test infrastructure work today.
> **No Python kernel compiles or executes yet** — the Python examples below
> are the design target, not working code. See
> [Current status](#current-status) for the exact line.

## The programming model (planned API)

The fixed-block starting point is deliberately Triton-like:

```python
import swage as sw
import swage.language as sl


@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)
```

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
LLVM NVPTX  →  PTX  →  CUDA Driver API  →  current PyTorch CUDA stream
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
| Python AST → Swage MLIR frontend (fixed-block subset) | In progress |
| Fixed vector add through LLVM NVPTX to a real GPU result | In progress |
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
build; it sets the build-tree `PYTHONPATH` for `mlir_swage`. The Python
frontend remains deferred to Issue #4, so no Python kernel compiles or runs.

See [docs/quickstart.md](docs/quickstart.md). A GPU is not required to
build, test, or contribute.

## More

- [DESIGN.md](DESIGN.md) — architecture and design invariants
- [ROADMAP.md](ROADMAP.md) — phased plan and honest phase status
- [CONTRIBUTING.md](CONTRIBUTING.md) — contributor paths, CPU-only onboarding
- [docs/adr/](docs/adr/) — architecture decision records

## License

MIT — see [LICENSE](LICENSE).
