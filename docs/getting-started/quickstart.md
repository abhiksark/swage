<!-- docs/getting-started/quickstart.md -->

# Quickstart

This tutorial takes the canonical fixed vector-add kernel from source
capture to a verified CUDA result. Capture works on a wheel-only install,
emitting MLIR requires the native build, and the launch at the end
requires a CUDA GPU. The committed
[`examples/fixed_vector_add.py`](https://github.com/abhiksark/swage/blob/main/examples/fixed_vector_add.py)
contains the same walkthrough as one runnable script.

Python source crosses a restricted AST validation boundary before becoming
verified semantic MLIR. From that point, `emit_mlir()` stops with a
compile-only module and needs no GPU. The canonical `launch()` path
continues through native compilation to CUDA.

<div class="doc-figure" tabindex="0" markdown="1">

![Frontend validation, compile-only emission, and canonical launch branches](../assets/diagrams/frontend-boundary.svg)

</div>

*The verified frontend boundary and its two public outcomes. [Open the full-size figure](../assets/diagrams/frontend-boundary.svg).*

## Prerequisites

Follow [Installation](installation.md) to install the Python package and
build the pinned LLVM/MLIR toolchain plus Swage. Then confirm the
environment and the native bindings:

```bash
python -m swage.env
PYTHONPATH="$PWD/python" python -m pytest tests/python -q
ninja -C build check-swage-python
```

The commands below assume the build-tree bindings are importable:

```bash
export PYTHONPATH=build/python_packages
```

## Write the kernel

A Swage kernel is ordinary-looking Python that is captured, never
executed. The decorator parses the source, validates it against the
restricted kernel language, and returns a kernel object:

```python
import swage as sw
import swage.language as sl

@sw.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK: sl.constexpr):
    """Add two vectors elementwise under a bounds mask."""
    pid = sl.program_id(0)
    offsets = pid * BLOCK + sl.arange(0, BLOCK)
    mask = offsets < n
    x = sl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = sl.load(y_ptr + offsets, mask=mask, other=0.0)
    sl.store(output_ptr + offsets, x + y, mask=mask)
```

Each program instance owns one block of `BLOCK` lanes: `program_id(0)`
names the block, `arange(0, BLOCK)` spreads its lanes, and the mask
retires lanes at or beyond `n`. The kernel is not directly callable;
calling it raises. The accepted source forms are listed in
[Kernel Language](../reference/kernel-language.md).

## Emit and read the MLIR

`emit_mlir()` needs the native bindings but no GPU and no PyTorch when
the signature is explicit (native-build tier):

```python
module = add_kernel.emit_mlir(
    signature={
        "x_ptr": sl.pointer(sl.float32),
        "y_ptr": sl.pointer(sl.float32),
        "output_ptr": sl.pointer(sl.float32),
        "n": sl.int32,
    },
    constexprs={"BLOCK": 128},
)
print(module)
```

The printed module is verified semantic MLIR: one function carrying the
logical `swage.program_id` operation surrounded by ordinary `arith` and
`vector` operations, with source locations preserved. Nothing has touched
a GPU yet.

## Launch on CUDA

The launch needs the CUDA GPU tier: a CUDA-enabled PyTorch build, an
admitted NVIDIA GPU, and the installed driver. Arguments are passed by
name, `BLOCK` stays a compile-time value, and the grid must cover `n`:

```python
import torch

n, block = 1025, 128
x = torch.randn(n, device="cuda", dtype=torch.float32)
y = torch.randn(n, device="cuda", dtype=torch.float32)
output = torch.empty_like(x)

add_kernel.launch(
    arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
    constexprs={"BLOCK": block},
    grid=((n + block - 1) // block,),
)
torch.testing.assert_close(output, torch.add(x, y))
```

The launch validates its complete host-visible boundary first, compiles
in process, and enqueues asynchronously on the current PyTorch stream.
The exact rules live in
[Runtime and Environment](../reference/runtime-environment.md).

## Inspect compiler artifacts

Use an isolated directory for cache and debug artifacts, then run the
committed example with dumps enabled:

```bash
export SWAGE_WALKTHROUGH_DIR="$(mktemp -d)"
export SWAGE_CACHE_DIR="$SWAGE_WALKTHROUGH_DIR/cache"
export SWAGE_DUMP_DIR="$SWAGE_WALKTHROUGH_DIR/dumps"
export SWAGE_DUMP_MLIR=1
export SWAGE_DUMP_PTX=1

PYTHONPATH=build/python_packages python examples/fixed_vector_add.py
find "$SWAGE_DUMP_DIR" -maxdepth 1 -type f -print
```

The dump directory receives the lowered MLIR and the emitted PTX, named
by specialization digest. On an identified clean checkout with its LLVM
pin, a second run exercises persistent-cache verification and reuse; a
dirty or unidentified build reuses compiled work only within the process.

## Where next

Continue with the [User Guide](../user-guide/index.md) for the ideas
behind the kernel, the [swage API reference](../reference/swage.md) for
the exact call contracts, or
[Troubleshooting](troubleshooting.md) when a package, binding, tool, or
CUDA component cannot be found.
