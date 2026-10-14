<!-- docs/getting-started/quickstart.md -->

# Quickstart

This tutorial takes a canonical fixed vector-add kernel from source capture
to a verified result on an explicitly selected CPU or CUDA backend, and then
reduces a ragged batch with the two segmented calls on CUDA. It uses the
`v0.5.2` native-wheel contract, whose publication and release qualification
are still pending; see [Installation](installation.md) for a local wheel
build and the source build. Four committed scripts follow the same steps:

- `examples/emit_fixed_vector_add.py` stops after emission, so it needs no
  GPU and no PyTorch.
- [`examples/fixed_vector_add.py`](https://github.com/abhiksark/swage/blob/main/examples/fixed_vector_add.py)
  runs the kernel walkthrough as one script, using metadata inference in
  place of the explicit signature.
- [`examples/fixed_vector_multiply.py`](https://github.com/abhiksark/swage/blob/main/examples/fixed_vector_multiply.py)
  uses the same ABI with `x * y`.
- [`examples/segment_reduce.py`](https://github.com/abhiksark/swage/blob/main/examples/segment_reduce.py)
  runs the segmented calls and needs a CUDA GPU.

Python source crosses a restricted AST validation boundary before becoming
verified semantic MLIR. From that point, `emit_mlir()` stops with a
compile-only module and needs no GPU. The canonical `launch()` path
continues through native compilation to the selected backend. Native
packaging does not change the semantics of the fixed kernel.

<div class="doc-figure" tabindex="0" markdown="1">

![Frontend validation, compile-only emission, and canonical launch branches](../assets/diagrams/frontend-boundary.svg)

</div>

*The verified frontend boundary and its two public outcomes. [Open the full-size figure](../assets/diagrams/frontend-boundary.svg).*

## Prerequisites

Follow [Installation](installation.md) to install a native wheel and a
suitable PyTorch build. No source checkout or local LLVM tree is needed to
use the wheel. Check the environment and choose one backend:

```bash
python -m swage.env --json --check native
python -m swage.env --json --check cpu
# Run this check if selecting CUDA:
python -m swage.env --json --check cuda
```

CPU works without a GPU. CUDA needs a CUDA-enabled PyTorch build and an
admitted device and driver; the release gate targets the A6000 (`sm_86`),
not every admitted GPU. See the [Support Matrix](../reference/support-matrix.md).
Only source-build users need to set `PYTHONPATH=build/python_packages`.

Save the Python snippets below together in a `.py` file so that `@sw.jit`
can read the kernel's source; do not define the kernel only at an
interactive prompt.

## Write the kernel

A Swage kernel is ordinary-looking Python that is captured, never
executed. The decorator reads the source and returns a kernel object;
the body is validated against the restricted kernel language when the
kernel is emitted or launched:

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
retires lanes at or beyond `n`. `sl.load` requires both `mask=` and
`other=`, and `sl.store` requires `mask=`. The kernel is not directly
callable; calling it raises. The accepted source forms are listed in
[Kernel Language](../reference/kernel-language.md).

Changing only the stored value to `x * y` selects multiplication. The public
subset accepts exactly one `x + y` or `x * y`; it does not accept operation
chains, floating vector and scalar arithmetic, broadcasting, or matrix
multiplication.

## Emit and read the MLIR

`emit_mlir()` uses the native bindings that the wheel includes. It needs no
GPU and no PyTorch when the signature is explicit (wheel-only tier):

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
a GPU yet. The emit-only example runs this step as a script, which is a way
to check an install on a machine without a GPU:

```bash
python examples/emit_fixed_vector_add.py
```

## Launch on CPU or CUDA

Arguments are passed by name, `BLOCK` stays a compile-time value, and the
grid must cover `n`. Start with an explicit CPU selection:

```python
import torch

backend = "cpu"  # Choose "cuda" explicitly to run on CUDA instead.
n, block = 1025, 128
x = torch.randn(n, device=backend, dtype=torch.float32)
y = torch.randn(n, device=backend, dtype=torch.float32)
output = torch.empty_like(x)

add_kernel.launch(
    arguments={"x_ptr": x, "y_ptr": y, "output_ptr": output, "n": n},
    constexprs={"BLOCK": block},
    grid=((n + block - 1) // block,),
    backend=backend,
)
if backend == "cuda":
    torch.cuda.synchronize()
torch.testing.assert_close(output, torch.add(x, y), rtol=0, atol=0)
```

The launch validates its complete host-visible boundary first and compiles
in process. CPU execution completes before return; CUDA enqueues
asynchronously on the current PyTorch stream, so this example synchronizes
before checking the result. The API default remains CUDA, but examples
choose explicitly. Failure on one backend never attempts the other. The
exact rules live in [Runtime and Environment](../reference/runtime-environment.md).

From a checkout containing the committed examples, using the installed
wheel:

```bash
python examples/fixed_vector_add.py --backend cpu
python examples/fixed_vector_add.py --backend cuda
python examples/fixed_vector_add.py --backend cpu --dtype float16
python examples/fixed_vector_add.py --backend cuda --dtype float8_e4m3fn
python examples/fixed_vector_add.py --backend cuda --dtype float8_e5m2
python examples/fixed_vector_multiply.py --backend cpu
python examples/fixed_vector_multiply.py --backend cuda --dtype float8_e4m3fn
```

Choose the backend and dtype to exercise. `--dtype` defaults to `float32`;
all four dtypes work on either backend. The example creates low-precision
inputs by casting generated FP32 data and checks the result against the
selected FP32 operation rounded back to that dtype. Swage itself never casts
or moves tensor storage. The installed runtime does not need the example
source; outside a checkout, download the linked script and run it with the
same options.

## Reduce segments on CUDA

Ragged data needs no kernel of your own. Two functions run fixed programs
over a values tensor and the offsets that divide it into segments (CUDA GPU
tier):

```python
import swage

# Six values in four segments: [1, 2], [], [3, 4, 5], and [6].
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int32, device="cuda")

totals = swage.segment_reduce(values, offsets, "sum")
maxima = swage.segment_reduce(values, offsets, "max")
weights = swage.segment_softmax(values, offsets)

print(totals.tolist())  # [3.0, 0.0, 12.0, 6.0]
print(maxima.tolist())  # [2.0, -inf, 5.0, 6.0]
```

`weights` holds the softmax of each segment at the positions of its
values. The empty second segment sums to `0.0`, has a maximum of negative
infinity, and has no softmax element.

Each call validates and classifies its offsets on the host before it
launches, every time, so it costs more than its kernels. With offsets that
change on every call, expect it to be slower than `torch.segment_reduce`.
The values above are rank-one `torch.float32`. Both calls also take
`[N, D]` rows of features, which they reduce or normalize per column, and a
reduction also takes `torch.float64` values and the kinds `"min"` and
`"mean"`. The offsets are `torch.int32` or `torch.int64`. The calls run on
the current CUDA device only, record no gradient, and are refused under
CUDA graph capture. [Segmented Calls](../user-guide/segmented-calls.md)
states the whole contract, including the cost and the cases a call refuses.

The committed example runs the same calls and compares them with PyTorch:

```bash
python examples/segment_reduce.py
```

## Inspect compiler artifacts

Use an isolated directory for cache and debug artifacts, then run the
committed example with dumps enabled (CUDA GPU tier):

```bash
export SWAGE_WALKTHROUGH_DIR="$(mktemp -d)"
export SWAGE_CACHE_DIR="$SWAGE_WALKTHROUGH_DIR/cache"
export SWAGE_DUMP_DIR="$SWAGE_WALKTHROUGH_DIR/dumps"
export SWAGE_DUMP_MLIR=1
export SWAGE_DUMP_PTX=1

python examples/fixed_vector_add.py --backend cuda
find "$SWAGE_DUMP_DIR" -maxdepth 1 -type f -print
```

The dump directory receives the lowered MLIR and the emitted PTX, named by
specialization digest. A second CUDA run in a new process exercises
persistent-cache verification and reuse, whether or not the checkout is
clean. A process that cannot identify its frontend sources or its native
libraries reuses compiled work only within the process, and CPU executables
are never persisted. See
[Specialization and cache](../reference/runtime-environment.md#specialization-and-cache).

For value-free cache and launch diagnostics, configure logging in your
application before launching:

```python
import logging

logging.basicConfig(level=logging.WARNING)
logging.getLogger("swage.runtime").setLevel(logging.DEBUG)
```

Swage installs no handlers and is silent by default. DEBUG records report
cache outcomes and backend launch events, not tensors, pointers, PTX, or
cache paths. The debug dumps above are separate, explicit writes of
compiler artifacts; do not publish them as though they were sanitized logs.

## Where next

Continue with the [User Guide](../user-guide/index.md) for the ideas
behind the kernel and the segmented calls, the
[swage API reference](../reference/swage.md) for the exact call contracts,
or [Troubleshooting](troubleshooting.md) when a package, binding, tool, or
CUDA component cannot be found.
