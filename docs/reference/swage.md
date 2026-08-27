<!-- docs/reference/swage.md -->

# swage

The public API is intentionally small. The `swage` package exports `jit`,
`CompilationError`, and `__version__`; captured kernels expose
`emit_mlir()` and `launch()`; `swage.env` reports the environment.
Segmented Python syntax and segmented launch are not public.

Compile-only emission and execution require the build-tree `mlir_swage`
package from [Installation](../getting-started/installation.md). The
published wheel captures kernels and reports the environment on its own.

## swage.jit

```python
swage.jit(function)
```

Capture a Python function as a non-executing Swage kernel. The function
body is parsed and validated against the restricted kernel language; it
never runs as Python.

Parameters
:   `function`: the kernel function to capture. Ordinary positional
    parameters only; compile-time parameters carry the exact annotation
    `sl.constexpr`.

Returns
:   A captured kernel object exposing `emit_mlir()` and `launch()`. The
    kernel is not directly callable.

Raises
:   `CompilationError`: the source is outside the accepted kernel
    language. The message carries the file, line, and column.
:   `RuntimeError`: the returned kernel is called directly.

Example

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

Related: [Kernel Language](kernel-language.md),
[swage.language](swage-language.md).

## Kernel.emit_mlir

```python
kernel.emit_mlir(*, signature=None, arguments=None, constexprs)
```

Emit and return a live, verified native MLIR module for the captured
kernel. Emission does not read tensor data pointers or contents, retain
arguments, launch work, or return a runtime result.

Parameters
:   `signature`: explicit parameter types. Accepts
    `sl.pointer(sl.float32)` and `sl.int32`. This path does not require
    PyTorch.
:   `arguments`: example values whose metadata infers the signature. A
    non-boolean Python integer in the signed i32 range infers `int32`; a
    contiguous, strided, rank-one `torch.float32` tensor on CPU or CUDA
    infers a pointer. Values are never read.
:   `constexprs`: the compile-time values, always required. Must contain
    exactly the declared compile-time parameters.

Exactly one of `signature` or `arguments` is required, and each mapping
must contain exactly the parameters declared for its mode.

Returns
:   A verified live `mlir_swage.ir.Module` with source locations
    preserved.

Raises
:   `CompilationError`: capture or input failures, including unavailable
    or unreadable PyTorch metadata on the inference path.
:   `RuntimeError`: the build-tree `mlir_swage` package is missing.

Example

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
```

Related: [Writing Kernels](../user-guide/writing-kernels.md).

## Kernel.launch

```python
kernel.launch(*, arguments, constexprs, grid)
```

Compile as needed and asynchronously launch the canonical fixed
vector-add kernel on CUDA. The call is keyword-only and returns `None`.
Its only public execution contract is the canonical one-dimensional
fixed vector add with parameters in this order:

```text
x_ptr, y_ptr, output_ptr, n, BLOCK
```

Parameters
:   `arguments`: `x_ptr`, `y_ptr`, `output_ptr`, and `n`. The pointers
    are contiguous rank-one `torch.float32` CUDA tensors on the current
    device; `n` is a nonnegative i32 no larger than any tensor.
:   `constexprs`: exactly `BLOCK`, a positive integer within the active
    device limit.
:   `grid`: the one-dimensional launch geometry, which must equal
    `(ceildiv(n, BLOCK),)`.

Returns
:   `None`. The launch enqueues asynchronously on the current PyTorch
    stream; submitted tensors are retained through `record_stream()`.

Raises
:   `TypeError`: wrong container, tensor, dtype, rank, or ABI category.
:   `ValueError`: invalid values, geometry, device placement, or native
    compiler admission such as an unsupported `sm_*` target.
:   `RuntimeError`: missing native bindings, missing PyTorch,
    unavailable CUDA, or runtime driver and cache failures.

Validation, target admission, zero-work, cache, stream, and retention
rules are normative in
[Runtime and Environment](runtime-environment.md).

## swage.CompilationError

```python
class swage.CompilationError(Exception)
```

A source-located error in a Swage kernel definition. Raised at capture
and by `emit_mlir()` input validation; the message names the offending
file, line, and column.

## Exceptions

The public surface uses four exception classes:

- `CompilationError` reports source-located capture and `emit_mlir()`
  input failures.
- `TypeError` reports launch inputs with the wrong container, tensor,
  dtype, rank, or ABI category.
- `ValueError` reports invalid launch values, geometry, device
  placement, or native compiler admission.
- `RuntimeError` reports direct kernel calls, symbolic language calls
  outside a captured kernel, missing native bindings, missing PyTorch
  for launch, unavailable CUDA, and runtime driver or cache failures.

## swage.\_\_version\_\_

```python
swage.__version__
```

The installed package version string.

## swage.env

```bash
python -m swage.env
```

Print the environment report as flat key and value lines. The report
never fails: unavailable components are reported as absent instead of
raising. Its keys are `swage`, `python`, `platform`, `torch`,
`torch_cuda_build`, `cuda_driver`, `cuda`, `gpu`, `llvm_pin`, and
`backends`. The `backends` field describes what is built into the
installed package; it does not detect a separate build-tree
`mlir_swage` package.

Continue with [swage.language](swage-language.md) for the kernel-language
exports, or [Kernel Language](kernel-language.md) for the accepted source
grammar.
