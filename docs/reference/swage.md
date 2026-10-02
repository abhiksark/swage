<!-- docs/reference/swage.md -->

# swage

The public API is intentionally small. The `swage` package exports `jit`,
`CompilationError`, and `__version__`; captured kernels expose
`emit_mlir()` and `launch()`; `swage.env` reports the environment.
Segmented Python syntax and segmented launch are not public.

Compile-only emission and execution require the build-tree `mlir_swage`
package from [Installation](../getting-started/installation.md). On its own
the pure Python package captures kernels, checks a kernel against the kernel
language, and reports the environment. That page lists what the released
`0.5.1` wheel lacks.

## swage.jit

```python
swage.jit(function)
```

Capture a Python function as a non-executing Swage kernel. The source is
read and stored; the body never runs as Python and is validated against
the restricted kernel language when the kernel is emitted or launched.

Parameters
:   `function`: the kernel function to capture. Ordinary positional
    parameters only, with no default values. A compile-time parameter
    carries the annotation `constexpr`, written as an attribute of a name
    bound to the `swage.language` module, such as `sl.constexpr`. No other
    parameter annotation is accepted, and `-> None` is the only accepted
    return annotation.

Returns
:   A captured kernel object exposing `emit_mlir()` and `launch()`. The
    kernel is not directly callable.

Raises
:   `CompilationError`: the source cannot be captured: it is unavailable or
    does not parse, the kernel has a stacked decorator, or the kernel name
    is not an ASCII identifier. Violations in the parameter list and the
    body surface later, at `emit_mlir()` or `launch()`, with the file,
    line, and column.
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

The inputs, the parameter list, and the body are checked before the native
package is imported. A wheel-only install therefore answers whether a kernel
is inside the kernel language: a kernel outside it raises the same
`CompilationError` as with the native build, and a kernel inside it raises
the `RuntimeError` below.

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
:   `CompilationError`: input failures, including unavailable or unreadable
    PyTorch metadata on the inference path, and a parameter list or body
    outside the kernel language.
:   `RuntimeError`: the build-tree `mlir_swage` package is missing. The
    message says that the kernel passed the language check and names the
    installation page.

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
    device; `n` is a nonnegative i32 no larger than any tensor. No tensor
    may be a lazy negation or conjugate view, and no tensor may require
    grad. The output must not share memory with either input, which also
    rules out in-place use; the two inputs may share memory with each
    other.
:   `constexprs`: exactly `BLOCK`, a positive integer within the active
    device limit.
:   `grid`: the one-dimensional launch geometry, which must equal
    `(ceildiv(n, BLOCK),)`.

Returns
:   `None`. The launch enqueues asynchronously on the current PyTorch
    stream; submitted tensors are retained through `record_stream()`, and
    the version counter of the output is advanced.

Raises
:   `TypeError`: wrong container, tensor, dtype, rank, or ABI category.
:   `ValueError`: invalid values, geometry, or device placement; a tensor
    that is not contiguous, is a lazy negation or conjugate view, or
    requires grad; an output that overlaps an input; native compiler
    admission such as an unsupported `sm_*` target; or a cache variable
    with a value other than the documented ones.
:   `RuntimeError`: missing PyTorch, a PyTorch older than 2.6 or without
    `torch.Tensor.record_stream` or
    `torch.autograd.graph.increment_version`, unavailable CUDA, missing
    native bindings, a kernel that is not cached while `SWAGE_NO_COMPILE=1`
    is set, or runtime driver and cache failures.
:   `CompilationError`: a parameter list outside the kernel language, on
    every call, or a body outside it, when the call compiles the kernel.

The PyTorch check runs first and validation second, both before any kernel
is compiled or enqueued. Validation, target admission, zero-work, cache,
stream, retention, and module-lifetime rules are normative in
[Runtime and Environment](runtime-environment.md). There is no public
`emit_ptx()` method, no CPU execution fallback, and no public segmented
launch.

## swage.CompilationError

```python
class swage.CompilationError(Exception)
```

A source-located error in a Swage kernel definition. Raised at capture,
by `emit_mlir()`, and by `launch()`; the message names the offending file,
line, and column.

## Exceptions

The public surface uses four exception classes:

- `CompilationError` reports a source-located failure at capture, in the
  inputs of `emit_mlir()`, or in a kernel outside the kernel language.
- `TypeError` reports launch inputs with the wrong container, tensor,
  dtype, rank, or ABI category.
- `ValueError` reports invalid launch values, geometry, device placement,
  lazy views, tensors that require grad, overlapping buffers, native
  compiler admission, or a cache variable with an undocumented value.
- `RuntimeError` reports direct kernel calls, symbolic language calls
  outside a captured kernel, missing native bindings, a missing or
  unsupported PyTorch for launch, unavailable CUDA, a refused compile under
  `SWAGE_NO_COMPILE=1`, and runtime driver or cache failures.

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
raising. Its keys, in order, are `swage`, `revision`, `swage_file`,
`python`, `platform`, `torch`, `torch_cuda_build`, `cuda_driver`, `cuda`,
`gpu`, `target`, `llvm_pin`, `llvm_linked`, `mlir_swage_file`, `backends`,
`cache_dir`, `cache`, and `compile_on_miss`.

- `swage_file` and `mlir_swage_file` name the package file and the native
  extension that were imported.
- `target` names the NVPTX processor of the current device and says
  whether it is qualified, admitted and not qualified, or not admitted.
- `backends` records whether the build-tree `mlir_swage` bindings import
  in the reporting process and, when they do, the LLVM version they were
  linked against.
- `cache_dir`, `cache`, and `compile_on_miss` describe the persistent
  cache as the reporting process would use it.

[Runtime and Environment](runtime-environment.md#environment-report)
defines every field.

Continue with [swage.language](swage-language.md) for the kernel-language
exports, or [Kernel Language](kernel-language.md) for the accepted source
grammar.
