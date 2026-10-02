<!-- docs/reference/swage.md -->

# swage

The public API is intentionally small. The `swage` package exports `jit`,
`CompilationError`, `segment_reduce`, `segment_softmax`, and `__version__`;
captured kernels expose `emit_mlir()` and `launch()`; `swage.env` reports
the environment, and `swage.compile` writes the kernels of the segmented
calls ahead of time. The two segmented calls run fixed programs. Segmented
Python syntax is not public, and neither are the prepared launches, the
scheduling policies, and the planning limits of the runner behind the calls.

Compile-only emission and execution require the build-tree `mlir_swage`
package from [Installation](../getting-started/installation.md). On its own
the pure Python package captures kernels, checks a kernel against the kernel
language, and reports the environment. It also runs the two segmented calls
from an artifact directory that `swage.compile` wrote on a host with that
package. The installation page lists what the released `0.5.1` wheel lacks,
which includes the two segmented calls.

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
`emit_ptx()` method and no CPU execution fallback, and `launch()` runs no
segmented kernel.

## swage.segment_reduce

```python
swage.segment_reduce(values, offsets, kind, *, out=None)
```

Reduce every segment of `values` to one f32 result on the GPU. Segment `i`
is `values[offsets[i]:offsets[i + 1]]`. The call validates its tensors,
copies the offsets to the host to validate and classify them, enqueues its
kernels on the current PyTorch CUDA stream, and returns without waiting for
them. Every call repeats the host work, so with offsets that change on every
call it is slower than `torch.segment_reduce`.
[Segmented Calls](../user-guide/segmented-calls.md#what-a-call-costs) cites
the committed record of the private preparation that the call repeats; no
record times the call itself.

Parameters
:   `values`: a contiguous rank-one `torch.float32` CUDA tensor on the
    current device. It must not require grad and must not be a lazy
    negation or conjugate view.
:   `offsets`: a contiguous rank-one `torch.int32` tensor on the same
    device with one entry more than there are segments. It starts at zero,
    never decreases, and ends at or below the number of values. Values past
    the final offset belong to no segment.
:   `kind`: `"sum"` or `"max"`. The sum of an empty segment is `0.0` and
    its maximum is negative infinity. A maximum over a NaN is NaN. A sum
    follows IEEE-754 addition, and its rounding depends on the schedule the
    call selects. No argument pins the schedule.
:   `out`: an optional result tensor, keyword-only. A contiguous rank-one
    `torch.float32` tensor on the device of `values` with exactly one
    element per segment, which shares no memory with `values` or `offsets`,
    does not require grad, and is not a lazy view. It is never resized.

Returns
:   `out`, or a new `torch.float32` tensor on the device of `values` when
    `out` is `None`, with one element per segment. The kernels that write
    it are enqueued and may not have finished. Submitted tensors are
    retained through `record_stream()`, and the version counter of the
    result is advanced when a kernel is enqueued. `values`, `offsets`, and
    `out` may be inference tensors.

Raises
:   `TypeError`: an argument is not a tensor, or a tensor has the wrong
    dtype, rank, or device type.
:   `ValueError`: an unsupported `kind`; a tensor that is not contiguous,
    is a lazy view, requires grad, or is on another device; offsets that
    break the offsets contract; an `out` of the wrong size or one that
    overlaps an input.
:   `RuntimeError`: missing PyTorch, a PyTorch older than 2.6, missing
    native bindings while no artifact is selected, an artifact selected by
    `SWAGE_ARTIFACT_DIR` that cannot be used or does not hold the kernels of
    the call, a missing `numpy`, unavailable CUDA, a current stream that is
    capturing a CUDA
    graph, a kernel that the process does not hold while
    `SWAGE_NO_COMPILE=1` is set and no artifact is selected, or a runtime
    driver failure.

The checks run in this order: the PyTorch check, `kind`, the tensor type of
`values` and `offsets` and the grad state of `values`, `out`, the selected
artifact or the native bindings, `numpy`, CUDA graph capture, and then the
shared validation of dtype, rank, layout, offsets, and device. All of them
precede the first enqueue.

Example

```python
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int32, device="cuda")

totals = swage.segment_reduce(values, offsets, "sum")  # [3, 0, 12, 6]
maxima = swage.segment_reduce(values, offsets, "max")  # [2, -inf, 5, 6]
```

Related: [Segmented Calls](../user-guide/segmented-calls.md),
[Ragged Data](../user-guide/ragged-data.md#the-offsets-contract),
[Runtime and Environment](runtime-environment.md#segmented-calls).

## swage.segment_softmax

```python
swage.segment_softmax(values, offsets, *, out=None)
```

Apply a softmax within every segment of `values` on the GPU. The result
holds the softmax of each segment at the positions of its values. The call
validates its tensors, copies the offsets to the host to validate them,
enqueues one kernel on the current PyTorch CUDA stream, and returns without
waiting for it.

Parameters
:   `values`: as for `segment_reduce`.
:   `offsets`: as for `segment_reduce`, with one difference. The final
    offset must equal the number of values, so that every value belongs to
    a segment.
:   `out`: as for `segment_reduce`, with exactly one element per value.

Returns
:   `out`, or a new `torch.float32` tensor on the device of `values` when
    `out` is `None`, with one element per value. An empty segment has no
    result element. A segment that holds a NaN or a positive infinity, or
    only negative infinities, gives NaN for each of its elements. The
    version counter of the result is advanced when a kernel is enqueued.

Raises
:   The exceptions of `segment_reduce`, without the `kind` error. Offsets
    that end below the number of values raise a `ValueError`.

Example

```python
weights = swage.segment_softmax(values, offsets)
```

Related: [Segmented Calls](../user-guide/segmented-calls.md),
[Ragged Softmax](../internals/ragged-softmax.md#accuracy).

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
- `TypeError` reports launch and segmented-call inputs with the wrong
  container, tensor, dtype, rank, or ABI category.
- `ValueError` reports invalid launch values, geometry, device placement,
  lazy views, tensors that require grad, overlapping buffers, native
  compiler admission, a cache variable with an undocumented value, an
  unsupported reduction kind, and offsets outside the offsets contract.
- `RuntimeError` reports direct kernel calls, symbolic language calls
  outside a captured kernel, missing native bindings, a missing or
  unsupported PyTorch for launch, unavailable CUDA, a refused compile under
  `SWAGE_NO_COMPILE=1`, a segmented call under CUDA graph capture, an
  artifact directory that cannot be used, and runtime driver or cache
  failures.

## swage.\_\_version\_\_

```python
swage.__version__
```

The installed package version string.

## swage.compile

```bash
python -m swage.compile --target TARGET --output DIRECTORY
    [--program {sum,max,softmax}] [--runtime-library LIBRARY]
```

Compile every kernel that `segment_reduce` and `segment_softmax` can launch
for one NVPTX processor, and write the PTX, the runtime library, and a
manifest to a new directory. The command needs the native `mlir_swage`
package and `numpy`. It needs no GPU and no PyTorch.

Options
:   `--target`: the NVPTX processor of the device that will run the
    kernels, such as `sm_86`. Required.
:   `--output`: the directory to create. It must not exist. Required.
:   `--program`: a program to include, `sum`, `max`, or `softmax`. It may
    be repeated. All three are included without it.
:   `--runtime-library`: a `libSwageRuntime.so` to ship in place of the one
    of the native build, for a serving host of another machine.

Output and exit status
:   On success the command prints the directory, the manifest format, the
    target, the programs, the number of kernels, the runtime library with
    its machine, and the SHA-256 digest of the manifest, one `key: value`
    line each, and exits with status 0.
:   Otherwise it prints `error:` and the reason on standard error, writes
    nothing, and exits with status 1: for missing bindings, a target the
    compiler rejects, an output directory that exists, a set
    `SWAGE_ARTIFACT_DIR`, `SWAGE_NO_COMPILE=1`, and a runtime library that
    is not an ELF library for `x86_64` or `aarch64`.

The module is a command: it defines no public function or class. A process
runs the two calls from the directory when `SWAGE_ARTIFACT_DIR` names it.
[Running Without the Compiler](../user-guide/deployment.md) describes the
files and the manifest, and
[Runtime and Environment](runtime-environment.md#artifacts) states what the
runtime verifies.

## swage.env

```bash
python -m swage.env
```

Print the environment report as flat key and value lines. The report
never fails: unavailable components are reported as absent instead of
raising. Its keys, in order, are `swage`, `revision`, `swage_file`,
`python`, `platform`, `torch`, `torch_cuda_build`, `cuda_driver`, `cuda`,
`gpu`, `target`, `llvm_pin`, `llvm_linked`, `native_version`,
`native_revision`, `mlir_swage_file`, `backends`, `cache_dir`, `cache`,
`compile_on_miss`, and `artifact`.

- `swage_file` and `mlir_swage_file` name the package file and the native
  extension that were imported.
- `target` names the NVPTX processor of the current device and says
  whether it is qualified, admitted and not qualified, or not admitted.
- `backends` records whether the build-tree `mlir_swage` bindings import
  in the reporting process and, when they do, the LLVM version they were
  linked against.
- `cache_dir`, `cache`, and `compile_on_miss` describe the persistent
  cache as the reporting process would use it.
- `artifact` names the directory that `SWAGE_ARTIFACT_DIR` selects for the
  segmented calls, or the reason it is rejected.

[Runtime and Environment](runtime-environment.md#environment-report)
defines every field.

Continue with [swage.language](swage-language.md) for the kernel-language
exports, or [Kernel Language](kernel-language.md) for the accepted source
grammar.
