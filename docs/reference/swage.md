<!-- docs/reference/swage.md -->

# swage

The public API is intentionally small. The `swage` package exports `jit`,
`SwageError`, `CompilationError`, `BackendUnavailableError`, and `__version__`;
captured kernels expose `emit_mlir()` and `launch()`; `swage.env` reports the
environment. Segmented Python syntax and segmented launch are not public.

This reference describes the implemented v0.5.2 contract, pending publication
and release qualification; v0.5.1 remains the latest released tag. Native
wheels include the self-contained private `mlir_swage` package needed for
emission and execution. A source build against exact LLVM/MLIR 22.1.8 is an
alternative, not a wheel prerequisite. See
[Installation](../getting-started/installation.md) and the authoritative
[support matrix](runtime-environment.md#support-matrix).

The package ships `py.typed` and stubs for this fixed public contract,
including `Literal["cpu", "cuda"]` backend selection. Symbolic DSL values do
not imply a broader public language or segmented execution surface.

The installed-wheel operational command
`python -m swage.bench vector-add --output result.json` is documented in
[Benchmarking](benchmarking.md). It does not add package exports or kernel
APIs, expose segmented execution, or independently qualify a release.

## swage.jit

```python
swage.jit(function)
```

Capture a Python function as a non-executing Swage kernel. The source is
read and stored; the body never runs as Python and is validated against
the restricted kernel language when the kernel is emitted or launched.

Parameters
:   `function`: the kernel function to capture. Ordinary positional
    parameters only; compile-time parameters carry the exact annotation
    `sl.constexpr`.

Returns
:   A captured kernel object exposing `emit_mlir()` and `launch()`. The
    kernel is not directly callable.

Raises
:   `CompilationError`: the source cannot be captured, for example
    unreadable or ambiguous source, stacked decorators, or a non-ASCII
    kernel name. Kernel-language violations in the body surface later,
    at `emit_mlir()` or `launch()`, with the file, line, and column.
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
    `sl.int32` and `sl.pointer(dtype)` for `sl.float32`, `sl.float16`,
    `sl.float8_e4m3fn`, or `sl.float8_e5m2`. This path does not require PyTorch.
:   `arguments`: example values whose metadata infers the signature. A
    non-boolean Python integer in the signed i32 range infers `int32`; a
    contiguous, strided, rank-one tensor on CPU or CUDA with one of those
    floating dtypes infers the corresponding pointer. Values are never read.
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
:   `BackendUnavailableError`: native bindings cannot be imported or linked
    (`code="native-unavailable"`, `backend="native"`).

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
kernel.launch(*, arguments, constexprs, grid, backend="cuda")
```

Compile as needed and launch a canonical fixed vector add or multiply kernel
on exactly one selected backend. CUDA remains the default and enqueues asynchronously;
`backend="cpu"` invokes a synchronous Native LLVM JIT entry. The call is
keyword-only and returns `None`. Its only public execution contract is the
canonical one-dimensional fixed vector operation with five parameters in this
semantic order:

```text
left input, right input, output, element count, constexpr block size
```

Names such as `x_ptr`, `y_ptr`, `output_ptr`, `n`, and `BLOCK` are
conventional, not required. The `arguments` and `constexprs` mappings use the
names declared by the captured function.

Parameters
:   `arguments`: exactly the four declared runtime parameters. The first
    three are contiguous rank-one tensors on the selected backend with the
    same supported dtype (`float32`, `float16`, `float8_e4m3fn`, or
    `float8_e5m2`); the fourth is a nonnegative i32 no larger than any tensor.
:   `constexprs`: exactly the declared final constexpr parameter, whose value
    is positive and within the selected backend's limit.
:   `grid`: the one-dimensional logical geometry, which must equal the
    ceiling division of the element count by the block size.
:   `backend`: exactly `"cuda"` or `"cpu"`. CUDA requires current-device CUDA
    tensors. CPU requires CPU tensors and uses the process-local `native`
    target.

Returns
:   `None`. CUDA enqueues on the current PyTorch stream and retains submitted
    tensors through `record_stream()`. CPU execution is complete on return.

Low-precision arithmetic widens both inputs to FP32, performs the selected
addition or multiplication, and rounds once to the tensor dtype. It does not
promote or cast tensor storage. The kernel body contains exactly one `x + y`
or `x * y`; chains, floating vector/scalar arithmetic, broadcasting, and
matrix multiplication are unsupported. See
[Dtypes and rounding](runtime-environment.md#dtypes-and-rounding) for
subnormal, overflow, and NaN behavior.

Raises
:   `CompilationError`: the captured source or inferred signature is outside
    the admitted fixed-vector language.
:   `TypeError`: wrong container, tensor, dtype, rank, backend category, or
    ABI category.
:   `ValueError`: unknown backend name, invalid values, geometry, device
    placement, or native compiler admission such as an unsupported `sm_*`
    target.
:   `BackendUnavailableError`: a native, PyTorch, or selected CUDA environment
    prerequisite is unavailable. Inspect `code`, `backend`, and `remediation`.
:   `RuntimeError`: selected-backend execution, driver-call, or cache failures.

Validation, target admission, zero-work, cache, stream, and retention rules
are normative in [Runtime and Environment](runtime-environment.md). There is
no public `emit_ptx()` method, no fallback between CPU and CUDA, and no public
segmented launch.

## swage.SwageError

```python
class swage.SwageError(RuntimeError)
```

Public base for Swage-specific compiler and runtime errors. Existing
`RuntimeError` catchers also catch its subclasses. This does not replace
`TypeError`, `ValueError`, or every runtime/driver exception.

## swage.CompilationError

```python
class swage.CompilationError(swage.SwageError)
```

A source-located error in a Swage kernel definition. Raised during capture,
emission, or launch when source or frontend inputs violate the admitted
language; the unchanged message names the offending file, line, and column.
This includes unavailable or unreadable PyTorch metadata on the inference
path.

## swage.BackendUnavailableError

```python
class swage.BackendUnavailableError(swage.SwageError)
```

An environment-prerequisite failure, with stable string attributes:

- `code`: one of the values below.
- `backend`: the attempted component, `"native"`, `"cpu"`, or `"cuda"`.
- `remediation`: an actionable suggestion, not a machine-readable error code.

| Code | Missing prerequisite |
|---|---|
| `native-unavailable` | Native compiler bindings cannot be imported or linked. |
| `pytorch-unavailable` | PyTorch cannot be imported for launch. |
| `cuda-unavailable` | PyTorch reports CUDA unavailable. |
| `cuda-driver-unavailable` | `libcuda.so.1` cannot be loaded. |
| `cuda-context-unavailable` | There is no current PyTorch CUDA context. |

Native import/link failures preserve exception chaining. Arbitrary compiler
exceptions are not relabeled as missing bindings. Catch this class around an
explicitly selected launch when presenting remediation; do not retry on the
other backend.

## Exceptions

- `CompilationError` reports source-located capture, emission, and launch-time
  frontend validation failures.
- `BackendUnavailableError` reports only unavailable environment prerequisites.
- `TypeError` reports launch inputs with the wrong container, tensor,
  dtype, rank, or ABI category.
- `ValueError` reports invalid launch values, geometry, device placement, or
  native compiler admission, including unsupported CUDA targets.
- `RuntimeError` reports direct kernel calls, symbolic language calls outside
  a captured kernel, and runtime driver or cache failures. Cache-integrity,
  contract-validation, compilation, and driver-call failures are not turned
  into backend-availability errors.

## swage.\_\_version\_\_

```python
swage.__version__
```

The installed package version string.

## swage.env

```python
from swage.env import report

environment = report()
```

```bash
python -m swage.env
python -m swage.env --json
python -m swage.env --json --check native
python -m swage.env --json --check cpu
python -m swage.env --json --check cuda
```

`report()` returns a non-throwing, schema-versioned dictionary, including when
native bindings or PyTorch are absent. The default CLI prints key/value
lines; `--json` prints one sorted JSON object to stdout. Without `--check`,
the CLI exits zero. A selected unavailable component exits one, still
emitting the report.
Invalid CLI options exit nonzero. These are prerequisite checks, not kernel
execution or release qualification.

The complete schema, malformed-build-metadata distinction, qualified hardware,
cache identity, and opt-in `swage.runtime` logging contract are defined in
[Runtime and Environment](runtime-environment.md#environment-report).

Continue with [swage.language](swage-language.md) for the kernel-language
exports, or [Kernel Language](kernel-language.md) for the accepted source
grammar.
