<!-- docs/reference/swage.md -->

# Public Python API

The public API is intentionally small. It captures the supported fixed
vector-add kernel, emits verified MLIR through native bindings, and launches
that canonical kernel on CUDA. Segmented Python syntax and segmented launch
are not public.

## Package exports

```python
swage.jit(function)
swage.CompilationError
swage.__version__
```

`jit(function)` returns a captured kernel object without executing the
function body. A decorated kernel is not directly callable.

## Compile-only emission

```python
kernel.emit_mlir(*, signature=None, arguments=None, constexprs)
```

Exactly one of `signature` or `arguments` is required. `constexprs` is always
required. All mappings must contain exactly the parameters declared for their
respective mode.

Explicit signatures accept:

```python
sl.pointer(sl.float32)
sl.int32
```

Argument inference accepts a non-boolean Python integer in the signed i32
range, or a contiguous, strided, rank-one `torch.float32` tensor on CPU or
CUDA. Inference reads tensor metadata only and discards the values.

`emit_mlir()` returns a verified live `mlir_swage.ir.Module`. It does not read
tensor data pointers or contents, retain arguments, launch work, or return a
runtime result. It requires the build-tree `mlir_swage` package. The explicit
signature path does not require PyTorch.

## Fixed vector-add launch

```python
kernel.launch(*, arguments, constexprs, grid)
```

`launch()` is keyword-only and returns `None`. Its only public execution
contract is the canonical one-dimensional fixed vector add with parameters in
this order:

```text
x_ptr, y_ptr, output_ptr, n, BLOCK
```

`arguments` contains the first four names, `constexprs` contains `BLOCK`, and
`grid` provides the one-dimensional launch geometry. Exact validation, target
admission, zero-work, cache, stream, and retention rules live in
[Runtime and Environment](runtime-environment.md).

There is no public `emit_ptx()` method, direct kernel call, CPU execution
fallback, or public segmented launch.

## Exceptions

- `CompilationError` reports source-located capture and `emit_mlir()` input
  failures. This includes unavailable or unreadable PyTorch metadata when the
  `arguments` inference path is selected.
- `TypeError` reports launch inputs with the wrong container, tensor, dtype,
  rank, or ABI category.
- `ValueError` reports invalid launch values, geometry, device placement, or
  native compiler admission such as an unsupported `sm_*` target.
- `RuntimeError` reports direct kernel calls, symbolic language calls outside
  a captured kernel, missing native bindings, missing PyTorch for launch,
  unavailable CUDA, and runtime driver or cache failures.

## Kernel language symbols

`swage.language` exports:

```python
arange
constexpr
float32
int32
load
pointer
program_id
store
```

The symbolic functions are valid only inside a captured kernel. Their exact
accepted source forms are listed in [Kernel Language](kernel-language.md).
For installation and native-package requirements, continue with
[Installation](../getting-started/installation.md). For stream, cache, and
diagnostic behavior, continue with [Runtime and Environment](runtime-environment.md).
