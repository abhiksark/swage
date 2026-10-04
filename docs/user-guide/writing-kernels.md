<!-- docs/user-guide/writing-kernels.md -->

# Writing Kernels

[Segmented Calls](segmented-calls.md) ran two fixed programs over segments,
and [Running Without the Compiler](deployment.md) served them from kernels
that were compiled ahead of time. This page leaves segments aside: the
public kernel language has no segment syntax, and the kernels that can be
written and launched are a fixed-block vector add and multiply.

A Swage kernel is ordinary-looking Python that is captured, never
executed. This page explains what each line of the canonical kernel
means and what the frontend does with it. The accepted grammar is
normative in [Kernel Language](../reference/kernel-language.md).

## Capture, not execution

`@swage.jit` reads and parses the function source. The body never runs
as Python: there is no tracing, no example input, and no hidden
execution, and the restricted kernel language is enforced when the
kernel is emitted or launched. What comes back is a kernel object whose
only public methods are `emit_mlir()` and `launch()`; calling the
kernel directly raises. The four symbolic functions in `swage.language`
share that property and raise outside a captured kernel; the types and
markers work anywhere.

Capture is fail closed. Anything outside the accepted grammar, a loop,
an unsupported operator, a stray keyword argument, fails with a
source-located `CompilationError` naming the file, line, and column.
Nothing partial survives. Decoration rejects source it cannot read or
parse and a stacked decorator; `emit_mlir()` rejects everything else in
the parameter list and the body.

## The canonical kernel, line by line

Capture itself needs only the `swage` package and does not use the native
bindings, so it also works in a frontend-only install:

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

- The parameters are the kernel's ABI, in order: three pointers with the
  same supported floating element dtype, an i32 count, and the
  compile-time block width. `BLOCK` is marked with the annotation
  `sl.constexpr`, so it is bound at compile time and never passed at
  launch. No other parameter annotation and no default value is accepted.
  A kernel returns nothing, so `-> None` is the only return annotation
  accepted.
- `sl` is the conventional import name. The frontend recognizes the
  `swage.language` module by object, so another import name works as
  long as the kernel uses it for the marker and for every call.
- `sl.program_id(0)` is the logical block coordinate. It is a semantic
  index, not a GPU thread ID; the lowering decides how it maps to
  hardware.
- `sl.arange(0, BLOCK)` spreads one block into `BLOCK` lanes, so
  `pid * BLOCK + arange(0, BLOCK)` is each lane's global element index.
- The mask compares those indices against `n` once and guards both
  loads and the store, so the tail block reads the `other` value and
  writes nothing out of bounds. `sl.load` requires both `mask=` and
  `other=`, `sl.store` requires `mask=`, and an `other=` literal must be
  representable in the storage dtype. The launch geometry this implies is
  drawn on [Kernel Language](../reference/kernel-language.md).
- The same source accepts `float32`, `float16`, `float8_e4m3fn`, and
  `float8_e5m2` tensors. The input metadata selects a distinct
  specialization; see
  [Dtypes and rounding](../reference/runtime-environment.md#dtypes-and-rounding).

## Check a kernel without native bindings

`emit_mlir()` checks the parameter list and the body before it imports
the native package, so an install without the bindings, such as the
frontend-only editable install of a checkout, still answers whether a
kernel is inside the language. The released `0.5.1` wheel predates this
check; [Installation](../getting-started/installation.md) lists what it
lacks.

```python
signature = {
    "x_ptr": sl.pointer(sl.float32),
    "y_ptr": sl.pointer(sl.float32),
    "output_ptr": sl.pointer(sl.float32),
    "n": sl.int32,
}
try:
    add_kernel.emit_mlir(signature=signature, constexprs={"BLOCK": 128})
except sw.CompilationError as error:
    print(error)  # The kernel is outside the language.
except sw.BackendUnavailableError as error:
    print(error)  # The kernel passed; the native bindings are missing.
```

Without the bindings, a kernel outside the language raises the same
`CompilationError` it raises with them. A kernel inside it raises a
`BackendUnavailableError` with the code `native-unavailable`, which says
the check passed and names the
[Installation](../getting-started/installation.md) page. Passing the
check means the kernel can be emitted. It does not mean the kernel can
be launched: launch accepts only the canonical add and multiply kernels.

## Emit without a GPU

With the native wheel, or the bindings of a source build, `emit_mlir()`
turns the captured kernel into a verified live MLIR module. With an
explicit signature it needs neither a GPU nor PyTorch (wheel-only tier):

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

Replacing the final `x + y` with `x * y` selects the other public
operation. The shape and the ABI stay identical. Exactly one floating
operation is admitted; chains, floating vector and scalar arithmetic,
broadcasting, and matrix multiplication are not.

Passing `arguments=` instead infers the same signature from PyTorch
tensor metadata without reading values. Exactly one of the two modes is
required; the full contract lives in [swage](../reference/swage.md).
The printed module preserves source locations, which is what makes the
fail-closed errors precise.

Continue with [Launching Kernels](launching.md) for what happens when
the kernel meets an explicitly selected backend.
