<!-- docs/user-guide/writing-kernels.md -->

# Writing Kernels

A Swage kernel is ordinary-looking Python that is captured, never
executed. This page explains what each line of the canonical kernel
means and what the frontend does with it. The exact accepted grammar is
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
an unsupported operator, a stray keyword argument, fails at decoration
or emission time with a source-located `CompilationError` naming the
file, line, and column. Nothing partial survives.

## The canonical kernel, line by line

Capture itself needs only the published wheel (wheel-only tier):

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

- The parameters are the kernel's ABI, in order: three f32 pointers, an
  i32 count, and the compile-time block width. `BLOCK` is marked with
  the exact annotation `sl.constexpr`, so it is bound at compile time
  and never passed at launch.
- `sl.program_id(0)` is the logical block coordinate. It is a semantic
  index, not a GPU thread ID; the lowering decides how it maps to
  hardware.
- `sl.arange(0, BLOCK)` spreads one block into `BLOCK` lanes, so
  `pid * BLOCK + arange(0, BLOCK)` is each lane's global element index.
- The mask compares those indices against `n` once and guards both
  loads and the store, so the tail block reads the `other` value and
  writes nothing out of bounds. The launch geometry this implies is
  drawn on [Kernel Language](../reference/kernel-language.md).

## Emit without a GPU

With the native build present, `emit_mlir()` turns the captured kernel
into a verified live MLIR module. With an explicit signature it needs
neither a GPU nor PyTorch (native-build tier):

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

Passing `arguments=` instead infers the same signature from PyTorch
tensor metadata without reading values. Exactly one of the two modes is
required; the full contract lives in [swage](../reference/swage.md).
The printed module preserves source locations, which is what makes the
fail-closed errors precise.

Continue with [Launching Kernels](launching.md) for what happens when
the kernel meets a GPU.
