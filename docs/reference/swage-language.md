<!-- docs/reference/swage-language.md -->

# swage.language

`swage.language` exports the markers and symbolic operations of the restricted
kernel language, conventionally imported as `sl`. The symbolic functions are
valid only inside a captured kernel: outside one they raise
`RuntimeError` instead of computing. Their exact accepted source forms
are normative in [Kernel Language](kernel-language.md).

```python
arange
constexpr
float32
float16
float8_e4m3fn
float8_e5m2
int32
load
pointer
program_id
store
```

## Types and markers

```python
sl.float32
sl.float16
sl.float8_e4m3fn
sl.float8_e5m2
sl.int32
sl.pointer(element_type)
sl.constexpr
```

Tensor pointer elements may use `float32`, `float16`, `float8_e4m3fn`, or
`float8_e5m2`. `int32` is the supported runtime scalar parameter type.
`pointer(element_type)` describes a tensor pointer for explicit
`emit_mlir(signature=...)` calls; it requires a supported floating element.
`constexpr` is the exact annotation that marks a compile-time kernel
parameter; annotated parameters are bound through `constexprs` at
emission and launch, never passed at run time.

## sl.program_id

```python
sl.program_id(axis)
```

Return the logical program coordinate inside a compiled kernel.

Parameters
:   `axis`: one nonnegative integer literal that fits signed i32. The
    public launch subset requires axis `0`.

The coordinate is semantic: GPU thread and block IDs never appear in
kernel source. The lowering maps one program instance to one fixed
block of threads.

## sl.arange

```python
sl.arange(start, end)
```

Return a compile-time-sized index vector inside a compiled kernel.

Parameters
:   `start`: only the literal `0` is accepted.
:   `end`: only the compile-time name `BLOCK` is accepted.

Together with `program_id`, `pid * BLOCK + arange(0, BLOCK)` gives each
lane its global element index; the geometry is drawn on
[Kernel Language](kernel-language.md).

## sl.load

```python
sl.load(pointer_value, *, mask=None, other=None)
```

Load a masked vector inside a compiled kernel.

Parameters
:   `pointer_value`: a pointer parameter plus an index-offset vector.
:   `mask`: required by the accepted grammar; lanes where the mask is
    false do not read memory.
:   `other`: required by the accepted grammar; the value produced for
    masked-off lanes.

## sl.store

```python
sl.store(pointer_value, value, *, mask=None)
```

Store a masked vector inside a compiled kernel.

Parameters
:   `pointer_value`: a pointer parameter plus an index-offset vector.
:   `value`: a floating vector with the same element dtype as the pointer.
:   `mask`: required by the accepted grammar; lanes where the mask is
    false write nothing.

`store` appears as an expression statement and is the kernel's only
effect.

Continue with [Kernel Language](kernel-language.md) for the accepted
grammar around these calls, or [swage](swage.md) for the package
surface.
