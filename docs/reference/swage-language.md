<!-- docs/reference/swage-language.md -->

# swage.language

`swage.language` exports the eight symbols of the restricted kernel
language, conventionally imported as `sl`. The frontend recognizes the
module by object, so another import name works as long as the kernel uses
it for the marker and for every call. The symbolic functions are valid only
inside a captured kernel: outside one they raise `RuntimeError` instead of
computing. Their accepted source forms are normative in
[Kernel Language](kernel-language.md).

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

## Types and markers

```python
sl.float32
sl.int32
sl.pointer(element_type)
sl.constexpr
```

`float32` and `int32` are the scalar types accepted by the current
frontend. `pointer(element_type)` describes a pointer to a scalar
element type for explicit `emit_mlir(signature=...)` calls.
`constexpr` is the annotation that marks a compile-time kernel parameter.
It is written as an attribute of a name bound to this module, such as
`sl.constexpr`, and it is the only parameter annotation the frontend
accepts. Annotated parameters are bound through `constexprs` at emission
and launch, never passed at run time.

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
sl.load(pointer_value, *, mask, other)
```

Load a masked vector inside a compiled kernel.

Parameters
:   `pointer_value`: a pointer parameter plus an index-offset vector.
:   `mask`: required keyword with no default; lanes where the mask is
    false do not read memory.
:   `other`: required keyword with no default; the value produced for
    masked-off lanes. It is an integer or float literal that float32
    represents, with an optional leading minus sign.

## sl.store

```python
sl.store(pointer_value, value, *, mask)
```

Store a masked vector inside a compiled kernel.

Parameters
:   `pointer_value`: a pointer parameter plus an index-offset vector.
:   `value`: the f32 vector to store.
:   `mask`: required keyword with no default; lanes where the mask is
    false write nothing.

`store` appears as an expression statement and is the kernel's only
effect.

Continue with [Kernel Language](kernel-language.md) for the accepted
grammar around these calls, or [swage](swage.md) for the package
surface.
