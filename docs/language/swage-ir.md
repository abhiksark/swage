<!-- docs/language/swage-ir.md -->

# Swage Textual IR

Textual MLIR is useful for native tests, debugging, and minimal reproducers.
The Python JIT constructs MLIR directly through native bindings; it does not
use textual IR as an intermediate representation. See the
[Compiler Pipeline](../internals/compiler-pipeline.md) for that frontend boundary.

This page describes the syntax and verifier contracts of the
[Swage Dialect](../internals/swage-dialect.md). Dialect parsing and verification
accept more shapes than the currently admitted public and private lowerings.
A valid module here is not a promise of execution support; the
[Compiler Pipeline](../internals/compiler-pipeline.md) describes those boundaries.

## The segment type

`!swage.segment<T>` is a symbolic handle for one runtime-sized, internally
dense segment. `T` must be an integer or floating-point type, such as `i32`,
`f16`, `bf16`, or `f32`.

The type carries only the element type: no length, values buffer, offsets
array, or segment identity. The SSA value produced by `swage.make_segment`
carries those runtime relationships. A segment is never a runtime-sized
register array, and constructing its handle does not materialize its data.

## A complete module

The following module starts with the coordinate, segment construction, and
extent forms in
[`test/Dialect/Swage/roundtrip.mlir`](https://github.com/abhiksark/swage/blob/main/test/Dialect/Swage/roundtrip.mlir),
then adds a map, reduction, and terminal store. It uses every current Swage
operation. The output buffer must not alias the values buffer.

```mlir
module {
  func.func @fixed_program_coordinate() -> index {
    %pid = swage.program_id 0
    return %pid : index
  }

  func.func @segment_walkthrough(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %scale: f32) -> index {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %length = swage.extent %segment : !swage.segment<f32>
    %scaled = swage.map %segment captures(%scale : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32, %factor: f32):
      %product = arith.mulf %element, %factor : f32
      swage.yield %product : f32
    }
    %total = swage.reduce %scaled kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %scaled, %output captures(%total : f32)
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%element: f32, %sum: f32):
      %result = arith.addf %element, %sum : f32
      swage.yield %result : f32
    }
    return %length : index
  }
}
```

Read the SSA data flow in source order:

1. `%pid` is a logical fixed-block coordinate in the first function, not a
   physical GPU block or thread identifier.
2. In the second function, `%sid` selects the slice
   `values[offsets[sid] : offsets[sid + 1]]`. The explicit values, offsets,
   and ID operands connect `%segment` to that runtime data.
3. `%length` is the runtime element count. It remains an `index` SSA value,
   not a parameter of `!swage.segment<f32>`.
4. `%scaled` is another symbolic segment. Its region receives one element
   and the explicit `%scale` capture as `%factor`, then yields their product.
5. `%total` is a scalar sum of the scaled elements. The reduction region
   yields each element unchanged; `kind<sum>` supplies the combining rule.
6. `swage.map_store` captures `%total` and writes each scaled element plus
   that total into the segment's corresponding output range. This is the
   terminal write, not a new segment result. The function returns `%length`.

`arith.mulf` and `arith.addf` are upstream MLIR operations. Swage supplies
segment semantics rather than duplicating ordinary scalar arithmetic.

## Argument roles

A segment function says what each of its arguments is with the argument
attribute `swage.role`:

```mlir
func.func @segmented_sum(
    %values: memref<?xf32> {swage.role = #swage.role<values>},
    %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
    %output: memref<?xf32> {swage.role = #swage.role<output>},
    %value_count: i32 {swage.role = #swage.role<value_count>},
    %segment_count: i32 {swage.role = #swage.role<segment_count>})
```

| Role | Argument |
|---|---|
| `values` | the buffer that `swage.make_segment` views |
| `offsets` | the buffer of segment bounds that `swage.make_segment` reads |
| `output` | the buffer the terminal writes |
| `value_count` | the element count of `values`, which bounds every range into it |
| `segment_count` | the extent of `swage.segment_id 0` |

The roles name the arguments, so their order carries no meaning. The
verifier accepts a role on a rank-one memref (`values`, `output`), on a
rank-one memref of signless integers (`offsets`), or on a signless integer
(a count), and rejects a role that two arguments of one function declare. It
does not require a function to declare roles: the walkthrough module above
declares none and verifies.

The segmented lowerings require more. Every argument of a function they
lower declares a role, each of the five roles appears once, and the types
are the admitted ones: f32 `values` and `output`, i32 `offsets`, and counts
of the offsets' element type, in dynamically sized buffers with the
identity layout in the default memory space. A function that falls short is
rejected with a diagnostic that names the argument or the missing role.
There is no positional default.

## Operations

These forms use concrete element types. Other integer or floating-point
types are accepted where the operation's type constraints allow them; this
does not imply that every such type has an admitted lowering.

### `swage.program_id`

```mlir
%pid = swage.program_id 0
```

The form is `%pid = swage.program_id <axis>`, where `<axis>` is a
nonnegative `i32` attribute, not an SSA operand. The result is `index`.
It identifies the current logical fixed-block program along that axis,
never a physical GPU block or thread.

### `swage.segment_id`

```mlir
%sid = swage.segment_id 0
```

The form is `%sid = swage.segment_id <axis>`. Like `program_id`, its axis
is a nonnegative `i32` attribute and its result is `index`. It identifies
the logical segment on the given logical grid axis. Dialect acceptance
does not guarantee a lowering for every axis.

### `swage.make_segment`

```mlir
%segment = swage.make_segment %values, %offsets, %sid
    : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
```

The operands are values, offsets, and segment ID, in that order. Their
types are explicit, followed by `-> !swage.segment<T>`:

- Both buffers must be rank-one memrefs.
- Offsets must have a signless integer element type.
- The segment ID must have type `index`.
- The values buffer's element type must equal the result segment's
  element type.

The handle denotes `values[offsets[id] : offsets[id + 1]]`. This operation
binds those runtime relationships without materializing or copying data.

### `swage.extent`

```mlir
%length = swage.extent %segment : !swage.segment<f32>
```

The general form is `%length = swage.extent %segment : !swage.segment<T>`.
Its `index` result is the runtime length
`offsets[id + 1] - offsets[id]`, which may be zero.

### `swage.map`

```mlir
%mapped = swage.map %segment captures(%scale, %bias : f32, f64)
    : !swage.segment<f32> -> !swage.segment<f64> {
^bb0(%element: f32, %factor: f32, %shift: f64):
  %product = arith.mulf %element, %factor : f32
  %wide = arith.extf %product : f32 to f64
  %result = arith.addf %wide, %shift : f64
  swage.yield %result : f64
}
```

`map` applies its region to each input element and produces another
symbolic segment. The optional `captures(values : types)` clause appears
before the segment and result types; omit the clause when there are no
captures.

The single-block region is isolated from above: it cannot refer directly
to outer SSA values. Its block arguments must be the input element,
followed by one argument per capture in operand order, with matching
types. Captures are integer or floating-point scalars, not segment or
buffer handles.

The region must terminate with `swage.yield`. The yielded type must equal
the result segment's element type, so a map may change element type, as
the `f32` to `f64` example does. An empty input maps to an empty segment.

### `swage.reduce`

```mlir
%total = swage.reduce %segment captures(%bias : f32) kind<sum>
    : !swage.segment<f32> -> f32 {
^bb0(%element: f32, %shift: f32):
  %adjusted = arith.addf %element, %shift : f32
  swage.yield %adjusted : f32
}
```

The optional capture clause follows the segment operand, before the
required `kind<sum>`, `kind<max>`, or `kind<min>`. The single-block,
isolated region receives the input element followed by explicit scalar
captures in operand order, with matching types, just as for `map`.

The region is a **per-element transform**, not a generic two-argument
combiner or an accumulator loop. The kind combines its yielded values.
`swage.yield` must terminate the region, and its type must equal the
integer or floating-point scalar result type.

For an empty segment, the result is the kind's identity:

- `sum`: zero.
- `max`: negative infinity for floating-point values, or the minimum
  integer for the integer type.
- `min`: positive infinity for floating-point values, or the maximum
  integer for the integer type.

Every element contributes exactly once, and the kind does not fix the order
in which the yielded values are combined. A lowering may fold them left to
right, reduce them as a tree, or reduce parts of the segment separately and
merge the partial results. What stays guaranteed depends on the kind and the
element type:

- `max` and `min`: the result does not depend on the order. A
  floating-point `max` is NaN when any element is NaN.
- Floating-point `sum`: the result equals the exact sum only up to rounding,
  and the rounding depends on the order that the lowering and the schedule
  choose. Bitwise equality between two lowerings, or between two schedules
  of one program, is not guaranteed. The difference is rounding error: it
  is bounded relative to the sum of the absolute values of the elements, and
  it can be large relative to the result when the elements cancel.
- Integer `sum`: the overflow behavior is not defined, and no lowering
  admits an integer element type.

Lowerings add no fast-math flags. The current lowerings admit `sum` and
`max` over `f32` and reject `min`.

### `swage.map_store`

```mlir
swage.map_store %segment, %output captures(%total : f32)
    : !swage.segment<f32>, memref<?xf32> {
^bb0(%element: f32, %sum: f32):
  %result = arith.addf %element, %sum : f32
  swage.yield %result : f32
}
```

The operands are the segment and a rank-one output memref, followed by
optional `captures(values : types)`. The single-block region is isolated
and receives the input element followed by captures in operand order,
with matching types. Captures are integer or floating-point scalars.
The region must terminate with `swage.yield`, whose type must match the
output buffer's integer or floating-point element type.

This terminal operation has no SSA result. It writes the per-element
results only to the segment's corresponding output range,
`output[offsets[id] : offsets[id + 1]]`. An empty segment writes nothing.
The explicit write effect on the output keeps the operation alive under
dead-code elimination.

The output buffer must not alias the segment's values buffer. This is a
runtime obligation, not something dialect type verification proves.

### `swage.yield`

```mlir
swage.yield %result : f32
```

`yield` takes exactly one integer or floating-point value with an explicit
type. It is valid only as the terminator of a `swage.map`, `swage.reduce`,
or `swage.map_store` region. The parent operation checks that the yielded
type matches its result segment element, scalar result, or output buffer
element, respectively.

## Design rules

- **Runtime identity stays in SSA values, not types.**
  `!swage.segment<T>` describes elements, while `swage.make_segment`
  connects a handle to values, offsets, and an ID. Runtime length never
  becomes a runtime-sized register array.
- **Semantic coordinates are logical.** Physical GPU thread and block
  IDs do not belong in semantic Swage IR.
- **Regions are isolated and captures are explicit.** Integer or
  floating-point scalar captures follow the element block argument in
  operand order; argument and yield types must agree with the parent
  operation's contract.
- **Reuse upstream dialects.** Ordinary arithmetic, memory operations,
  and backend work belong to upstream MLIR dialects, not duplicate Swage
  operations.
- **The terminal store owns the write.** `swage.map_store` is the only
  Swage operation that declares a write. `map` and `reduce` track
  nested operations' memory effects recursively.
- **Segment consumers declare their read.** `swage.extent`, `swage.map`,
  `swage.reduce`, and `swage.map_store` read the values and offsets
  buffers behind the segment handle and declare a read on their segment
  operand. Common subexpression elimination therefore does not merge two
  of them across a write. A read does not keep an operation alive: an
  unused `extent`, and an unused `map` or `reduce` with a pure region,
  is still removable. `swage.make_segment` and `swage.segment_id` read
  no memory and stay effect-free.

Continue with [Segmented Reductions](../internals/segmented-reductions.md)
for the subset of these shapes that a lowering admits today. For exhaustive
operand and trait tables, go back to the generated
[Swage Dialect reference](../internals/swage-dialect.md). For `swage-opt`
and the registered pass surface, use
[Compiler Tools and Passes](../internals/compiler-tools.md).
