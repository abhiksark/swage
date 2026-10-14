// include/swage/Dialect/Swage/IR/SwageOps.td
//===- SwageOps.td - Swage dialect operations --------------*- tablegen -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGE_IR_SWAGEOPS
#define SWAGE_DIALECT_SWAGE_IR_SWAGEOPS

include "swage/Dialect/Swage/IR/SwageTypes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Swage_ArgumentRoleValues : I32EnumAttrCase<"Values", 0, "values">;
def Swage_ArgumentRoleOffsets : I32EnumAttrCase<"Offsets", 1, "offsets">;
def Swage_ArgumentRoleOutput : I32EnumAttrCase<"Output", 2, "output">;
def Swage_ArgumentRoleValueCount
    : I32EnumAttrCase<"ValueCount", 3, "value_count">;
def Swage_ArgumentRoleSegmentCount
    : I32EnumAttrCase<"SegmentCount", 4, "segment_count">;
def Swage_ArgumentRoleFeatureCount
    : I32EnumAttrCase<"FeatureCount", 5, "feature_count">;

def Swage_ArgumentRole
    : I32EnumAttr<"ArgumentRole",
                  "what one argument of a segment function is",
                  [Swage_ArgumentRoleValues, Swage_ArgumentRoleOffsets,
                   Swage_ArgumentRoleOutput, Swage_ArgumentRoleValueCount,
                   Swage_ArgumentRoleSegmentCount,
                   Swage_ArgumentRoleFeatureCount]> {
  let cppNamespace = "::mlir::swage";
  let genSpecializedAttr = 0;
}

// The value of the `swage.role` argument attribute. The dialect description
// states the contract of each role.
def Swage_ArgumentRoleAttr
    : EnumAttr<Swage_Dialect, Swage_ArgumentRole, "role"> {
  let assemblyFormat = "`<` $value `>`";
}

def Swage_ProgramIdOp : Swage_Op<"program_id", [Pure]> {
  let summary = "Id of the current logical fixed-block program along an axis";
  let description = [{
    Returns the coordinate of the current logical fixed-block program along
    the given axis. This is a logical program coordinate, not a GPU thread or
    block id.

    Example:

    ```mlir
    %pid = swage.program_id 0
    ```
  }];
  let arguments = (ins ConfinedAttr<I32Attr, [IntNonNegative]>:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = "$axis attr-dict";
}

def Swage_SegmentIdOp : Swage_Op<"segment_id", [Pure]> {
  let summary = "Id of the current logical segment along an axis";
  let description = [{
    Returns the index of the logical segment the surrounding program instance
    operates on, along the given logical grid axis. This is a semantic
    coordinate in the logical grid, not a GPU thread or block id.

    Axis 0 runs over the segments of the offsets array. Axis 1 runs over the
    trailing dimension of rank-two values: a program instance is then one
    segment and one column, and `swage.segment_id 1` is the column.

    Example:

    ```mlir
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    ```
  }];
  let arguments = (ins ConfinedAttr<I32Attr, [IntNonNegative]>:$axis);
  let results = (outs Index:$result);
  let assemblyFormat = "$axis attr-dict";
}

def Swage_MakeSegmentOp : Swage_Op<"make_segment", [Pure]> {
  let summary = "Bind one logical segment of a values buffer";
  let description = [{
    Binds segment `segment_id` of the dense `values` buffer, as delimited by
    the `offsets` array, into a `!swage.segment` value:

        segment = values[offsets[segment_id] : offsets[segment_id + 1]]

    The result is a symbolic handle. No data is materialized; consumers of
    the segment are tiled during lowering.

    With `column`, `values` is a rank-two buffer of rows and the segment is
    one column of the rows the offsets delimit:

        segment = values[offsets[segment_id] : offsets[segment_id + 1], column]

    The segment is still a run of scalars, so its type carries the element
    type alone: neither the number of columns nor the column enters a type.
    A rank-two buffer needs a column, and a rank-one buffer takes none.

    Example:

    ```mlir
    %seg = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %column = swage.make_segment %rows, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    ```
  }];
  let arguments = (ins
    MemRefRankOf<[AnyType], [1, 2]>:$values,
    MemRefRankOf<[AnySignlessInteger], [1]>:$offsets,
    Index:$segment_id,
    Optional<Index>:$column);
  let results = (outs Swage_SegmentType:$result);
  let assemblyFormat = [{
    $values `,` $offsets `,` $segment_id (`column` `(` $column^ `)`)? attr-dict
    `:` type($values) `,` type($offsets) `,` type($segment_id)
    (`,` type($column)^)? `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Swage_ExtentOp : Swage_Op<"extent"> {
  let summary = "Number of elements in a segment";
  let description = [{
    Returns the runtime element count of a segment,
    `offsets[segment_id + 1] - offsets[segment_id]`. May be zero.

    The operation reads the offsets buffer behind the segment handle and
    declares that read on its operand. Two instances are not merged across
    a write, and an unused instance is still removable.

    The lowerings admit the extent in one place: the scalar epilogue of a
    mean, where the extent is cast to the count type, converted to the
    element type, and divides the result of a `sum` reduction of the same
    segment. The division runs once per segment, after the reduction, so a
    mean is a composition and not a reduction kind: a kind has an identity
    and an order-free combine, which a mean lacks. A lowering that splits a
    segment sums the partial results and divides once.

    Example:

    ```mlir
    %n = swage.extent %seg : !swage.segment<f32>
    %count = arith.index_cast %n : index to i32
    %divisor = arith.sitofp %count : i32 to f32
    %mean = arith.divf %sum, %divisor : f32
    ```
  }];
  let arguments = (ins
    Arg<Swage_SegmentType, "segment handle", [MemRead]>:$segment);
  let results = (outs Index:$result);
  let assemblyFormat = "$segment attr-dict `:` qualified(type($segment))";
}

def Swage_ScalarType : AnyTypeOf<[AnyInteger, AnyFloat]>;

def Swage_ReductionKindSum : I32EnumAttrCase<"Sum", 0, "sum">;
def Swage_ReductionKindMax : I32EnumAttrCase<"Max", 1, "max">;
def Swage_ReductionKindMin : I32EnumAttrCase<"Min", 2, "min">;

def Swage_ReductionKind
    : I32EnumAttr<"ReductionKind",
                  "combining function of a swage.reduce",
                  [Swage_ReductionKindSum, Swage_ReductionKindMax,
                   Swage_ReductionKindMin]> {
  let cppNamespace = "::mlir::swage";
  let genSpecializedAttr = 0;
}

def Swage_ReductionKindAttr
    : EnumAttr<Swage_Dialect, Swage_ReductionKind, "reduction_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

def Swage_MapOp : Swage_Op<"map", [IsolatedFromAbove,
                                   RecursiveMemoryEffects]> {
  let summary = "Apply a per-element region across a segment";
  let description = [{
    Produces a new symbolic segment by applying the region to every element
    of the operand segment. The region is isolated from above: outer SSA
    values enter only through `captures`, whose values append to the entry
    block arguments after the element argument, in order. The yield type
    defines the result segment's element type. Over an empty segment the
    result is an empty segment (ADR-0008).

    The result is a lazy view of the operand segment. The map computes no
    element where it appears: each consumer of the result applies the region
    to the elements it reads, when it reads them. A consumer therefore sees
    a write to the buffers that happens between the map and the consumer,
    and fusing a map into its consumer does not change a program.

    The operation declares a read of the values and offsets buffers behind
    the segment handle on its segment operand, in addition to the effects of
    its region, so two instances are not merged across a write.

    Example:

    ```mlir
    %exps = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%x: f32):
      %e = math.exp %x : f32
      swage.yield %e : f32
    }
    ```
  }];
  let arguments = (ins Arg<Swage_SegmentType, "segment handle",
                           [MemRead]>:$segment,
                       Variadic<Swage_ScalarType>:$captures);
  let results = (outs Swage_SegmentType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $segment (`captures` `(` $captures^ `:` type($captures) `)`)?
    attr-dict `:` qualified(type($segment)) `->` qualified(type($result))
    $body
  }];
  let hasRegionVerifier = 1;
}

def Swage_ReduceOp : Swage_Op<"reduce", [IsolatedFromAbove,
                                         RecursiveMemoryEffects]> {
  let summary = "Reduce a segment to a scalar with a combining kind";
  let description = [{
    Applies the region to every element (with captures appended to the
    block arguments, ADR-0008) and combines the yielded values with `kind`.
    The result type equals the yield type. Over an empty segment the result
    is the identity of `kind`: `sum` -> 0, `max` -> negative infinity or the
    minimum integer, `min` -> positive infinity or the maximum integer.

    Every element contributes exactly once. The order in which the yielded
    values are combined is not part of the operation: a lowering may fold
    them left to right, reduce them as a tree, or reduce parts of the
    segment separately and merge the partial results. What stays guaranteed
    depends on the kind and the element type:

    - `max` and `min` are associative and commutative, so the result does
      not depend on the order. A floating-point `max` is NaN when any
      element is NaN.
    - Floating-point `sum` is commutative but not associative. The result
      equals the exact sum of the elements only up to rounding, and the
      rounding depends on the order that the lowering and the schedule
      choose. Two lowerings, or two schedules of one program, may return
      different sums, and bitwise equality between them is not guaranteed.
      The difference is rounding error: it is bounded relative to the sum of
      the absolute values of the elements, and it can be large relative to
      the result when the elements cancel.
    - Integer element types pass the verifier, and no lowering admits them.
      The overflow behavior of an integer `sum` is not defined.

    Swage lowerings add no fast-math flags, so a later pass has no license
    to reorder a sum once a lowering has fixed its order.

    The operation reads the values and offsets buffers behind the segment
    handle. It declares that read on its segment operand in addition to the
    effects of its region, so two instances are not merged across a write.

    Example:

    ```mlir
    %max = swage.reduce %seg kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%x: f32):
      swage.yield %x : f32
    }
    ```
  }];
  let arguments = (ins Arg<Swage_SegmentType, "segment handle",
                           [MemRead]>:$segment,
                       Variadic<Swage_ScalarType>:$captures,
                       Swage_ReductionKindAttr:$kind);
  let results = (outs Swage_ScalarType:$result);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $segment (`captures` `(` $captures^ `:` type($captures) `)`)?
    `kind` `` $kind attr-dict `:` qualified(type($segment))
    `->` type($result) $body
  }];
  let hasRegionVerifier = 1;
}

def Swage_MapStoreOp : Swage_Op<"map_store", [IsolatedFromAbove,
                                               RecursiveMemoryEffects]> {
  let summary = "Write a per-element region's results to the segment's "
                "range of an output buffer";
  let description = [{
    For every element of the segment, applies the region (captures appended
    to the block arguments, ADR-0008) and writes the yielded value to the
    segment's own index range in `output`; writing outside that range is
    inexpressible. For a column segment of rank-two values, `output` is a
    rank-two buffer and the range is the rows of the segment in its column. The declared write effect on `output` keeps the
    operation alive under dead-code elimination. The output buffer must not
    alias the segment's values buffer (a runtime obligation). Over an empty
    segment nothing is written.

    The operation also reads the values and offsets buffers behind the
    segment handle and declares that read on its segment operand. Its
    effects include the effects of its region, as for `swage.map` and
    `swage.reduce`, because a fused map runs inside that region.

    Example:

    ```mlir
    swage.map_store %seg, %out : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%x: f32):
      swage.yield %x : f32
    }
    ```
  }];
  let arguments = (ins Arg<Swage_SegmentType, "segment handle",
                           [MemRead]>:$segment,
                       Arg<MemRefRankOf<[AnyInteger, AnyFloat], [1, 2]>,
                           "output buffer", [MemWrite]>:$output,
                       Variadic<Swage_ScalarType>:$captures);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    $segment `,` $output
    (`captures` `(` $captures^ `:` type($captures) `)`)?
    attr-dict `:` qualified(type($segment)) `,` type($output) $body
  }];
  let hasRegionVerifier = 1;
}

def Swage_YieldOp : Swage_Op<"yield", [Pure, Terminator,
                                       ParentOneOf<["MapOp", "ReduceOp",
                                                    "MapStoreOp"]>]> {
  let summary = "Yield the per-element result of a Swage region";
  let description = [{
    Terminates the region of a Swage region-based operation, producing the
    per-element result value.
  }];
  let arguments = (ins Swage_ScalarType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // SWAGE_DIALECT_SWAGE_IR_SWAGEOPS
