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

    Example:

    ```mlir
    %sid = swage.segment_id 0
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

    Example:

    ```mlir
    %seg = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    ```
  }];
  let arguments = (ins
    MemRefRankOf<[AnyType], [1]>:$values,
    MemRefRankOf<[AnySignlessInteger], [1]>:$offsets,
    Index:$segment_id);
  let results = (outs Swage_SegmentType:$result);
  let assemblyFormat = [{
    $values `,` $offsets `,` $segment_id attr-dict
    `:` type($values) `,` type($offsets) `,` type($segment_id)
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

def Swage_ExtentOp : Swage_Op<"extent", [Pure]> {
  let summary = "Number of elements in a segment";
  let description = [{
    Returns the runtime element count of a segment,
    `offsets[segment_id + 1] - offsets[segment_id]`. May be zero.

    Example:

    ```mlir
    %n = swage.extent %seg : !swage.segment<f32>
    ```
  }];
  let arguments = (ins Swage_SegmentType:$segment);
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

    Example:

    ```mlir
    %exps = swage.map %seg : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%x: f32):
      %e = math.exp %x : f32
      swage.yield %e : f32
    }
    ```
  }];
  let arguments = (ins Swage_SegmentType:$segment,
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
    Every admitted kind is associative and commutative with a known
    identity; that gate is what licenses split reductions later.

    Example:

    ```mlir
    %max = swage.reduce %seg kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%x: f32):
      swage.yield %x : f32
    }
    ```
  }];
  let arguments = (ins Swage_SegmentType:$segment,
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

def Swage_MapStoreOp : Swage_Op<"map_store", [IsolatedFromAbove]> {
  let summary = "Write a per-element region's results to the segment's "
                "range of an output buffer";
  let description = [{
    For every element of the segment, applies the region (captures appended
    to the block arguments, ADR-0008) and writes the yielded value to the
    segment's own index range in `output`; writing outside that range is
    inexpressible. The declared write effect on `output` keeps the
    operation alive under dead-code elimination. The output buffer must not
    alias the segment's values buffer (a runtime obligation). Over an empty
    segment nothing is written.

    Example:

    ```mlir
    swage.map_store %seg, %out : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%x: f32):
      swage.yield %x : f32
    }
    ```
  }];
  let arguments = (ins Swage_SegmentType:$segment,
                       Arg<MemRefRankOf<[AnyInteger, AnyFloat], [1]>,
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
