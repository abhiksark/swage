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
include "mlir/Interfaces/SideEffectInterfaces.td"

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

#endif // SWAGE_DIALECT_SWAGE_IR_SWAGEOPS
