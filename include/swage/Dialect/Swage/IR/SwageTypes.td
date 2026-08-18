//===- SwageTypes.td - Swage dialect types -----------------*- tablegen -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGE_IR_SWAGETYPES
#define SWAGE_DIALECT_SWAGE_IR_SWAGETYPES

include "mlir/IR/AttrTypeBase.td"
include "swage/Dialect/Swage/IR/SwageDialect.td"

class Swage_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Swage_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Swage_SegmentType : Swage_Type<"Segment", "segment"> {
  let summary = "A runtime-sized, internally dense segment of values";
  let description = [{
    `!swage.segment<T>` is the semantic handle for one logical segment with
    element type `T`. The type carries only the element type. The runtime
    identity of a segment (which values buffer, which offsets array, which
    segment index) is carried by the SSA value produced by
    `swage.make_segment`, never by the type itself.
  }];
  let parameters = (ins "::mlir::Type":$elementType);
  let assemblyFormat = "`<` $elementType `>`";
  let genVerifyDecl = 1;
}

#endif // SWAGE_DIALECT_SWAGE_IR_SWAGETYPES
