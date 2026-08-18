//===- SwageDialect.td - Swage dialect definition ----------*- tablegen -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGE_IR_SWAGEDIALECT
#define SWAGE_DIALECT_SWAGE_IR_SWAGEDIALECT

include "mlir/IR/OpBase.td"

def Swage_Dialect : Dialect {
  let name = "swage";
  let summary = "Segment-level semantic operations for the Swage compiler.";
  let description = [{
    The `swage` dialect models computations over variable-sized, internally
    dense segments. A segment is a logical, runtime-sized slice of a dense
    values buffer described by an offsets array:

        segment i = values[offsets[i] : offsets[i + 1]]

    Operations in this dialect describe what happens to one logical segment.
    They deliberately contain no GPU thread or block structure: how a segment
    becomes fixed-size tile tasks is decided by later lowering stages, not by
    this dialect. Runtime segment identity is carried by SSA values and
    operands, never by types.
  }];
  let cppNamespace = "::mlir::swage";

  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

class Swage_Op<string mnemonic, list<Trait> traits = []>
    : Op<Swage_Dialect, mnemonic, traits>;

#endif // SWAGE_DIALECT_SWAGE_IR_SWAGEDIALECT
