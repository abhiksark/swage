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

    A function that holds a segment program declares what each of its
    arguments is with the argument attribute `swage.role`:

    ```mlir
    func.func @segmented_sum(
        %values: memref<?xf32> {swage.role = #swage.role<values>},
        %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
        %output: memref<?xf32> {swage.role = #swage.role<output>},
        %value_count: i32 {swage.role = #swage.role<value_count>},
        %segment_count: i32 {swage.role = #swage.role<segment_count>})
    ```

    - `values` and `offsets` are the buffers `swage.make_segment` binds.
    - `output` is the buffer the program writes its result to.
    - `value_count` is the element count that bounds every range into
      `values`. For a program that ends in `swage.map_store` it also bounds
      the store.
    - `segment_count` is the extent of `swage.segment_id 0` and the bound of
      every segment index a lowering loads.

    The dialect verifies that a role sits on an argument of a fitting type (a
    rank-one memref for a buffer, with signless integer elements for
    `offsets`, and a signless integer for a count) and that no role is
    declared twice. Which roles a function must declare, and with which
    element and count types, is decided by the lowering that admits it.
  }];
  let cppNamespace = "::mlir::swage";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
  let hasRegionArgAttrVerify = 1;
  let extraClassDeclaration = [{
    /// The name of the argument attribute that declares a role.
    static ::llvm::StringRef getRoleAttrName() { return "swage.role"; }

    void registerAttributes();
    void registerTypes();
  }];
}

class Swage_Op<string mnemonic, list<Trait> traits = []>
    : Op<Swage_Dialect, mnemonic, traits>;

#endif // SWAGE_DIALECT_SWAGE_IR_SWAGEDIALECT
