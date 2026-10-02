//===- SwageDialect.cpp - Swage dialect -------------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/Swage/IR/SwageTypes.h"

using namespace mlir;
using namespace mlir::swage;

#include "swage/Dialect/Swage/IR/SwageOpsDialect.cpp.inc"

void SwageDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "swage/Dialect/Swage/IR/SwageOps.cpp.inc"
      >();
  registerAttributes();
  registerTypes();
}

/// Whether `type` can carry `role`, and what the role requires if not.
static const char *requirementOf(ArgumentRole role, Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  switch (role) {
  case ArgumentRole::Values:
  case ArgumentRole::Output:
    return memref && memref.getRank() == 1 ? nullptr : "a rank-one memref";
  case ArgumentRole::Offsets:
    return memref && memref.getRank() == 1 &&
                   memref.getElementType().isSignlessInteger()
               ? nullptr
               : "a rank-one memref of signless integers";
  case ArgumentRole::ValueCount:
  case ArgumentRole::SegmentCount:
    return type.isSignlessInteger() ? nullptr : "a signless integer";
  }
  llvm_unreachable("unknown argument role");
}

LogicalResult SwageDialect::verifyRegionArgAttribute(Operation *op,
                                                     unsigned regionIndex,
                                                     unsigned argIndex,
                                                     NamedAttribute attribute) {
  if (attribute.getName() != getRoleAttrName())
    return op->emitError()
           << "argument #" << argIndex << " carries '"
           << attribute.getName().strref()
           << "', which is not an argument attribute of the swage dialect; "
              "the dialect defines "
           << getRoleAttrName();
  auto role = dyn_cast<ArgumentRoleAttr>(attribute.getValue());
  if (!role)
    return op->emitError() << getRoleAttrName()
                           << " must be a #swage.role attribute, got "
                           << attribute.getValue();
  auto function = dyn_cast<FunctionOpInterface>(op);
  if (!function)
    return op->emitError() << getRoleAttrName()
                           << " belongs on an argument of a function";

  Type type = function.getArgumentTypes()[argIndex];
  StringRef name = stringifyArgumentRole(role.getValue());
  if (const char *requirement = requirementOf(role.getValue(), type))
    return op->emitError() << getRoleAttrName() << "<" << name << "> requires "
                           << requirement << ", got " << type;
  // Reported once, at the later of two arguments that declare one role.
  for (unsigned earlier = 0; earlier < argIndex; ++earlier)
    if (function.getArgAttr(earlier, getRoleAttrName()) == role)
      return op->emitError() << getRoleAttrName() << "<" << name
                             << "> is declared by argument #" << earlier
                             << " and again by argument #" << argIndex;
  return success();
}
