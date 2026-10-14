// lib/CAPI/Dialects.cpp
//===- Dialects.cpp - C API for the Swage dialects ------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Dialects.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageTypes.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"

using namespace mlir;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Swage, swage, swage::SwageDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(SwagePlan, swage_plan,
                                      swage_plan::SwagePlanDialect)

bool swageTypeIsASegment(MlirType type) {
  return isa<swage::SegmentType>(unwrap(type));
}

MlirTypeID swageSegmentTypeGetTypeID(void) {
  return wrap(swage::SegmentType::getTypeID());
}

MlirType swageSegmentTypeGet(MlirType elementType) {
  Type element = unwrap(elementType);
  MLIRContext *context = element.getContext();
  auto emit = [context] { return emitError(UnknownLoc::get(context)); };
  // Uniquing a type of a dialect that is not loaded is a fatal error in
  // MLIR, so the missing dialect is reported here instead.
  if (!context->getLoadedDialect<swage::SwageDialect>()) {
    emit() << "cannot build !swage.segment<" << element
           << ">: the swage dialect is not loaded in this context";
    return wrap(Type());
  }
  return wrap(swage::SegmentType::getChecked(emit, context, element));
}

MlirType swageSegmentTypeGetElementType(MlirType type) {
  return wrap(cast<swage::SegmentType>(unwrap(type)).getElementType());
}
