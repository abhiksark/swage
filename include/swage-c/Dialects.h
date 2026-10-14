// include/swage-c/Dialects.h
//===- Dialects.h - C API for the Swage dialects ------------------*- C -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// Declares the registration handles of the swage and swage_plan dialects and
// the constructor and accessors of the `!swage.segment<T>` type.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_C_DIALECTS_H
#define SWAGE_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// `mlirGetDialectHandle__swage__()`: the semantic dialect.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Swage, swage);

/// `mlirGetDialectHandle__swage_plan__()`: the planning dialect. A caller
/// needs it only to parse or build planning IR; the functions of Codegen.h
/// load the dialect themselves.
MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(SwagePlan, swage_plan);

/// Returns true if `type` is a `!swage.segment<T>`.
MLIR_CAPI_EXPORTED bool swageTypeIsASegment(MlirType type);

/// Returns the type ID of `!swage.segment<T>`, the same for every `T`.
MLIR_CAPI_EXPORTED MlirTypeID swageSegmentTypeGetTypeID(void);

/// Returns `!swage.segment<elementType>` in the context of `elementType`.
///
/// The swage dialect must be loaded in that context and `elementType` must be
/// an integer or float type. When either requirement fails, the function
/// emits a diagnostic on the context and returns a null type.
MLIR_CAPI_EXPORTED MlirType swageSegmentTypeGet(MlirType elementType);

/// Returns the element type `T` of a `!swage.segment<T>`. `type` must
/// satisfy `swageTypeIsASegment`.
MLIR_CAPI_EXPORTED MlirType swageSegmentTypeGetElementType(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // SWAGE_C_DIALECTS_H
