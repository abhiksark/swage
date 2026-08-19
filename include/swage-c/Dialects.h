//===- Dialects.h - C API for the swage dialect -------------------*- C -*-===//
//
// Declares the CAPI registration handle for the swage dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_C_DIALECTS_H
#define SWAGE_C_DIALECTS_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Swage, swage);

#ifdef __cplusplus
}
#endif

#endif // SWAGE_C_DIALECTS_H
