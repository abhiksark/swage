//===- Dialects.cpp - C API for the swage dialect -------------------------===//
//
// Defines the CAPI registration handle for the swage dialect.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Swage, swage,
                                      mlir::swage::SwageDialect)
