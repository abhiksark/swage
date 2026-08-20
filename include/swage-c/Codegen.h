// include/swage-c/Codegen.h
//===- Codegen.h - Swage code generation C API -----------------*- C -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_C_CODEGEN_H
#define SWAGE_C_CODEGEN_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*SwageStringCallback)(MlirStringRef value, void *userData);

MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileFixedBlockToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData);

MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData);

#ifdef __cplusplus
}
#endif

#endif // SWAGE_C_CODEGEN_H
