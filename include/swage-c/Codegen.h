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

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*SwageStringCallback)(MlirStringRef value, void *userData);
typedef void (*SwageTaskIdsCallback)(const int32_t *taskIds, intptr_t taskCount,
                                     void *userData);
typedef struct {
  void *ptr;
} SwageHostExecutable;

MLIR_CAPI_EXPORTED SwageHostExecutable swageCompileFixedBlockToHost(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback contractCallback, void *contractUserData);
MLIR_CAPI_EXPORTED bool
swageHostExecutableIsNull(SwageHostExecutable executable);
MLIR_CAPI_EXPORTED MlirLogicalResult swageHostExecutableInvoke(
    SwageHostExecutable executable, void **arguments, intptr_t argumentCount,
    SwageStringCallback errorCallback, void *errorUserData);
MLIR_CAPI_EXPORTED void
swageHostExecutableDestroy(SwageHostExecutable executable);

MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileFixedBlockToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData);

MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, bool useTaskIds, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData);

MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileFusedSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData);

MLIR_CAPI_EXPORTED MlirLogicalResult
swageCompilePersistentSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData);

MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileSplitPartialReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData);

MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileSplitMergeReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData);

// Callback counts below are flat i32 element counts. Partial records use
// [begin, end] pairs; merge records use
// [segment_id, partial_begin, partial_end] triples.
MLIR_CAPI_EXPORTED MlirLogicalResult swageMaterializeSegmentedPlan(
    MlirModule module, const int64_t *offsets, intptr_t offsetCount,
    int64_t valueCount, int64_t segmentCount, int64_t warpMaxElements,
    int64_t ctaChunkElements, SwageTaskIdsCallback warpCallback,
    void *warpUserData, SwageTaskIdsCallback ctaCallback, void *ctaUserData,
    SwageTaskIdsCallback partialCallback, void *partialUserData,
    SwageTaskIdsCallback mergeCallback, void *mergeUserData);

#ifdef __cplusplus
}
#endif

#endif // SWAGE_C_CODEGEN_H
