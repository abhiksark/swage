// include/swage-c/Codegen.h
//===- Codegen.h - Swage code generation C API -----------------*- C -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// Compiles a Swage semantic module to NVPTX assembly and classifies segment
// metadata into task records. The contract below holds for every function in
// this header that takes a module. swageClassifySegments takes none; its own
// comment states what differs.
//
// Module and context
//   The module is read and never modified: a call clones it and lowers the
//   clone. The clone lives in the context of the module, so a call changes
//   that context. A compile function appends a dialect registry that carries
//   the LLVM conversion interfaces, registers the LLVM IR translation
//   interfaces, and loads the dialects the lowering produces, among them gpu,
//   scf, llvm, and nvvm, and swage_plan when the kernel is planned before
//   it is lowered. These changes stay after the call returns and repeating
//   them is harmless. swageMaterializeSegmentedPlan builds no IR and leaves
//   the context as it was.
//   The dialects of the input itself (swage, func, arith, math, memref,
//   vector) must already be loaded, which parsing the module ensures.
//
// Code generation
//   A compile function lowers the clone to the LLVM dialect, translates it
//   to LLVM IR, runs a fixed list of LLVM passes on that IR, and emits PTX
//   with the NVPTX backend. The passes keep the kernel name, its parameters,
//   its launch width, each barrier, warp shuffle, memory fence, and atomic,
//   and the result bits of its floating-point arithmetic. The lowered module
//   that loweredCallback receives is the MLIR before the translation, so the
//   effect of the LLVM passes shows in the PTX and not in that text.
//
// Threads
//   A call uses the context of its module the way a pass pipeline does:
//   no other thread may use that context, or any IR in it, until the call
//   returns. Calls on modules of different contexts may run at the same time.
//   The functions keep no state between calls except the one-time NVPTX
//   target initialization, which is safe to race.
//
// Callbacks
//   A callback runs at most once, on the calling thread, before the function
//   returns, and only when the call succeeds. A failed call runs no callback.
//   The data a callback receives belongs to the call and is valid only until
//   the callback returns, so a callback copies what it keeps. Strings are
//   given by length and are not promised to be null-terminated. The user data
//   pointer is passed through and not retained.
//
// Failure
//   A failed call returns mlirLogicalResultFailure and has emitted at least
//   one error diagnostic through the diagnostic handlers of the context of
//   the module. Attach a handler with mlirContextAttachDiagnosticHandler
//   before the call to capture the text; without one MLIR prints it to
//   stderr. The one exception is a null module, which has no context to
//   report on and fails without a diagnostic.
//
//   Invalid input is rejected with a diagnostic, including the two inputs
//   that would make the LLVM NVPTX backend terminate the process: a function
//   name that is not a PTX identifier and a processor the pinned LLVM does
//   not define. No fatal error handler is installed, so an internal LLVM
//   error outside those cases still terminates the process.
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

/// Receives one result string of a compile function.
typedef void (*SwageStringCallback)(MlirStringRef value, void *userData);
/// Receives one flat list of i32 plan records.
typedef void (*SwageTaskIdsCallback)(const int32_t *taskIds, intptr_t taskCount,
                                     void *userData);

// Arguments shared by the compile functions below.
//
//   kernelName      names the function to compile, which must hold Swage
//                   operations. A segmented compile lowers that function and
//                   leaves every other function of the module as it is, so a
//                   module may hold several segment functions. The lowering
//                   creates a gpu.module named `<kernel>_module`, where
//                   `<kernel>` is the name of the kernel in the PTX: that
//                   symbol must be free and nothing in the module may refer
//                   to the function. The kernel carries the name of the
//                   function unless noted.
//   target          is an NVPTX processor: sm_80, sm_86, sm_87, sm_88, sm_89,
//                   sm_90, sm_100, sm_101, sm_103, sm_110, sm_120, or sm_121.
//   loweredCallback receives the lowered module as MLIR text.
//   ptxCallback     receives the PTX text. It runs after loweredCallback.

/// Compiles the fixed vector-add kernel. `blockSize` is the launch width in
/// threads, from 1 to 1024, and must equal the vector width of the module.
MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileFixedBlockToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData);

/// Compiles a segmented reduction to a kernel that reduces one segment per
/// block. `blockSize` is the launch width in threads, from 1 to 1024; its
/// warp count, `blockSize` divided by 32 and rounded up, must be a power of
/// two. `useTaskIds` selects the launch ABI that reads segment ids from a
/// task buffer.
MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, bool useTaskIds, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData);

/// Compiles a segmented reduction to the fused kernel that serves warp and
/// block tasks in one launch. The launch width is fixed at 128 threads.
MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileFusedSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData);

/// Compiles an identity f32 sum to the persistent queue kernel. The launch
/// width is fixed at 512 threads.
MLIR_CAPI_EXPORTED MlirLogicalResult
swageCompilePersistentSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData);

/// Compiles the first stage of a split reduction: one partial result per
/// chunk of a long segment. The launch width is fixed at 512 threads and the
/// kernel in the PTX is named `<kernelName>__partial`.
MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileSplitPartialReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData);

/// Compiles the second stage of a split reduction: one result per segment
/// from its partial results. The launch width is fixed at 512 threads and the
/// kernel in the PTX is named `<kernelName>__merge`.
MLIR_CAPI_EXPORTED MlirLogicalResult swageCompileSplitMergeReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData);

/// Admits the function `kernelName` names for planning and classifies the
/// offsets into one record list per callback. The function must hold a
/// capture-free sum or max; other functions of the module are not looked at.
/// The limits must satisfy
/// `0 < warpMaxElements <= ctaChunkElements <= INT32_MAX`. The call reads
/// the module and runs no pass.
/// Callback counts are flat i32 element counts. Partial records use
/// [begin, end] pairs; merge records use
/// [segment_id, partial_begin, partial_end] triples.
MLIR_CAPI_EXPORTED MlirLogicalResult swageMaterializeSegmentedPlan(
    MlirModule module, MlirStringRef kernelName, const int64_t *offsets,
    intptr_t offsetCount, int64_t valueCount, int64_t segmentCount,
    int64_t warpMaxElements, int64_t ctaChunkElements,
    SwageTaskIdsCallback warpCallback, void *warpUserData,
    SwageTaskIdsCallback ctaCallback, void *ctaUserData,
    SwageTaskIdsCallback partialCallback, void *partialUserData,
    SwageTaskIdsCallback mergeCallback, void *mergeUserData);

/// Receives the records of one classification in one buffer: `warpCount`
/// warp segment ids, then `ctaCount` CTA segment ids, then `partialCount`
/// [begin, end] pairs, then `mergeCount`
/// [segment_id, partial_begin, partial_end] triples, then for each of the
/// `partialCount` partial tasks the index of its merge triple. The counts
/// are in ids, pairs, and triples, so the buffer holds
/// `warpCount + ctaCount + 3 * partialCount + 3 * mergeCount` values.
typedef void (*SwageTaskRecordsCallback)(const int32_t *records,
                                         intptr_t warpCount, intptr_t ctaCount,
                                         intptr_t partialCount,
                                         intptr_t mergeCount, void *userData);

/// Classifies segment offsets into the records swageMaterializeSegmentedPlan
/// produces for the same offsets and limits, without a module, and adds the
/// merge index of every partial task, which follows from those records.
///
/// A program is admitted for planning by swageMaterializeSegmentedPlan, which
/// checks the named function of its module and the two limits. That result
/// depends on the program and the limits only, so a caller
/// that classifies many layouts of one program admits it once, for example
/// with a layout of no segments, and classifies each layout here.
///
/// The call reads `offsetCount` i32 offsets, which must equal
/// `segmentCount + 1`, and nothing else: it takes no module, uses no MLIR
/// context, and keeps no state, so calls may run on any threads at the same
/// time. The offsets must not change during the call.
///
/// On success `recordsCallback` runs once, under the callback rules above,
/// and `errorCallback` does not run. On failure `recordsCallback` does not
/// run, and `errorCallback`, which may be null, receives the reason: there
/// is no context to carry a diagnostic. The metadata reasons are the ones
/// swageMaterializeSegmentedPlan reports for the same input.
MLIR_CAPI_EXPORTED MlirLogicalResult swageClassifySegments(
    const int32_t *offsets, intptr_t offsetCount, int64_t valueCount,
    int64_t segmentCount, int64_t warpMaxElements, int64_t ctaChunkElements,
    SwageTaskRecordsCallback recordsCallback, void *recordsUserData,
    SwageStringCallback errorCallback, void *errorUserData);

#ifdef __cplusplus
}
#endif

#endif // SWAGE_C_CODEGEN_H
