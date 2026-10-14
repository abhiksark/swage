// include/swage-c/Target.h
//===- Target.h - C API for the Swage target description -----------*- C
//-*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// The numbers a host needs to launch what the compiler emits: block widths,
// claim batches, planning defaults, and the admitted processors. The
// lowerings and the code generation entry points read the same record, so a
// host that takes its values from here cannot disagree with a kernel.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_C_TARGET_H
#define SWAGE_C_TARGET_H

#include "mlir-c/Support.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The description of the one target Swage compiles for. The strings and the
/// processor list point into static storage and stay valid for the life of
/// the process.
typedef struct SwageTargetDescription {
  /// A short name for diagnostics and reports.
  MlirStringRef name;
  /// The LLVM target triple kernels are emitted for.
  MlirStringRef triple;
  /// What a processor name starts with; the number follows.
  MlirStringRef processorPrefix;
  /// The `processorCount` processor numbers code generation admits.
  const uint16_t *processors;
  intptr_t processorCount;
  /// Threads that execute in lockstep and exchange values with a shuffle.
  int32_t subgroupWidth;
  /// The widest block a device launches.
  int32_t maxBlockThreads;
  /// The block of a CTA task kernel and of the fused mixed kernel.
  int32_t ctaBlockThreads;
  /// The block of the split partial and merge kernels.
  int32_t splitBlockThreads;
  /// The block of the persistent queue kernel.
  int32_t persistentBlockThreads;
  /// Partial tasks one persistent block claims at a time.
  int32_t persistentPartialClaim;
  /// Warp tasks one persistent subgroup claims at a time.
  int32_t persistentWarpClaim;
  /// The longest segment the planner gives to one subgroup by default.
  int32_t defaultWarpMaxElements;
  /// The longest input range the planner gives to one block by default.
  int32_t defaultCtaChunkElements;
} SwageTargetDescription;

/// Returns the target description. The call takes no context, keeps no state,
/// and may run on any thread.
MLIR_CAPI_EXPORTED SwageTargetDescription swageGetTargetDescription(void);

#ifdef __cplusplus
}
#endif

#endif // SWAGE_C_TARGET_H
