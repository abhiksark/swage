// lib/CAPI/Target.cpp
//===- Target.cpp - C API for the Swage target description ----------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Target.h"

#include "mlir/CAPI/Support.h"
#include "swage/Target/TargetDescription.h"

using namespace mlir;

SwageTargetDescription swageGetTargetDescription(void) {
  const swage::TargetDescription &target = swage::nvidiaTarget();
  SwageTargetDescription record;
  record.name = wrap(llvm::StringRef(target.name));
  record.triple = wrap(llvm::StringRef(target.triple));
  record.processorPrefix = wrap(llvm::StringRef(target.processorPrefix));
  record.processors = target.processors.data();
  record.processorCount = static_cast<intptr_t>(target.processors.size());
  record.subgroupWidth = target.subgroupWidth;
  record.maxBlockThreads = target.maxBlockThreads;
  record.ctaBlockThreads = target.ctaBlockThreads;
  record.splitBlockThreads = target.splitBlockThreads;
  record.persistentBlockThreads = target.persistentBlockThreads;
  record.persistentPartialClaim = target.persistentPartialClaim;
  record.persistentWarpClaim = target.persistentWarpClaim;
  record.defaultWarpMaxElements = target.defaultWarpMaxElements;
  record.defaultCtaChunkElements = target.defaultCtaChunkElements;
  return record;
}
