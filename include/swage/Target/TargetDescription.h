// include/swage/Target/TargetDescription.h
//===- TargetDescription.h - Swage target description ----------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// What the lowerings, the code generation C API, and the host need to know
// about the device they compile for, in one record. There is one target and
// therefore one instance; nothing here selects between targets.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_TARGET_TARGETDESCRIPTION_H
#define SWAGE_TARGET_TARGETDESCRIPTION_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>

namespace mlir {
class Location;
class OpBuilder;
namespace gpu {
class GPUFuncOp;
} // namespace gpu
} // namespace mlir

namespace mlir::swage {

struct TargetDescription {
  /// A short name for diagnostics and reports.
  llvm::StringLiteral name;
  /// The LLVM target triple kernels are emitted for.
  llvm::StringLiteral triple;
  /// What a processor name starts with; the number follows.
  llvm::StringLiteral processorPrefix;
  /// The processor numbers code generation admits.
  llvm::ArrayRef<uint16_t> processors;

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

  /// Emits a fence that orders this thread's earlier writes before every
  /// later read on the device.
  void (*emitDeviceFence)(OpBuilder &, Location);
  /// Makes `threads` the only block width the kernel can be launched with.
  void (*pinLaunchWidth)(gpu::GPUFuncOp, int32_t threads);

  /// The number of subgroups `threads` threads occupy, counting a partly
  /// filled last subgroup as one.
  constexpr int64_t subgroupCount(int64_t threads) const {
    return (threads + subgroupWidth - 1) / subgroupWidth;
  }

  /// Whether a kernel that reduces across a block may use `threads` threads.
  ///
  /// A block-wide reduction combines one partial result per subgroup with a
  /// butterfly and stores the result from every participating lane. Each of
  /// those lanes holds the complete reduction only when the subgroup count
  /// is a power of two; for any other count the surviving store may carry an
  /// incomplete reduction.
  constexpr bool admitsBlockThreads(int64_t threads) const {
    return threads > 0 && threads <= maxBlockThreads &&
           llvm::isPowerOf2_64(subgroupCount(threads));
  }

  /// The number of whole subgroups in a block of `threads` threads.
  constexpr int32_t slotsPerBlock(int32_t threads) const {
    return threads / subgroupWidth;
  }

  /// The columns one group of the row-stripe tile covers for rank-two
  /// values of `features` columns: the smallest power of two that is at
  /// least `features`, capped at the subgroup width, and 1 for `features`
  /// of 1 or less. A power of two that divides the subgroup width is what
  /// lets an XOR shuffle pair the lanes of one column only. The kernels
  /// compute it from the feature count with the same rule, and so does the
  /// host that sizes their launch.
  constexpr int32_t columnGroupWidth(int64_t features) const {
    int32_t width = 1;
    while (width < subgroupWidth && width < features)
      width <<= 1;
    return width;
  }
};

/// The description of the one target Swage compiles for.
const TargetDescription &nvidiaTarget();

} // namespace mlir::swage

#endif // SWAGE_TARGET_TARGETDESCRIPTION_H
