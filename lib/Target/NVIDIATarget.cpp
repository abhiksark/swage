// lib/Target/NVIDIATarget.cpp
//===- NVIDIATarget.cpp - The NVIDIA target description -------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Target/TargetDescription.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace {

/// The admitted subset of the NVPTX processors defined by the pinned LLVM
/// release (llvm/lib/Target/NVPTX/NVPTX.td in llvmorg-22.1.8). An unknown
/// processor is only a warning to the MC layer, which then falls back to a
/// subtarget that either aborts instruction selection or emits PTX no
/// driver can load. Revisit when cmake/llvm-version.txt moves.
constexpr uint16_t processors[] = {80,  86,  87,  88,  89,  90,
                                   100, 101, 103, 110, 120, 121};

void emitDeviceFence(OpBuilder &builder, Location loc) {
  NVVM::MembarOp::create(builder, loc, NVVM::MemScopeKind::GPU);
}

void pinLaunchWidth(gpu::GPUFuncOp kernel, int32_t threads) {
  kernel->setAttr(
      NVVM::NVVMDialect::getReqntidAttrName(),
      Builder(kernel->getContext()).getDenseI32ArrayAttr({threads, 1, 1}));
}

constexpr swage::TargetDescription description = {
    /*name=*/"nvidia",
    /*triple=*/"nvptx64-nvidia-cuda",
    /*processorPrefix=*/"sm_",
    /*processors=*/processors,
    // The upstream all-reduce lowering and the NVVM conversion hard-code
    // this width, so it describes them as much as it configures Swage.
    /*subgroupWidth=*/32,
    /*maxBlockThreads=*/1024,
    /*ctaBlockThreads=*/128,
    // Wide enough to stream oversized segments at memory bandwidth, still
    // fully occupied by one default chunk at eight elements per thread.
    /*splitBlockThreads=*/512,
    /*persistentBlockThreads=*/512,
    /*persistentPartialClaim=*/4,
    /*persistentWarpClaim=*/8,
    /*defaultWarpMaxElements=*/32,
    /*defaultCtaChunkElements=*/4096,
    /*emitDeviceFence=*/emitDeviceFence,
    /*pinLaunchWidth=*/pinLaunchWidth,
};

// Each fixed block reduces through gpu.all_reduce, or holds whole subgroups.
static_assert(description.admitsBlockThreads(description.ctaBlockThreads),
              "the CTA task kernel reduces through gpu.all_reduce");
static_assert(description.admitsBlockThreads(description.splitBlockThreads),
              "the split CTA reduces through gpu.all_reduce");
static_assert(
    description.admitsBlockThreads(description.persistentBlockThreads),
    "the persistent CTA reduces through gpu.all_reduce");

} // namespace

const swage::TargetDescription &swage::nvidiaTarget() { return description; }
