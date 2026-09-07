//===- FixedBlock.h - Fixed-block lowering --------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_FIXEDBLOCK_FIXEDBLOCK_H
#define SWAGE_CONVERSION_FIXEDBLOCK_FIXEDBLOCK_H

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::swage {

std::unique_ptr<Pass> createFixedBlockToGPUPass(int64_t blockSize = 0);
std::unique_ptr<Pass> createFixedBlockToHostPass(int64_t blockSize = 0);
void registerFixedBlockPasses();

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_FIXEDBLOCK_FIXEDBLOCK_H
