// lib/Conversion/FixedBlock/GPU.cpp
//===- GPU.cpp - Fixed-block GPU emission -------------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "swage/Support/KernelContract.h"

#include <array>
#include <vector>

using namespace mlir;

namespace mlir::swage::detail {

void buildFixedGPUProgram(ModuleOp module, func::FuncOp source,
                          int64_t blockSize, FixedElementwiseKind kind) {
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(builder, loc,
                                            source.getName().str() + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type elementType =
      cast<MemRefType>(source.getArgument(0).getType()).getElementType();
  std::vector<KernelArgument> arguments{
      KernelArgument::userPointer(0, KernelArgumentAccess::Read),
      KernelArgument::userPointer(1, KernelArgumentAccess::Read),
      KernelArgument::userPointer(2, KernelArgumentAccess::Write),
      KernelArgument::userScalar(KernelArgumentKind::I32, 3)};
  auto kernelType = FunctionType::get(
      module.getContext(),
      getKernelArgumentTypes(module.getContext(), arguments), {});
  auto kernel =
      gpu::GPUFuncOp::create(builder, loc, source.getName(), kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  KernelContract contract;
  contract.backend = KernelBackend::CUDA;
  contract.entry = source.getName().str();
  contract.launch = {
      KernelLaunchModel::SPMDGrid,
      std::array<int32_t, 3>{static_cast<int32_t>(blockSize), 1, 1}};
  contract.arguments = arguments;
  kernel->setDiscardableAttr(kernelContractAttrName,
                             buildKernelContractAttr(builder, contract));
  kernel->setAttr(NVVM::NVVMDialect::getReqntidAttrName(),
                  builder.getDenseI32ArrayAttr(*contract.launch.block));

  Block *entry = &kernel.getBody().front();
  builder.setInsertionPointToStart(entry);

  Value blockId = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value base = arith::MulIOp::create(builder, loc, blockId, block);
  Value offset = arith::AddIOp::create(builder, loc, base, threadId);
  Value n = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                       entry->getArgument(3));
  Value inBounds =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, offset, n);
  Value byteOffset =
      arith::IndexCastOp::create(builder, loc, builder.getI64Type(), offset);

  scf::IfOp::create(
      builder, loc, inBounds, [&](OpBuilder &body, Location bodyLoc) {
        buildFixedScalarElementwise(
            body, bodyLoc, elementType, entry->getArgument(0),
            entry->getArgument(1), entry->getArgument(2), byteOffset, kind);
        scf::YieldOp::create(body, bodyLoc);
      });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
}

} // namespace mlir::swage::detail
