//===- Host.cpp - Fixed-block host emission -----------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "swage/Support/KernelContract.h"

#include <vector>

using namespace mlir;

namespace mlir::swage::detail {

void buildFixedHostProgram(ModuleOp module, func::FuncOp source,
                           int64_t blockSize) {
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  builder.setInsertionPoint(source);

  Type elementType =
      cast<MemRefType>(source.getArgument(0).getType()).getElementType();
  std::vector<KernelArgument> arguments{
      KernelArgument::userPointer(0, KernelArgumentAccess::Read),
      KernelArgument::userPointer(1, KernelArgumentAccess::Read),
      KernelArgument::userPointer(2, KernelArgumentAccess::Write),
      KernelArgument::userScalar(KernelArgumentKind::I32, 3)};
  auto functionType = FunctionType::get(
      module.getContext(),
      getKernelArgumentTypes(module.getContext(), arguments), {});
  auto function =
      func::FuncOp::create(builder, loc, source.getName(), functionType);
  KernelContract contract;
  contract.backend = KernelBackend::CPU;
  contract.entry = source.getName().str();
  contract.launch = {KernelLaunchModel::HostCall, std::nullopt};
  contract.arguments = arguments;
  function->setDiscardableAttr(kernelContractAttrName,
                               buildKernelContractAttr(builder, contract));

  Block *entry = function.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value lower = arith::ConstantIndexOp::create(builder, loc, 0);
  Value upper = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                           entry->getArgument(3));
  Value step = arith::ConstantIndexOp::create(builder, loc, 1);
  scf::ForOp::create(
      builder, loc, lower, upper, step, ValueRange{},
      [&](OpBuilder &body, Location bodyLoc, Value offset, ValueRange) {
        Value byteOffset = arith::IndexCastOp::create(
            body, bodyLoc, body.getI64Type(), offset);
        buildFixedScalarAdd(body, bodyLoc, elementType, entry->getArgument(0),
                            entry->getArgument(1), entry->getArgument(2),
                            byteOffset);
        scf::YieldOp::create(body, bodyLoc);
      });
  func::ReturnOp::create(builder, loc);
  source.erase();
  (void)blockSize;
}

} // namespace mlir::swage::detail
