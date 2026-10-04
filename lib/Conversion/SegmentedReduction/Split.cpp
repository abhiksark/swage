// lib/Conversion/SegmentedReduction/Split.cpp
//===- Split.cpp - Split segmented emitters
//---------------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "SegmentProgram.h"

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "swage/Support/KernelContract.h"

using namespace mlir;

namespace mlir::swage::detail {
namespace {

std::vector<KernelArgument>
getSplitKernelArguments(const SemanticBufferRoles &roles, bool merge) {
  using Access = KernelArgumentAccess;
  using Origin = KernelArgumentOrigin;
  if (merge)
    return {
        KernelArgument::keyedPointer(Origin::Scratch, "partials", Access::Read),
        KernelArgument::userPointer(roles.outputSourceIndex, Access::Write),
        KernelArgument::keyedPointer(Origin::Plan, "merge_ranges",
                                     Access::Read),
        KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                    "partial_task_count"),
        KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                    "merge_task_count")};
  return {
      KernelArgument::userPointer(roles.valuesSourceIndex, Access::Read),
      KernelArgument::keyedPointer(Origin::Plan, "partial_ranges",
                                   Access::Read),
      KernelArgument::keyedPointer(Origin::Scratch, "partials", Access::Write),
      KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                  "value_count"),
      KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                  "partial_task_count")};
}

} // namespace

void buildSplitGPUProgram(ModuleOp module, func::FuncOp source,
                          const SemanticBufferRoles &roles,
                          const ReductionStage &stage, bool merge) {
  // 512 threads per split CTA: wide enough to stream oversized
  // segments at memory bandwidth, still fully occupied by one 4096-element
  // chunk (8 elements per thread).
  constexpr int64_t blockSize = 512;
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  std::string suffix = merge ? "__merge" : "__partial";
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(
      builder, loc, source.getName().str() + suffix + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  std::vector<KernelArgument> arguments = getSplitKernelArguments(roles, merge);
  auto kernelType = FunctionType::get(
      module.getContext(),
      getKernelArgumentTypes(module.getContext(), arguments), {});
  std::string entryName = source.getName().str() + suffix;
  auto kernel = gpu::GPUFuncOp::create(builder, loc, entryName, kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  KernelContract contract;
  contract.backend = KernelBackend::CUDA;
  contract.entry = entryName;
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
  Value taskIndex = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value taskCount = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), entry->getArgument(4));
  Value inRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        taskIndex, taskCount);

  scf::IfOp::create(
      builder, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
        Value fields =
            arith::ConstantIndexOp::create(body, bodyLoc, merge ? 3 : 2);
        Value recordBase =
            arith::MulIOp::create(body, bodyLoc, taskIndex, fields);
        Value recordPointer = entry->getArgument(merge ? 2 : 1);
        auto loadRecord = [&](int64_t field) {
          Value index = recordBase;
          if (field)
            index = arith::AddIOp::create(
                body, bodyLoc, recordBase,
                arith::ConstantIndexOp::create(body, bodyLoc, field));
          Value index64 = arith::IndexCastOp::create(body, bodyLoc,
                                                     body.getI64Type(), index);
          Value address = LLVM::GEPOp::create(body, bodyLoc, pointer, i32,
                                              recordPointer, index64);
          return Value(LLVM::LoadOp::create(body, bodyLoc, i32, address));
        };

        Value outputIndex = taskIndex;
        int64_t rangeField = 0;
        if (merge) {
          outputIndex = arith::IndexCastOp::create(
              body, bodyLoc, body.getIndexType(), loadRecord(0));
          rangeField = 1;
        }
        Value begin = arith::IndexCastOp::create(
            body, bodyLoc, body.getIndexType(), loadRecord(rangeField));
        Value end = arith::IndexCastOp::create(
            body, bodyLoc, body.getIndexType(), loadRecord(rangeField + 1));
        Value first = arith::AddIOp::create(body, bodyLoc, begin, threadId);
        Value identity = identityFor(body, bodyLoc, stage.kind);
        auto local = scf::ForOp::create(
            body, bodyLoc, first, end, block, ValueRange(identity),
            [&](OpBuilder &loop, Location loopLoc, Value index,
                ValueRange accumulator) {
              Value index64 = arith::IndexCastOp::create(
                  loop, loopLoc, loop.getI64Type(), index);
              Value address = LLVM::GEPOp::create(
                  loop, loopLoc, pointer, f32, entry->getArgument(0), index64);
              Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
              // Only input elements are transformed; scratch holds completed
              // partial reductions and must never run the element program.
              if (!merge)
                value = evaluateElement(loop, stage.element, value, {});
              scf::YieldOp::create(loop, loopLoc,
                                   combine(loop, loopLoc, stage.kind,
                                           accumulator.front(), value));
            });
        auto operation = gpu::AllReduceOperationAttr::get(
            module.getContext(), stage.kind == ReductionKind::Sum
                                     ? gpu::AllReduceOperation::ADD
                                     : gpu::AllReduceOperation::MAXIMUMF);
        Value total = gpu::AllReduceOp::create(
            body, bodyLoc, local.getResult(0), operation, true);
        Value firstThread = arith::CmpIOp::create(
            body, bodyLoc, arith::CmpIPredicate::eq, threadId, zero);
        scf::IfOp::create(
            body, bodyLoc, firstThread,
            [&](OpBuilder &store, Location storeLoc) {
              Value outputIndex64 = arith::IndexCastOp::create(
                  store, storeLoc, store.getI64Type(), outputIndex);
              Value outputAddress = LLVM::GEPOp::create(
                  store, storeLoc, pointer, f32,
                  entry->getArgument(merge ? 1 : 2), outputIndex64);
              LLVM::StoreOp::create(store, storeLoc, total, outputAddress);
              scf::YieldOp::create(store, storeLoc);
            });
        scf::YieldOp::create(body, bodyLoc);
      });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
}

} // namespace mlir::swage::detail
