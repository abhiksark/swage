// lib/CAPI/Codegen.cpp
//===- Codegen.cpp - Swage code generation C API ------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage-c/Codegen.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVM/ModuleToObject.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "swage/Conversion/FixedBlockToGPU/FixedBlockToGPU.h"
#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Dialect/SwagePlan/IR/TaskClassifier.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;

namespace {

enum class KernelKind {
  FixedBlock,
  SegmentedReduction,
  SplitPartialReduction,
  SplitMergeReduction,
};

/// The NVPTX assembly printer calls report_fatal_error on symbols it cannot
/// print, which aborts the embedding process, so names must be rejected here
/// while a diagnostic is still possible.
bool isPTXIdentifier(llvm::StringRef name) {
  auto isFollowing = [](char character) {
    return llvm::isAlnum(character) || character == '_' || character == '$';
  };
  return !name.empty() && !llvm::isDigit(name.front()) &&
         llvm::all_of(name, isFollowing);
}

bool isSupportedTarget(llvm::StringRef target, unsigned &value) {
  llvm::StringRef digits = target.consume_front("sm_") ? target : "";
  if (digits.size() < 2 || digits.size() > 3 ||
      !llvm::all_of(digits, llvm::isDigit))
    return false;
  return !digits.getAsInteger(10, value) && value >= 80 && value <= 129;
}

/// The admitted subset of the NVPTX processors defined by the pinned LLVM
/// release (llvm/lib/Target/NVPTX/NVPTX.td in llvmorg-22.1.8). An unknown
/// processor is only a warning to the MC layer, which then falls back to a
/// subtarget that either aborts instruction selection or emits PTX no
/// driver can load. Revisit when cmake/llvm-version.txt moves.
bool isPinnedProcessor(unsigned value) {
  switch (value) {
  case 80:
  case 86:
  case 87:
  case 88:
  case 89:
  case 90:
  case 100:
  case 101:
  case 103:
  case 110:
  case 120:
  case 121:
    return true;
  default:
    return false;
  }
}

/// Replace libdevice calls with LLVM intrinsics the NVPTX backend lowers
/// natively.
///
/// `--convert-gpu-to-nvvm` rewrites every `math` operation into a call to a
/// libdevice symbol such as `__nv_exp2f`. Nothing in this path links
/// libdevice, so the emitted PTX would carry an unresolvable `.extern .func`
/// and fail to load. The equivalent LLVM intrinsic becomes a native
/// instruction, `ex2.approx.f32` for exp2, at the cost of the hardware
/// approximation rather than libdevice's correctly rounded result.
LogicalResult replaceLibdeviceCalls(gpu::GPUModuleOp gpuModule) {
  SmallVector<LLVM::CallOp> calls;
  gpuModule.walk([&](LLVM::CallOp call) {
    if (call.getCallee() == "__nv_exp2f")
      calls.push_back(call);
  });
  for (LLVM::CallOp call : calls) {
    OpBuilder builder(call);
    Value exponential =
        LLVM::Exp2Op::create(builder, call.getLoc(), call.getOperand(0));
    call.getResult().replaceAllUsesWith(exponential);
    call.erase();
  }
  // Any surviving libdevice declaration is an unresolvable extern, so fail
  // rather than emit a module that cannot load.
  WalkResult remaining = gpuModule.walk([&](LLVM::LLVMFuncOp function) {
    if (!function.isExternal() || !function.getName().starts_with("__nv_"))
      return WalkResult::advance();
    if (function.symbolKnownUseEmpty(gpuModule)) {
      function.erase();
      return WalkResult::advance();
    }
    function.emitError("no libdevice implementation is linked for ")
        << function.getName();
    return WalkResult::interrupt();
  });
  return failure(remaining.wasInterrupted());
}

LogicalResult compilePTX(ModuleOp source, llvm::StringRef kernelName,
                         int64_t blockSize, llvm::StringRef target,
                         KernelKind kind, bool useTaskIds, bool fusedMixed,
                         std::string &lowered, std::string &ptx) {
  unsigned smValue = 0;
  if (!isSupportedTarget(target, smValue))
    return source.emitError(
        "target must match sm_<major><minor> and be sm_80 or newer");
  if (!isPinnedProcessor(smValue))
    return source.emitError("target ")
           << target << " is not a processor supported by the pinned LLVM";
  if (blockSize <= 0)
    return source.emitError("block_size must be a positive integer");
  // The pass manager verifies only after each pass, never before the first
  // one, so an unverified module would reach pass code that dereferences
  // region internals.
  if (failed(verify(source)))
    return failure();

  OwningOpRef<ModuleOp> module = source.clone();
  MLIRContext *context = module->getContext();
  if (!module->lookupSymbol<func::FuncOp>(kernelName))
    return source.emitError("kernel_name does not name the module function");
  WalkResult invalidName = source.walk([](func::FuncOp function) {
    if (isPTXIdentifier(function.getName()))
      return WalkResult::advance();
    function.emitError("function name is not a valid PTX identifier");
    return WalkResult::interrupt();
  });
  if (invalidName.wasInterrupted())
    return failure();

  registerBuiltinDialectTranslation(*context);
  registerGPUDialectTranslation(*context);
  registerLLVMDialectTranslation(*context);
  registerNVVMDialectTranslation(*context);

  DialectRegistry registry;
  arith::registerConvertArithToLLVMInterface(registry);
  cf::registerConvertControlFlowToLLVMInterface(registry);
  registerConvertFuncToLLVMInterface(registry);
  index::registerConvertIndexToLLVMInterface(registry);
  registerConvertMathToLLVMInterface(registry);
  registerConvertMemRefToLLVMInterface(registry);
  registerConvertNVVMToLLVMInterface(registry);
  ub::registerConvertUBToLLVMInterface(registry);
  vector::registerConvertVectorToLLVMInterface(registry);
  context->appendDialectRegistry(registry);

  PassManager manager(context);
  if (kind == KernelKind::FixedBlock) {
    manager.addPass(swage::createFixedBlockToGPUPass(blockSize));
  } else if (kind == KernelKind::SegmentedReduction) {
    manager.addPass(swage::createSegmentedReductionToGPUPass(
        blockSize, useTaskIds, fusedMixed));
  } else if (kind == KernelKind::SplitPartialReduction) {
    manager.addPass(swage::createSplitPartialReductionToGPUPass());
  } else {
    manager.addPass(swage::createSplitMergeReductionToGPUPass());
  }
  OpPassManager &gpuManager = manager.nest<gpu::GPUModuleOp>();
  gpuManager.addPass(createSCFToControlFlowPass());
  ConvertGpuOpsToNVVMOpsOptions options;
  options.indexBitwidth = 64;
  gpuManager.addPass(createConvertGpuOpsToNVVMOps(options));
  if (failed(manager.run(*module)))
    return failure();

  auto gpuModules = module->getOps<gpu::GPUModuleOp>();
  if (std::distance(gpuModules.begin(), gpuModules.end()) != 1)
    return source.emitError("lowering did not produce exactly one GPU module");
  gpu::GPUModuleOp gpuModule = *gpuModules.begin();
  if (failed(replaceLibdeviceCalls(gpuModule)))
    return failure();
  gpuModule.setTargetsAttr(ArrayAttr::get(
      context,
      {NVVM::NVVMTargetAttr::get(context, 2, "nvptx64-nvidia-cuda", target)}));

  llvm::raw_string_ostream loweredStream(lowered);
  module->print(loweredStream, OpPrintingFlags());
  loweredStream.flush();

  static llvm::once_flag initializeNVPTXOnce;
  llvm::call_once(initializeNVPTXOnce, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });

  constexpr llvm::StringLiteral triple = "nvptx64-nvidia-cuda";
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = translateModuleToLLVMIR(
      gpuModule.getOperation(), llvmContext, gpuModule.getName());
  if (!llvmModule)
    return source.emitError("failed to translate lowered MLIR to LLVM IR");

  std::string error;
  const llvm::Target *nvptx =
      llvm::TargetRegistry::lookupTarget(llvm::Triple(triple), error);
  if (!nvptx)
    return source.emitError(error);
  std::unique_ptr<llvm::TargetMachine> machine(
      nvptx->createTargetMachine(llvm::Triple(triple), target, "", {}, {}));
  if (!machine)
    return source.emitError("failed to create the requested NVPTX target");
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(machine->getTargetTriple());
  FailureOr<llvm::SmallString<0>> generated =
      LLVM::ModuleToObject::translateModuleToISA(*llvmModule, *machine, [&]() {
        return source.emitError("failed to emit PTX");
      });
  if (failed(generated))
    return failure();
  ptx.assign(generated->begin(), generated->end());
  return success();
}

} // namespace

MlirLogicalResult swageCompileFixedBlockToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData) {
  if (!loweredCallback || !ptxCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), blockSize,
                        unwrap(target), KernelKind::FixedBlock, false, false,
                        lowered, ptx)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, bool useTaskIds, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData) {
  if (!loweredCallback || !ptxCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), blockSize,
                        unwrap(target), KernelKind::SegmentedReduction,
                        useTaskIds, false, lowered, ptx)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileFusedSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData) {
  if (!loweredCallback || !ptxCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 128, unwrap(target),
                        KernelKind::SegmentedReduction, true, true, lowered,
                        ptx)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileSplitPartialReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData) {
  if (!loweredCallback || !ptxCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 128, unwrap(target),
                        KernelKind::SplitPartialReduction, false, false,
                        lowered, ptx)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileSplitMergeReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData) {
  if (!loweredCallback || !ptxCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 128, unwrap(target),
                        KernelKind::SplitMergeReduction, false, false, lowered,
                        ptx)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageMaterializeSegmentedPlan(
    MlirModule module, const int64_t *offsets, intptr_t offsetCount,
    int64_t valueCount, int64_t segmentCount, int64_t warpMaxElements,
    int64_t ctaChunkElements, SwageTaskIdsCallback warpCallback,
    void *warpUserData, SwageTaskIdsCallback ctaCallback, void *ctaUserData,
    SwageTaskIdsCallback partialCallback, void *partialUserData,
    SwageTaskIdsCallback mergeCallback, void *mergeUserData) {
  if (offsetCount < 0 || (offsetCount && !offsets) || !warpCallback ||
      !ctaCallback || !partialCallback || !mergeCallback)
    return mlirLogicalResultFailure();

  ModuleOp source = unwrap(module);
  if (failed(verify(source)))
    return mlirLogicalResultFailure();
  OwningOpRef<ModuleOp> planned = source.clone();
  PassManager manager(source.getContext());
  manager.addPass(
      swage::createSwageToPlanPass(warpMaxElements, ctaChunkElements));
  if (failed(manager.run(*planned)))
    return mlirLogicalResultFailure();

  SmallVector<swage_plan::ClassifyOp> classifiers;
  planned->walk([&](swage_plan::ClassifyOp classify) {
    classifiers.push_back(classify);
  });
  if (classifiers.size() != 1) {
    source.emitError(
        "planning did not produce exactly one swage_plan.classify");
    return mlirLogicalResultFailure();
  }

  auto tasks = swage_plan::classifyTasks(
      ArrayRef(offsets, static_cast<size_t>(offsetCount)), valueCount,
      segmentCount, classifiers.front().getWarpMaxElements(),
      classifiers.front().getCtaChunkElements());
  if (!tasks) {
    source.emitError(llvm::toString(tasks.takeError()));
    return mlirLogicalResultFailure();
  }

  std::vector<int32_t> warp;
  std::vector<int32_t> cta;
  std::vector<int32_t> partial;
  std::vector<int32_t> merge;
  llvm::SmallDenseSet<int32_t, 8> splitSegments;
  for (const swage_plan::TaskDescriptor &task : *tasks)
    if (task.stage == 1)
      splitSegments.insert(task.segment_id);
  warp.reserve(tasks->size());
  cta.reserve(tasks->size());
  for (const swage_plan::TaskDescriptor &task : *tasks) {
    if (task.stage == 1) {
      merge.insert(merge.end(), {task.segment_id, task.begin, task.end});
    } else if (task.policy == swage_plan::TaskPolicy::Warp) {
      warp.push_back(task.segment_id);
    } else if (splitSegments.contains(task.segment_id)) {
      partial.insert(partial.end(), {task.begin, task.end});
    } else {
      cta.push_back(task.segment_id);
    }
  }
  warpCallback(warp.data(), static_cast<intptr_t>(warp.size()), warpUserData);
  ctaCallback(cta.data(), static_cast<intptr_t>(cta.size()), ctaUserData);
  partialCallback(partial.data(), static_cast<intptr_t>(partial.size()),
                  partialUserData);
  mergeCallback(merge.data(), static_cast<intptr_t>(merge.size()),
                mergeUserData);
  return mlirLogicalResultSuccess();
}
