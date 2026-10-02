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
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/IR/PassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
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
  PersistentSegmentedReduction,
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

LogicalResult verifyPTXFunctionNames(ModuleOp source) {
  WalkResult invalidName = source.walk([](func::FuncOp function) {
    if (isPTXIdentifier(function.getName()))
      return WalkResult::advance();
    function.emitError() << "function name '" << function.getName()
                         << "' is not a valid PTX identifier";
    return WalkResult::interrupt();
  });
  return failure(invalidName.wasInterrupted());
}

LogicalResult validateCompileRequest(ModuleOp source,
                                     llvm::StringRef kernelName,
                                     int64_t blockSize,
                                     llvm::StringRef target) {
  unsigned smValue = 0;
  if (!isSupportedTarget(target, smValue))
    return source.emitError()
           << "target must match sm_<major><minor> and be sm_80 or newer, got '"
           << target << "'";
  if (!isPinnedProcessor(smValue))
    return source.emitError("target ")
           << target << " is not a processor supported by the pinned LLVM";
  if (blockSize <= 0)
    return source.emitError()
           << "block_size must be a positive integer, got " << blockSize;
  if (blockSize > 1024)
    return source.emitError()
           << "block_size must be at most 1024, got " << blockSize;
  // The pass manager verifies only after each pass, never before the first
  // one, so an unverified module would reach pass code that dereferences
  // region internals.
  if (failed(verify(source)))
    return failure();
  if (!source.lookupSymbol<func::FuncOp>(kernelName))
    return source.emitError() << "kernel_name '" << kernelName
                              << "' does not name a function of the module";
  return verifyPTXFunctionNames(source);
}

void registerCodegenInterfaces(MLIRContext &context) {
  registerBuiltinDialectTranslation(context);
  registerGPUDialectTranslation(context);
  registerLLVMDialectTranslation(context);
  registerNVVMDialectTranslation(context);

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
  context.appendDialectRegistry(registry);
}

void addKernelLoweringPass(PassManager &manager, KernelKind kind,
                           int64_t blockSize, bool useTaskIds,
                           bool fusedMixed) {
  switch (kind) {
  case KernelKind::FixedBlock:
    manager.addPass(swage::createFixedBlockToGPUPass(blockSize));
    return;
  case KernelKind::SegmentedReduction:
    manager.addPass(swage::createSegmentedReductionToGPUPass(
        blockSize, useTaskIds, fusedMixed));
    return;
  case KernelKind::PersistentSegmentedReduction:
    manager.addPass(swage::createPersistentSegmentedReductionToGPUPass());
    return;
  case KernelKind::SplitPartialReduction:
    manager.addPass(swage::createSplitPartialReductionToGPUPass());
    return;
  case KernelKind::SplitMergeReduction:
    manager.addPass(swage::createSplitMergeReductionToGPUPass());
    return;
  }
}

void configureCodegenPasses(PassManager &manager, KernelKind kind,
                            int64_t blockSize, bool useTaskIds,
                            bool fusedMixed) {
  addKernelLoweringPass(manager, kind, blockSize, useTaskIds, fusedMixed);
  OpPassManager &gpuManager = manager.nest<gpu::GPUModuleOp>();
  gpuManager.addPass(createSCFToControlFlowPass());
  ConvertGpuOpsToNVVMOpsOptions options;
  options.indexBitwidth = 64;
  gpuManager.addPass(createConvertGpuOpsToNVVMOps(options));
}

FailureOr<gpu::GPUModuleOp> lowerToGPU(ModuleOp source, ModuleOp module,
                                       KernelKind kind, int64_t blockSize,
                                       bool useTaskIds, bool fusedMixed) {
  PassManager manager(module.getContext());
  configureCodegenPasses(manager, kind, blockSize, useTaskIds, fusedMixed);
  if (failed(manager.run(module)))
    return failure();

  auto gpuModules = module.getOps<gpu::GPUModuleOp>();
  auto gpuModuleCount = std::distance(gpuModules.begin(), gpuModules.end());
  if (gpuModuleCount != 1) {
    source.emitError() << "lowering did not produce exactly one GPU module, "
                       << "found " << gpuModuleCount;
    return failure();
  }
  return *gpuModules.begin();
}

std::string expectedKernelName(llvm::StringRef kernelName, KernelKind kind) {
  std::string expected = kernelName.str();
  if (kind == KernelKind::SplitPartialReduction)
    expected += "__partial";
  else if (kind == KernelKind::SplitMergeReduction)
    expected += "__merge";
  return expected;
}

LogicalResult verifyCompiledKernel(ModuleOp source, gpu::GPUModuleOp gpuModule,
                                   llvm::StringRef kernelName,
                                   KernelKind kind) {
  llvm::StringRef compiledKernel;
  for (LLVM::LLVMFuncOp function : gpuModule.getOps<LLVM::LLVMFuncOp>())
    if (!function.isExternal())
      compiledKernel = function.getName();
  if (compiledKernel != expectedKernelName(kernelName, kind))
    return source.emitError("kernel_name '")
           << kernelName << "' does not match the compiled kernel '"
           << compiledKernel << "'";
  return success();
}

void printLoweredModule(ModuleOp module, std::string &lowered) {
  llvm::raw_string_ostream loweredStream(lowered);
  module.print(loweredStream, OpPrintingFlags());
  loweredStream.flush();
}

void initializeNVPTX() {
  static llvm::once_flag initializeOnce;
  llvm::call_once(initializeOnce, []() {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
  });
}

/// The LLVM passes that run on every kernel before NVPTX code generation,
/// in the syntax of `opt -passes`.
///
/// The list is curated instead of a default O2 or O3 pipeline because a
/// kernel's synchronization structure must come out exactly as the lowering
/// built it: the same barriers, shuffles, fences, and atomics. A default
/// pipeline built for this target starts with NVVMIntrRange, which narrows
/// the thread-index ranges from the launch width and lets later passes delete
/// shuffle paths for small block sizes. Removing those paths is a job for
/// the lowering, where the block size is a constant.
///
/// What runs, and why:
///   - early-cse: one value per repeated subexpression of an element program.
///   - instcombine: folds casts, comparisons, and index arithmetic. Fixpoint
///     verification is off, because it aborts the process when one iteration
///     does not reach a fixpoint.
///   - simplifycfg: merges blocks and turns small branches into selects. It
///     runs with its default options, which neither hoist nor sink common
///     code, and it never duplicates a block that holds a convergent call.
///   - loop-rotate: gives each reduction loop one conditional branch per
///     iteration instead of a test at the top and a jump at the bottom. It
///     does not rotate a loop whose header holds a convergent call.
///   - licm: hoists loop-invariant arithmetic, which in the persistent kernel
///     is the thread geometry of each block reduction inside a queue loop. It
///     does not hoist or sink a convergent call, and it does not move a load
///     or a store across a barrier or a fence, which are opaque calls.
///   - instcombine again, on what rotation and hoisting exposed.
///
/// What is left out, and why:
///   - Loop unrolling: it would repeat the barriers, shuffles, and atomics of
///     the persistent queue loops, and a reduction loop has one accumulator
///     that cannot be split without reassociating.
///   - Reassociation, vectorization, and anything else that needs a fast-math
///     flag. No pass here adds such a flag or contracts a multiply and an
///     add, so the bits of every result are those of the unoptimized kernel.
///   - GVN, dead store elimination, and memcpy optimization: their gain is
///     reasoning about memory, and shared and global memory is how the
///     threads of these kernels talk to each other.
///   - Jump threading, value propagation, and SCCP: they restructure control
///     flow from value ranges and change nothing in today's kernels.
///   - Every module and call-graph pass: a module holds one kernel, whose
///     signature is its launch ABI and whose shared buffers must stay as
///     lowered.
constexpr llvm::StringLiteral midEndPipeline =
    "function("
    "early-cse,"
    "instcombine<no-verify-fixpoint>,"
    "simplifycfg,"
    "loop-mssa(loop-rotate,licm),"
    "instcombine<no-verify-fixpoint>)";

/// Run the mid-end pipeline on a translated kernel module. The target
/// machine supplies the cost model and the address-space alias analysis; its
/// pipeline-start callbacks are not used, because no default pipeline is
/// built.
LogicalResult optimizeKernel(ModuleOp source, llvm::Module &llvmModule,
                             llvm::TargetMachine &machine) {
  llvm::PassBuilder builder(&machine);
  llvm::LoopAnalysisManager loopAnalyses;
  llvm::FunctionAnalysisManager functionAnalyses;
  llvm::CGSCCAnalysisManager callGraphAnalyses;
  llvm::ModuleAnalysisManager moduleAnalyses;
  builder.registerModuleAnalyses(moduleAnalyses);
  builder.registerCGSCCAnalyses(callGraphAnalyses);
  builder.registerFunctionAnalyses(functionAnalyses);
  builder.registerLoopAnalyses(loopAnalyses);
  builder.crossRegisterProxies(loopAnalyses, functionAnalyses,
                               callGraphAnalyses, moduleAnalyses);

  llvm::ModulePassManager passes;
  if (llvm::Error error = builder.parsePassPipeline(passes, midEndPipeline))
    return source.emitError("failed to build the LLVM pass pipeline '")
           << midEndPipeline << "': " << llvm::toString(std::move(error));
  passes.run(llvmModule, moduleAnalyses);
  return success();
}

LogicalResult emitPTX(ModuleOp source, gpu::GPUModuleOp gpuModule,
                      llvm::StringRef target, std::string &ptx) {
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
  if (failed(optimizeKernel(source, *llvmModule, *machine)))
    return failure();
  FailureOr<llvm::SmallString<0>> generated =
      LLVM::ModuleToObject::translateModuleToISA(*llvmModule, *machine, [&]() {
        return source.emitError("failed to emit PTX");
      });
  if (failed(generated))
    return failure();
  ptx.assign(generated->begin(), generated->end());
  return success();
}

LogicalResult compilePTX(ModuleOp source, llvm::StringRef kernelName,
                         int64_t blockSize, llvm::StringRef target,
                         KernelKind kind, bool useTaskIds, bool fusedMixed,
                         std::string &lowered, std::string &ptx) {
  if (failed(validateCompileRequest(source, kernelName, blockSize, target)))
    return failure();

  OwningOpRef<ModuleOp> module = source.clone();
  MLIRContext *context = module->getContext();
  registerCodegenInterfaces(*context);
  FailureOr<gpu::GPUModuleOp> loweredGPU =
      lowerToGPU(source, *module, kind, blockSize, useTaskIds, fusedMixed);
  if (failed(loweredGPU))
    return failure();
  gpu::GPUModuleOp gpuModule = *loweredGPU;
  // The passes compile the function containing swage operations, not the
  // function kernelName selected; reject a mismatch before module loading.
  if (failed(verifyCompiledKernel(source, gpuModule, kernelName, kind)) ||
      failed(replaceLibdeviceCalls(gpuModule)))
    return failure();
  gpuModule.setTargetsAttr(ArrayAttr::get(
      context,
      {NVVM::NVVMTargetAttr::get(context, 2, "nvptx64-nvidia-cuda", target)}));

  printLoweredModule(*module, lowered);
  initializeNVPTX();
  return emitPTX(source, gpuModule, target, ptx);
}

/// A null module has no context to report on, so it is the one failure
/// without a diagnostic. Every other rejected argument is reported on the
/// module, which keeps the header's promise that a failed call always leaves
/// a diagnostic behind.
LogicalResult verifyStringCallbacks(MlirModule module,
                                    SwageStringCallback loweredCallback,
                                    SwageStringCallback ptxCallback) {
  if (mlirModuleIsNull(module))
    return failure();
  if (!loweredCallback)
    return unwrap(module).emitError("loweredCallback must not be null");
  if (!ptxCallback)
    return unwrap(module).emitError("ptxCallback must not be null");
  return success();
}

/// Reports why swageMaterializeSegmentedPlan refused its buffer or callback
/// arguments. The offsets pointer is only tested, so its element type is not
/// part of this signature.
MlirLogicalResult reportInvalidPlanArguments(MlirModule module,
                                             const void *offsets,
                                             intptr_t offsetCount) {
  if (mlirModuleIsNull(module))
    return mlirLogicalResultFailure();
  ModuleOp source = unwrap(module);
  if (offsetCount < 0)
    source.emitError() << "offsetCount must not be negative, got "
                       << offsetCount;
  else if (offsetCount && !offsets)
    source.emitError() << "offsets must not be null when offsetCount is "
                       << offsetCount;
  else
    source.emitError("warpCallback, ctaCallback, partialCallback, and "
                     "mergeCallback must not be null");
  return mlirLogicalResultFailure();
}

} // namespace

MlirLogicalResult swageCompileFixedBlockToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData) {
  if (failed(verifyStringCallbacks(module, loweredCallback, ptxCallback)))
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
  if (failed(verifyStringCallbacks(module, loweredCallback, ptxCallback)))
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
  if (failed(verifyStringCallbacks(module, loweredCallback, ptxCallback)))
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 128, unwrap(target),
                        KernelKind::SegmentedReduction, false, true, lowered,
                        ptx)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompilePersistentSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData) {
  if (failed(verifyStringCallbacks(module, loweredCallback, ptxCallback)))
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 512, unwrap(target),
                        KernelKind::PersistentSegmentedReduction, false, false,
                        lowered, ptx)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileSplitPartialReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData) {
  if (failed(verifyStringCallbacks(module, loweredCallback, ptxCallback)))
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 512, unwrap(target),
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
  if (failed(verifyStringCallbacks(module, loweredCallback, ptxCallback)))
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 512, unwrap(target),
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
  if (mlirModuleIsNull(module) || offsetCount < 0 ||
      (offsetCount && !offsets) || !warpCallback || !ctaCallback ||
      !partialCallback || !mergeCallback)
    return reportInvalidPlanArguments(module, offsets, offsetCount);

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

MlirLogicalResult swageClassifySegments(
    const int32_t *offsets, intptr_t offsetCount, int64_t valueCount,
    int64_t segmentCount, int64_t warpMaxElements, int64_t ctaChunkElements,
    SwageTaskRecordsCallback recordsCallback, void *recordsUserData,
    SwageStringCallback errorCallback, void *errorUserData) {
  auto fail = [&](const llvm::Twine &reason) {
    if (errorCallback) {
      std::string message = reason.str();
      errorCallback(wrap(llvm::StringRef(message)), errorUserData);
    }
    return mlirLogicalResultFailure();
  };
  if (offsetCount < 0)
    return fail("offsetCount must not be negative, got " +
                llvm::Twine(offsetCount));
  if (offsetCount && !offsets)
    return fail("offsets must not be null when offsetCount is " +
                llvm::Twine(offsetCount));
  if (!recordsCallback)
    return fail("recordsCallback must not be null");

  auto records = swage_plan::classifyTaskRecords(
      ArrayRef(offsets, static_cast<size_t>(offsetCount)), valueCount,
      segmentCount, warpMaxElements, ctaChunkElements);
  if (!records)
    return fail(llvm::toString(records.takeError()));
  recordsCallback(records->records.data(), records->warpCount,
                  records->ctaCount, records->partialCount, records->mergeCount,
                  recordsUserData);
  return mlirLogicalResultSuccess();
}
