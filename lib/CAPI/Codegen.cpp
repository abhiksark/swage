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
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
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
#include "mlir/Transforms/Passes.h"
#include "swage/Conversion/FixedBlock/FixedBlock.h"
#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Dialect/SwagePlan/IR/TaskClassifier.h"
#include "swage/Support/KernelContract.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
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
  PersistentSegmentedReduction,
  SplitPartialReduction,
  SplitMergeReduction,
};

struct HostExecutable {
  std::unique_ptr<ExecutionEngine> engine;
  std::string entry;
  intptr_t argumentCount;
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
    function.emitError("function name is not a valid PTX identifier");
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
    return source.emitError(
        "target must match sm_<major><minor> and be sm_80 or newer");
  if (!isPinnedProcessor(smValue))
    return source.emitError("target ")
           << target << " is not a processor supported by the pinned LLVM";
  if (blockSize <= 0)
    return source.emitError("block_size must be a positive integer");
  if (blockSize > 1024)
    return source.emitError("block_size must be at most 1024");
  // The pass manager verifies only after each pass, never before the first
  // one, so an unverified module would reach pass code that dereferences
  // region internals.
  if (failed(verify(source)))
    return failure();
  if (!source.lookupSymbol<func::FuncOp>(kernelName))
    return source.emitError("kernel_name does not name the module function");
  return verifyPTXFunctionNames(source);
}

LogicalResult validateHostCompileRequest(ModuleOp source,
                                         llvm::StringRef kernelName,
                                         int64_t blockSize) {
  if (blockSize <= 0)
    return source.emitError("block_size must be a positive integer");
  if (blockSize > 1024)
    return source.emitError("block_size must be at most 1024");
  if (failed(verify(source)))
    return failure();
  if (!source.lookupSymbol<func::FuncOp>(kernelName))
    return source.emitError("kernel_name does not name the module function");
  return success();
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

void configureUpstreamCodegenPasses(PassManager &manager) {
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
  addKernelLoweringPass(manager, kind, blockSize, useTaskIds, fusedMixed);
  if (failed(manager.run(module)))
    return failure();

  auto gpuModules = module.getOps<gpu::GPUModuleOp>();
  if (std::distance(gpuModules.begin(), gpuModules.end()) != 1) {
    source.emitError("lowering did not produce exactly one GPU module");
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

FailureOr<swage::KernelContract>
extractKernelContract(ModuleOp source, gpu::GPUModuleOp gpuModule,
                      llvm::StringRef kernelName, KernelKind kind) {
  SmallVector<gpu::GPUFuncOp> functions =
      llvm::to_vector(gpuModule.getOps<gpu::GPUFuncOp>());
  SmallVector<gpu::GPUFuncOp> contractFunctions;
  for (gpu::GPUFuncOp function : functions)
    if (function->getDiscardableAttr(swage::kernelContractAttrName))
      contractFunctions.push_back(function);
  if (contractFunctions.size() != 1) {
    source.emitError(
        "lowering did not produce exactly one contract-bearing GPU function");
    return failure();
  }
  if (functions.size() != 1) {
    source.emitError(
        "kernel contract must describe exactly one generated GPU function");
    return failure();
  }

  gpu::GPUFuncOp function = contractFunctions.front();
  std::string expected = expectedKernelName(kernelName, kind);
  if (function.getName() != expected) {
    source.emitError("kernel_name '")
        << kernelName << "' does not match the compiled kernel '"
        << function.getName() << "'";
    return failure();
  }
  llvm::Expected<swage::KernelContract> contract =
      swage::parseKernelContractAttr(
          function->getDiscardableAttr(swage::kernelContractAttrName));
  if (!contract) {
    source.emitError("malformed swage.kernel_contract: ")
        << llvm::toString(contract.takeError());
    return failure();
  }
  auto requiredBlock = function->getAttrOfType<DenseI32ArrayAttr>(
      NVVM::NVVMDialect::getReqntidAttrName());
  if (!requiredBlock) {
    source.emitError("contract-bearing GPU function requires nvvm.reqntid");
    return failure();
  }
  if (llvm::Error error = swage::validateCUDAKernelContract(
          *contract, function.getName(), function.getFunctionType(),
          requiredBlock.asArrayRef())) {
    source.emitError("invalid swage.kernel_contract: ")
        << llvm::toString(std::move(error));
    return failure();
  }
  function->removeDiscardableAttr(swage::kernelContractAttrName);
  return std::move(*contract);
}

LogicalResult lowerGPUToNVVM(ModuleOp module) {
  PassManager manager(module.getContext());
  configureUpstreamCodegenPasses(manager);
  return manager.run(module);
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
                         std::string &lowered, std::string &ptx,
                         std::string &contractJSON) {
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
  FailureOr<swage::KernelContract> contract =
      extractKernelContract(source, gpuModule, kernelName, kind);
  if (failed(contract))
    return failure();
  contractJSON = swage::serializeKernelContractJSON(*contract);
  if (failed(lowerGPUToNVVM(*module)) ||
      failed(verifyCompiledKernel(source, gpuModule, kernelName, kind)) ||
      failed(replaceLibdeviceCalls(gpuModule)))
    return failure();
  gpuModule.setTargetsAttr(ArrayAttr::get(
      context,
      {NVVM::NVVMTargetAttr::get(context, 2, "nvptx64-nvidia-cuda", target)}));

  printLoweredModule(*module, lowered);
  initializeNVPTX();
  return emitPTX(source, gpuModule, target, ptx);
}

FailureOr<swage::KernelContract>
extractHostKernelContract(ModuleOp source, ModuleOp module,
                          llvm::StringRef kernelName) {
  SmallVector<func::FuncOp> functions =
      llvm::to_vector(module.getOps<func::FuncOp>());
  if (functions.size() != 1 ||
      !functions.front()->getDiscardableAttr(swage::kernelContractAttrName)) {
    source.emitError(
        "lowering did not produce exactly one contract-bearing host function");
    return failure();
  }
  func::FuncOp function = functions.front();
  if (function.getName() != kernelName) {
    source.emitError("kernel_name '")
        << kernelName << "' does not match the compiled kernel '"
        << function.getName() << "'";
    return failure();
  }
  llvm::Expected<swage::KernelContract> contract =
      swage::parseKernelContractAttr(
          function->getDiscardableAttr(swage::kernelContractAttrName));
  if (!contract) {
    source.emitError("malformed swage.kernel_contract: ")
        << llvm::toString(contract.takeError());
    return failure();
  }
  if (llvm::Error error = swage::validateHostKernelContract(
          *contract, function.getName(), function.getFunctionType())) {
    source.emitError("invalid swage.kernel_contract: ")
        << llvm::toString(std::move(error));
    return failure();
  }
  function->removeDiscardableAttr(swage::kernelContractAttrName);
  return std::move(*contract);
}

LogicalResult lowerHostToLLVM(ModuleOp module) {
  PassManager manager(module.getContext());
  manager.addPass(createSCFToControlFlowPass());
  manager.addPass(createConvertIndexToLLVMPass());
  manager.addPass(createArithToLLVMConversionPass());
  manager.addPass(createConvertControlFlowToLLVMPass());
  manager.addPass(createConvertFuncToLLVMPass());
  manager.addPass(createReconcileUnrealizedCastsPass());
  return manager.run(module);
}

void initializeNativeTarget() {
  static llvm::once_flag initializeOnce;
  llvm::call_once(initializeOnce, []() {
    if (llvm::InitializeNativeTarget())
      llvm::report_fatal_error("failed to initialize the native LLVM target");
    if (llvm::InitializeNativeTargetAsmPrinter())
      llvm::report_fatal_error(
          "failed to initialize the native LLVM assembly printer");
  });
}

std::unique_ptr<HostExecutable>
compileHost(ModuleOp source, llvm::StringRef kernelName, int64_t blockSize,
            std::string &lowered, std::string &contractJSON) {
  if (failed(validateHostCompileRequest(source, kernelName, blockSize)))
    return nullptr;

  OwningOpRef<ModuleOp> module = source.clone();
  MLIRContext *context = module->getContext();
  registerCodegenInterfaces(*context);
  PassManager manager(context);
  manager.addPass(swage::createFixedBlockToHostPass(blockSize));
  if (failed(manager.run(*module)))
    return nullptr;

  FailureOr<swage::KernelContract> contract =
      extractHostKernelContract(source, *module, kernelName);
  if (failed(contract))
    return nullptr;
  contractJSON = swage::serializeKernelContractJSON(*contract);
  if (failed(lowerHostToLLVM(*module)))
    return nullptr;
  printLoweredModule(*module, lowered);

  initializeNativeTarget();
  llvm::Expected<std::unique_ptr<ExecutionEngine>> engine =
      ExecutionEngine::create(module->getOperation());
  if (!engine) {
    source.emitError("failed to create native execution engine: ")
        << llvm::toString(engine.takeError());
    return nullptr;
  }
  (*engine)->initialize();
  llvm::Expected<void (*)(void **)> packed =
      (*engine)->lookupPacked(contract->entry);
  if (!packed) {
    source.emitError("failed to resolve packed host entry: ")
        << llvm::toString(packed.takeError());
    return nullptr;
  }

  return std::make_unique<HostExecutable>(
      HostExecutable{std::move(*engine), contract->entry,
                     static_cast<intptr_t>(contract->arguments.size())});
}

} // namespace

SwageHostExecutable swageCompileFixedBlockToHost(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback contractCallback, void *contractUserData) {
  SwageHostExecutable result{nullptr};
  if (!loweredCallback || !contractCallback)
    return result;
  std::string lowered;
  std::string contract;
  std::unique_ptr<HostExecutable> executable = compileHost(
      unwrap(module), unwrap(kernelName), blockSize, lowered, contract);
  if (!executable)
    return result;
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  contractCallback(wrap(llvm::StringRef(contract)), contractUserData);
  result.ptr = executable.release();
  return result;
}

bool swageHostExecutableIsNull(SwageHostExecutable executable) {
  return executable.ptr == nullptr;
}

MlirLogicalResult swageHostExecutableInvoke(SwageHostExecutable executable,
                                            void **arguments,
                                            intptr_t argumentCount,
                                            SwageStringCallback errorCallback,
                                            void *errorUserData) {
  if (!errorCallback)
    return mlirLogicalResultFailure();
  auto report = [&](llvm::StringRef message) {
    errorCallback(wrap(message), errorUserData);
    return mlirLogicalResultFailure();
  };
  if (!executable.ptr)
    return report("host executable is null");
  auto *owned = static_cast<HostExecutable *>(executable.ptr);
  if (argumentCount != owned->argumentCount)
    return report("host executable argument count does not match contract");
  if (argumentCount < 0 || (argumentCount != 0 && !arguments))
    return report("host executable arguments are invalid");

  llvm::MutableArrayRef<void *> packed(arguments,
                                       static_cast<size_t>(argumentCount));
  if (llvm::Error error = owned->engine->invokePacked(owned->entry, packed)) {
    std::string message = llvm::toString(std::move(error));
    return report(message);
  }
  return mlirLogicalResultSuccess();
}

void swageHostExecutableDestroy(SwageHostExecutable executable) {
  delete static_cast<HostExecutable *>(executable.ptr);
}

MlirLogicalResult swageCompileFixedBlockToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData) {
  if (!loweredCallback || !ptxCallback || !contractCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  std::string contract;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), blockSize,
                        unwrap(target), KernelKind::FixedBlock, false, false,
                        lowered, ptx, contract)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  contractCallback(wrap(llvm::StringRef(contract)), contractUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, int64_t blockSize,
    MlirStringRef target, bool useTaskIds, SwageStringCallback loweredCallback,
    void *loweredUserData, SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData) {
  if (!loweredCallback || !ptxCallback || !contractCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  std::string contract;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), blockSize,
                        unwrap(target), KernelKind::SegmentedReduction,
                        useTaskIds, false, lowered, ptx, contract)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  contractCallback(wrap(llvm::StringRef(contract)), contractUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileFusedSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData) {
  if (!loweredCallback || !ptxCallback || !contractCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  std::string contract;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 128, unwrap(target),
                        KernelKind::SegmentedReduction, true, true, lowered,
                        ptx, contract)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  contractCallback(wrap(llvm::StringRef(contract)), contractUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompilePersistentSegmentedReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData) {
  if (!loweredCallback || !ptxCallback || !contractCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  std::string contract;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 512, unwrap(target),
                        KernelKind::PersistentSegmentedReduction, false, false,
                        lowered, ptx, contract)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  contractCallback(wrap(llvm::StringRef(contract)), contractUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileSplitPartialReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData) {
  if (!loweredCallback || !ptxCallback || !contractCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  std::string contract;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 512, unwrap(target),
                        KernelKind::SplitPartialReduction, false, false,
                        lowered, ptx, contract)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  contractCallback(wrap(llvm::StringRef(contract)), contractUserData);
  return mlirLogicalResultSuccess();
}

MlirLogicalResult swageCompileSplitMergeReductionToPTX(
    MlirModule module, MlirStringRef kernelName, MlirStringRef target,
    SwageStringCallback loweredCallback, void *loweredUserData,
    SwageStringCallback ptxCallback, void *ptxUserData,
    SwageStringCallback contractCallback, void *contractUserData) {
  if (!loweredCallback || !ptxCallback || !contractCallback)
    return mlirLogicalResultFailure();
  std::string lowered;
  std::string ptx;
  std::string contract;
  if (failed(compilePTX(unwrap(module), unwrap(kernelName), 512, unwrap(target),
                        KernelKind::SplitMergeReduction, false, false, lowered,
                        ptx, contract)))
    return mlirLogicalResultFailure();
  loweredCallback(wrap(llvm::StringRef(lowered)), loweredUserData);
  ptxCallback(wrap(llvm::StringRef(ptx)), ptxUserData);
  contractCallback(wrap(llvm::StringRef(contract)), contractUserData);
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
