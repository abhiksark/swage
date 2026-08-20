# lib/CAPI/CMakeLists.txt
add_mlir_public_c_api_library(SwageCAPI
  Codegen.cpp
  Dialects.cpp

  LINK_COMPONENTS
  NVPTXCodeGen
  NVPTXDesc
  NVPTXInfo

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRGPUToLLVMIRTranslation
  MLIRGPUToNVVMTransforms
  MLIRIndexToLLVM
  MLIRLLVMToLLVMIRTranslation
  MLIRMathToLLVM
  MLIRMemRefToLLVM
  MLIRNVVMToLLVM
  MLIRNVVMToLLVMIRTranslation
  MLIRSCFToControlFlow
  MLIRSwage
  MLIRSwageFixedBlockToGPU
  MLIRSwageSegmentedReduction
  MLIRTargetLLVM
  MLIRUBToLLVM
  MLIRVectorToLLVM
)
