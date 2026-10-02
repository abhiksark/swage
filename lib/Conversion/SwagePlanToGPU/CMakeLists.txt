# lib/Conversion/SwagePlanToGPU/CMakeLists.txt
add_mlir_conversion_library(MLIRSwagePlanToGPU
  Emission.cpp
  SwagePlanToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Conversion/SwagePlanToGPU

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSwage
  MLIRSwagePlan
  MLIRSwageTarget
  MLIRSwageToPlan
  MLIRTransformUtils
)
