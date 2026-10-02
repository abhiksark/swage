# lib/Conversion/SwagePlanToGPU/CMakeLists.txt
add_mlir_conversion_library(MLIRSwagePlanToGPU
  Emission.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Conversion/SwagePlanToGPU

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSwage
  MLIRSwageTarget
)
