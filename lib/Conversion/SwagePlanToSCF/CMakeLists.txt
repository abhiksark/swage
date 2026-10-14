# lib/Conversion/SwagePlanToSCF/CMakeLists.txt
add_mlir_conversion_library(MLIRSwagePlanToSCF
  SwagePlanToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Conversion/SwagePlanToSCF

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSwage
  MLIRSwagePlan
  MLIRSwagePlanToGPU
  MLIRTransformUtils
)
