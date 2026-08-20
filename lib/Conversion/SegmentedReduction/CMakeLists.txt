# lib/Conversion/SegmentedReduction/CMakeLists.txt
add_mlir_conversion_library(MLIRSwageSegmentedReduction
  SegmentedReduction.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Conversion/SegmentedReduction

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSwage
)
