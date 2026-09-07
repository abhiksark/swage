# lib/Conversion/SegmentedReduction/CMakeLists.txt
add_mlir_conversion_library(MLIRSwageSegmentedReduction
  SegmentProgram.cpp
  Sequential.cpp
  GPU.cpp
  Split.cpp
  Passes.cpp

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
  MLIRSwagePlan
  MLIRSwageSupport
)
