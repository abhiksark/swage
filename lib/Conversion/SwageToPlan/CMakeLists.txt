# lib/Conversion/SwageToPlan/CMakeLists.txt
add_mlir_conversion_library(MLIRSwageToPlan
  Admission.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Conversion/SwageToPlan

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSwage
)
