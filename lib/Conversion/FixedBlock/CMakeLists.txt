add_mlir_conversion_library(MLIRSwageFixedBlock
  Analysis.cpp
  GPU.cpp
  Host.cpp
  Passes.cpp
  Scalar.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Conversion/FixedBlock

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
  MLIRSwageSupport
  MLIRVectorDialect
)
