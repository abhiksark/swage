add_mlir_conversion_library(MLIRSwageFixedBlockToGPU
  FixedBlockToGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Conversion/FixedBlockToGPU

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSwage
  MLIRVectorDialect
)
