# lib/Support/CMakeLists.txt
add_mlir_library(MLIRSwageSupport
  KernelContract.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Support

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
