# lib/Target/CMakeLists.txt
add_mlir_library(MLIRSwageTarget
  NVIDIATarget.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Target

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRNVVMDialect
)
