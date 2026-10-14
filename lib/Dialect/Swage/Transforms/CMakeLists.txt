# lib/Dialect/Swage/Transforms/CMakeLists.txt
add_mlir_dialect_library(MLIRSwageTransforms
  FuseMaps.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Dialect/Swage/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRSwage
  MLIRTransformUtils
)
