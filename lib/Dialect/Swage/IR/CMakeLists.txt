add_mlir_dialect_library(MLIRSwage
  SwageDialect.cpp
  SwageOps.cpp
  SwageTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Dialect/Swage/IR

  DEPENDS
  MLIRSwageOpsIncGen
  MLIRSwageAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
