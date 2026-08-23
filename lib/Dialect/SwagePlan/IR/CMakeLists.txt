# lib/Dialect/SwagePlan/IR/CMakeLists.txt
add_mlir_dialect_library(MLIRSwagePlan
  SwagePlanDialect.cpp
  TaskClassifier.cpp

  ADDITIONAL_HEADER_DIRS
  ${SWAGE_SOURCE_DIR}/include/swage/Dialect/SwagePlan/IR

  DEPENDS
  MLIRSwagePlanOpsIncGen
  MLIRSwagePlanAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRIR
)
