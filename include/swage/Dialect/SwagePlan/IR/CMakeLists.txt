# include/swage/Dialect/SwagePlan/IR/CMakeLists.txt
add_mlir_dialect(SwagePlanOps swage_plan)
add_mlir_doc(SwagePlanDialect SwagePlanDialect swage_plan/ -gen-dialect-doc)
add_mlir_doc(SwagePlanOps SwagePlanOps swage_plan/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS SwagePlanOps.td)
mlir_tablegen(SwagePlanEnums.h.inc -gen-enum-decls)
mlir_tablegen(SwagePlanEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(SwagePlanAttributes.h.inc -gen-attrdef-decls
              -attrdefs-dialect=swage_plan)
mlir_tablegen(SwagePlanAttributes.cpp.inc -gen-attrdef-defs
              -attrdefs-dialect=swage_plan)
add_mlir_generic_tablegen_target(MLIRSwagePlanAttributesIncGen)
