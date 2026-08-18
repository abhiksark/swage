set(LIBS
  MLIRSwage
  MLIROptLib
  MLIRRegisterAllPasses
  MLIRFuncDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRSCFDialect
  MLIRMemRefDialect
)
add_llvm_executable(swage-opt swage-opt.cpp)

llvm_update_compile_flags(swage-opt)
target_link_libraries(swage-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(swage-opt)
