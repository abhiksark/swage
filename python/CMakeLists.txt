include(AddMLIRPython)

# All MLIR packages are co-located under the mlir_swage top-level package
# (the API is embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

declare_mlir_python_sources(SwagePythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT SwagePythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_swage"
  TD_FILE dialects/SwageOps.td
  SOURCES dialects/swage.py
  DIALECT_NAME swage)

declare_mlir_python_extension(SwagePythonSources.NanobindExtension
  MODULE_NAME _swageDialectsNanobind
  ADD_TO_PARENT SwagePythonSources
  SOURCES
    SwageExtensionNanobind.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIFunc
    MLIRCAPIArith
    MLIRCAPIMath
    MLIRCAPIMemRef
    MLIRCAPIVector
    MLIRCAPITransforms
    SwageCAPI
)

add_mlir_python_common_capi_library(SwagePythonCAPI
  INSTALL_COMPONENT SwagePythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    SwagePythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.func
    MLIRPythonSources.Dialects.arith
    MLIRPythonSources.Dialects.math
    MLIRPythonSources.Dialects.memref
    MLIRPythonSources.Dialects.vector
)

add_mlir_python_modules(SwagePythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    SwagePythonSources
    MLIRPythonSources.Core
    MLIRPythonSources.Dialects.builtin
    MLIRPythonSources.Dialects.func
    MLIRPythonSources.Dialects.arith
    MLIRPythonSources.Dialects.math
    MLIRPythonSources.Dialects.memref
    MLIRPythonSources.Dialects.vector
  COMMON_CAPI_LINK_LIBS
    SwagePythonCAPI
)

add_custom_target(check-swage-python
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${MLIR_BINARY_DIR}/python_packages"
    "${Python3_EXECUTABLE}" -m pytest -q
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS SwagePythonModules
)
