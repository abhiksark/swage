include(AddMLIRPython)

# Build-tree packages retain checkout identity too. Wheel builds must supply
# this information explicitly, including when building from an sdist.
if(NOT SWAGE_WHEEL_BUILD)
  find_package(Git REQUIRED)
  execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE SWAGE_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE SWAGE_GIT_STATUS OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  if(SWAGE_GIT_STATUS STREQUAL "")
    set(SWAGE_SOURCE_CLEAN true)
  else()
    set(SWAGE_SOURCE_CLEAN false)
  endif()
endif()
file(STRINGS "${PROJECT_SOURCE_DIR}/python/swage/__init__.py"
     SWAGE_VERSION_LINE REGEX "^__version__ = ")
string(REGEX REPLACE "^__version__ = \"([^\"]+)\"$" "\\1"
       SWAGE_PACKAGE_VERSION "${SWAGE_VERSION_LINE}")
set(SWAGE_PYTHON_ROOT
    "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")
configure_file(mlir_swage/_build_info.json.in
               "${SWAGE_PYTHON_ROOT}/_build_info.json" @ONLY)
install(FILES "${SWAGE_PYTHON_ROOT}/_build_info.json"
        DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
        COMPONENT SwagePythonModules)

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

if(SWAGE_WHEEL_BUILD)
  set(SWAGE_PYTHON_TEST_ROOT "${MLIR_BINARY_DIR}")
else()
  set(SWAGE_PYTHON_TEST_ROOT "${MLIR_BINARY_DIR}/python_packages")
endif()

add_custom_target(check-swage-python
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${SWAGE_PYTHON_TEST_ROOT}"
    "${Python3_EXECUTABLE}" -m pytest -q
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS SwagePythonModules
)
