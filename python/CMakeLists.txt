include(AddMLIRPython)

# The packaged build record, written with the compiled-in identity below.
set(SWAGE_PYTHON_ROOT
    "${MLIR_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")
set(SWAGE_BUILD_INFO "${SWAGE_PYTHON_ROOT}/_build_info.json")
install(FILES "${SWAGE_BUILD_INFO}"
        DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
        COMPONENT SwagePythonModules)

# All MLIR packages are co-located under the mlir_swage top-level package
# (the API is embedded in a relocatable way).
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

# The disk cache of `swage` identifies each native library by its ELF build
# id, which the linker derives from the linked contents. Some toolchains
# emit none unless asked.
if(UNIX AND NOT APPLE)
  add_link_options("LINKER:--build-id=sha1")
endif()

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

# The extension records the swage version, the source revision, and the
# digest of the frontend sources it was built from, and the package records
# the same facts in _build_info.json. Both are regenerated on every build
# and rewritten only when they change, so a new commit or a frontend edit
# recompiles one file. A wheel build has no git checkout and passes the
# revision it was configured with.
set(SWAGE_IDENTITY_HEADER
  "${SWAGE_BINARY_DIR}/include/swage/Python/BuildIdentity.h")
set(SWAGE_IDENTITY_SOURCE_ARGUMENTS)
if(SWAGE_WHEEL_BUILD)
  list(APPEND SWAGE_IDENTITY_SOURCE_ARGUMENTS
    "-DSWAGE_SOURCE_REVISION=${SWAGE_SOURCE_REVISION}"
    "-DSWAGE_SOURCE_CLEAN=${SWAGE_SOURCE_CLEAN}")
endif()
add_custom_target(SwageBuildIdentity
  COMMAND ${CMAKE_COMMAND}
    "-DSWAGE_SOURCE_DIR=${SWAGE_SOURCE_DIR}"
    "-DSWAGE_IDENTITY_HEADER=${SWAGE_IDENTITY_HEADER}"
    "-DSWAGE_BUILD_INFO=${SWAGE_BUILD_INFO}"
    "-DSWAGE_LLVM_TAG=${SWAGE_LLVM_TAG}"
    "-DSWAGE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    ${SWAGE_IDENTITY_SOURCE_ARGUMENTS}
    -P "${SWAGE_SOURCE_DIR}/cmake/SwageBuildIdentity.cmake"
  BYPRODUCTS "${SWAGE_IDENTITY_HEADER}" "${SWAGE_BUILD_INFO}"
  COMMENT "Recording the Swage build identity"
  VERBATIM
)
add_dependencies(SwagePythonModules.extension._swageDialectsNanobind.dso
  SwageBuildIdentity)
add_dependencies(SwagePythonModules SwageBuildIdentity)

# The runtime library travels with the bindings. `python -m swage.compile`
# copies it from this package into every artifact it writes, so a native
# wheel carries it too.
set(SWAGE_PYTHON_LIBRARY_DIR
  "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs")
add_custom_target(SwageRuntimeInPackage
  COMMAND ${CMAKE_COMMAND} -E make_directory
    "${MLIR_BINARY_DIR}/${SWAGE_PYTHON_LIBRARY_DIR}"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "$<TARGET_FILE:SwageRuntime>"
    "${MLIR_BINARY_DIR}/${SWAGE_PYTHON_LIBRARY_DIR}/"
  DEPENDS SwageRuntime
  COMMENT "Placing the Swage runtime library beside the bindings"
  VERBATIM
)
add_dependencies(SwagePythonModules SwageRuntimeInPackage)
install(TARGETS SwageRuntime
  LIBRARY DESTINATION "${SWAGE_PYTHON_LIBRARY_DIR}"
  COMPONENT SwagePythonModules)

if(SWAGE_WHEEL_BUILD)
  set(SWAGE_PYTHON_TEST_ROOT "${MLIR_BINARY_DIR}")
else()
  set(SWAGE_PYTHON_TEST_ROOT "${MLIR_BINARY_DIR}/python_packages")
endif()

# The private CPU oracle of the segmented tests runs bin/swage-opt of a
# build directory. This target names its own, wherever it is, instead of
# leaving the oracle to look for `build` in the checkout.
add_custom_target(check-swage-python
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}:${SWAGE_PYTHON_TEST_ROOT}"
    "SWAGE_ORACLE_BUILD_DIR=${CMAKE_BINARY_DIR}"
    "${Python3_EXECUTABLE}" -m pytest -q
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS SwagePythonModules
)
