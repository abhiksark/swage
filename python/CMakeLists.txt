include(AddMLIRPython)

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

# The extension records the swage version and the source revision it was
# built from. The header is regenerated on every build, and rewritten only
# when one of the two changed, so a new commit recompiles one file.
set(SWAGE_IDENTITY_HEADER
  "${SWAGE_BINARY_DIR}/include/swage/Python/BuildIdentity.h")
add_custom_target(SwageBuildIdentity
  COMMAND ${CMAKE_COMMAND}
    "-DSWAGE_SOURCE_DIR=${SWAGE_SOURCE_DIR}"
    "-DSWAGE_IDENTITY_HEADER=${SWAGE_IDENTITY_HEADER}"
    -P "${SWAGE_SOURCE_DIR}/cmake/SwageBuildIdentity.cmake"
  BYPRODUCTS "${SWAGE_IDENTITY_HEADER}"
  COMMENT "Recording the Swage build identity"
  VERBATIM
)
add_dependencies(SwagePythonModules.extension._swageDialectsNanobind.dso
  SwageBuildIdentity)

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

add_custom_target(check-swage-python
  COMMAND ${CMAKE_COMMAND} -E env
    "PYTHONPATH=${MLIR_BINARY_DIR}/python_packages"
    "${Python3_EXECUTABLE}" -m pytest -q
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS SwagePythonModules
)
