cmake_minimum_required(VERSION 3.20.0)
project(swage LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(SWAGE_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SWAGE_BINARY_DIR ${PROJECT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${SWAGE_SOURCE_DIR}/include)
include_directories(${SWAGE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/swage/Dialect/Swage/IR)
add_subdirectory(include/swage/Dialect/SwagePlan/IR)
add_subdirectory(lib/Dialect/Swage/IR)
add_subdirectory(lib/Dialect/SwagePlan/IR)
add_subdirectory(lib/CAPI)
add_subdirectory(lib/Conversion)
add_subdirectory(tools/swage-opt)
add_subdirectory(test)

# Python bindings: default to whatever the MLIR install supports; an
# explicit ON against a non-bindings install is an error, never a skip.
set(SWAGE_PYTHON_BINDINGS "${MLIR_ENABLE_BINDINGS_PYTHON}" CACHE STRING
    "Build the mlir_swage Python bindings")
if(SWAGE_PYTHON_BINDINGS)
  if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
    message(FATAL_ERROR
      "SWAGE_PYTHON_BINDINGS=ON but the MLIR at ${MLIR_DIR} was built "
      "without MLIR_ENABLE_BINDINGS_PYTHON. Rebuild the pin with "
      "SWAGE_LLVM_PYTHON_BINDINGS=ON (scripts/build_llvm.sh).")
  endif()
  message(STATUS "Swage Python bindings: ON (package mlir_swage)")
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
  set(MLIR_PYTHON_PACKAGE_PREFIX "mlir_swage" CACHE STRING "" FORCE)
  set(MLIR_BINDINGS_PYTHON_NB_DOMAIN "mlir_swage" CACHE STRING "" FORCE)
  set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/mlir_swage"
      CACHE STRING "" FORCE)
  # No type stubs at this stage (ADR-0009); they return with wheels.
  set(MLIR_PYTHON_STUBGEN_ENABLED OFF CACHE BOOL "" FORCE)
  add_subdirectory(python)
else()
  message(STATUS "Swage Python bindings: OFF")
endif()
