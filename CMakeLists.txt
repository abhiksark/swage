# CMakeLists.txt
cmake_minimum_required(VERSION 3.20.0)

# Checked before project(), which would otherwise write compiler detection
# files into the source tree.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR
    "Swage does not support in-source builds: the build directory is the "
    "source directory ${CMAKE_SOURCE_DIR}.\n"
    "Configure into a separate directory:\n"
    "  cmake -S . -B build\n"
    "Then delete the CMakeCache.txt and CMakeFiles/ that this attempt left "
    "in the source directory.")
endif()

project(swage LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

# ADR-0004: Swage builds against exactly the release named in
# cmake/llvm-version.txt. Any other LLVM/MLIR install is rejected here
# instead of failing later in the build or passing with different behavior.
set(SWAGE_LLVM_PIN_FILE "${PROJECT_SOURCE_DIR}/cmake/llvm-version.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  "${SWAGE_LLVM_PIN_FILE}")
file(STRINGS "${SWAGE_LLVM_PIN_FILE}" SWAGE_LLVM_TAG LIMIT_COUNT 1)
string(STRIP "${SWAGE_LLVM_TAG}" SWAGE_LLVM_TAG)
if(NOT SWAGE_LLVM_TAG MATCHES "^llvmorg-([0-9]+\\.[0-9]+\\.[0-9]+)$")
  message(FATAL_ERROR "Invalid LLVM pin in cmake/llvm-version.txt")
endif()
set(SWAGE_LLVM_VERSION "${CMAKE_MATCH_1}")
find_package(MLIR ${SWAGE_LLVM_VERSION} EXACT CONFIG QUIET)
if(NOT MLIR_FOUND OR NOT LLVM_PACKAGE_VERSION STREQUAL SWAGE_LLVM_VERSION)
  message(FATAL_ERROR
    "Swage requires LLVM/MLIR ${SWAGE_LLVM_VERSION}, the release pinned "
    "as ${SWAGE_LLVM_TAG} in cmake/llvm-version.txt, but found LLVM "
    "'${LLVM_PACKAGE_VERSION}' at '${LLVM_DIR}'. Build the pinned release "
    "with scripts/fetch_llvm.sh and scripts/build_llvm.sh, or point MLIR_DIR "
    "and LLVM_DIR at the lib/cmake directories of an install of it.")
endif()

option(SWAGE_WHEEL_BUILD "Install a self-contained native wheel package" OFF)
option(SWAGE_ENABLE_SANITIZERS "Instrument Swage with ASan and UBSan" OFF)
if(SWAGE_WHEEL_BUILD)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR "SWAGE_WHEEL_BUILD requires CMAKE_BUILD_TYPE=Release")
  endif()
  string(LENGTH "${SWAGE_SOURCE_REVISION}" SWAGE_REVISION_LENGTH)
  if(NOT SWAGE_REVISION_LENGTH EQUAL 40 OR
     NOT SWAGE_SOURCE_REVISION MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
      "SWAGE_WHEEL_BUILD requires SWAGE_SOURCE_REVISION: 40 lowercase hex digits")
  endif()
  if(NOT DEFINED SWAGE_SOURCE_CLEAN OR
     NOT "${SWAGE_SOURCE_CLEAN}" MATCHES "^(true|false)$")
    message(FATAL_ERROR
      "SWAGE_WHEEL_BUILD requires explicit SWAGE_SOURCE_CLEAN=true or false")
  endif()
endif()

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

if(SWAGE_WHEEL_BUILD)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$" OR
     NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Native wheels require Linux GCC or Clang")
  endif()
  foreach(root IN ITEMS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
    add_compile_options("-ffile-prefix-map=${root}=/swage"
                        "-fdebug-prefix-map=${root}=/swage")
  endforeach()
  add_link_options("-Wl,--build-id=sha1")
  # Absolute build RUNPATHs change ELF layout and build IDs before stripping.
  # Use the same relocatable dependency paths at link and install time.
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
endif()

if(SWAGE_ENABLE_SANITIZERS)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux" OR
     NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$" OR
     NOT CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang)$")
    message(FATAL_ERROR "SWAGE_ENABLE_SANITIZERS requires Linux GCC or Clang")
  endif()
  # A finding stops the program, so a test cannot pass over one.
  add_compile_options(-fsanitize=address,undefined -fno-sanitize-recover=all
                      -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${SWAGE_SOURCE_DIR}/include)
include_directories(${SWAGE_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/swage/Dialect/Swage/IR)
add_subdirectory(include/swage/Dialect/SwagePlan/IR)
add_subdirectory(lib/Dialect/Swage/IR)
add_subdirectory(lib/Dialect/Swage/Transforms)
add_subdirectory(lib/Dialect/SwagePlan/IR)
add_subdirectory(lib/Target)
add_subdirectory(lib/Support)
add_subdirectory(lib/CAPI)
add_subdirectory(lib/Conversion)
add_subdirectory(lib/Runtime)
add_subdirectory(tools/swage-opt)
add_subdirectory(test)
add_subdirectory(unittests)

# Python bindings: default to whatever the MLIR install supports; an
# explicit ON against a non-bindings install is an error, never a skip.
set(SWAGE_PYTHON_BINDINGS "${MLIR_ENABLE_BINDINGS_PYTHON}" CACHE STRING
    "Build the mlir_swage Python bindings")
if(SWAGE_WHEEL_BUILD AND NOT SWAGE_PYTHON_BINDINGS)
  message(FATAL_ERROR "SWAGE_WHEEL_BUILD requires SWAGE_PYTHON_BINDINGS=ON")
endif()
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
  if(SWAGE_WHEEL_BUILD)
    set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "mlir_swage" CACHE STRING "" FORCE)
  else()
    set(MLIR_BINDINGS_PYTHON_INSTALL_PREFIX "python_packages/mlir_swage"
        CACHE STRING "" FORCE)
  endif()
  # The private binding API is not typed; public swage stubs ship separately.
  set(MLIR_PYTHON_STUBGEN_ENABLED OFF CACHE BOOL "" FORCE)
  add_subdirectory(python)
else()
  message(STATUS "Swage Python bindings: OFF")
endif()

# Install tree. The libraries install through the MLIR library macros, which
# also place them in the MLIRTargets export set. The rules below add the
# headers and the package that an out-of-tree project loads with
# `find_package(Swage CONFIG)`.
install(DIRECTORY include/swage include/swage-c
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.td")
install(DIRECTORY ${SWAGE_BINARY_DIR}/include/swage
  DESTINATION include
  FILES_MATCHING
  PATTERN "*.inc"
  PATTERN "CMakeFiles" EXCLUDE)
install(EXPORT MLIRTargets
  FILE SwageTargets.cmake
  DESTINATION lib/cmake/swage)
configure_file(cmake/SwageConfig.cmake.in
  ${SWAGE_BINARY_DIR}/lib/cmake/swage/SwageConfig.cmake @ONLY)
install(FILES ${SWAGE_BINARY_DIR}/lib/cmake/swage/SwageConfig.cmake
  DESTINATION lib/cmake/swage)
