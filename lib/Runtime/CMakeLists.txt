# lib/Runtime/CMakeLists.txt
# The library a process loads to run kernels that were compiled elsewhere.
# It is plain C and must not link against LLVM or MLIR, so it is a bare
# library target and not one of the MLIR library macros.
add_library(SwageRuntime SHARED SwageRuntime.c)
set_target_properties(SwageRuntime PROPERTIES
  C_STANDARD 11
  C_STANDARD_REQUIRED ON
  C_VISIBILITY_PRESET hidden
  LIBRARY_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}")
target_link_libraries(SwageRuntime PRIVATE ${CMAKE_DL_LIBS})
# The header installs with the other C API headers, so the library goes
# with it. It is not part of the exported CMake targets.
install(TARGETS SwageRuntime LIBRARY DESTINATION lib)
