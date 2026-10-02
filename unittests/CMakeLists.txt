# unittests/CMakeLists.txt
add_custom_target(check-swage-unit
  COMMAND $<TARGET_FILE:SwageUnitTests>
  COMMENT "Running the Swage unit tests"
)
set_target_properties(check-swage-unit PROPERTIES FOLDER "Tests")

add_unittest(check-swage-unit SwageUnitTests
  CodegenCAPITest.cpp
  DialectsCAPITest.cpp
  TargetDescriptionTest.cpp
  TaskClassifierTest.cpp
)

# The C API tests reach Swage only through SwageCAPI and the MLIR C API, as a
# C embedder would. MLIRCAPIGPU is for one test that parses a gpu.module. The
# target description test lowers a module through the C++ pass factory.
target_link_libraries(SwageUnitTests
  PRIVATE
  MLIRCAPIArith
  MLIRCAPIFunc
  MLIRCAPIGPU
  MLIRCAPIIR
  MLIRCAPIMath
  MLIRCAPIMemRef
  MLIRCAPIVector
  MLIRParser
  MLIRSwagePlan
  MLIRSwageSegmentedReduction
  MLIRSwageTarget
  SwageCAPI
)
