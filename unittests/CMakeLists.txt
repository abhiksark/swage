# unittests/CMakeLists.txt
add_custom_target(check-swage-unit
  COMMAND $<TARGET_FILE:SwageUnitTests>
  COMMENT "Running the Swage unit tests"
)
set_target_properties(check-swage-unit PROPERTIES FOLDER "Tests")

add_unittest(check-swage-unit SwageUnitTests
  TaskClassifierTest.cpp
)

target_link_libraries(SwageUnitTests
  PRIVATE
  MLIRSwagePlan
)
