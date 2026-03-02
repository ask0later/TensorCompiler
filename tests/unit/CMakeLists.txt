add_executable(unit-tests onnx2mlir.cpp graph.cpp onnx2graph.cpp)

target_link_libraries(unit-tests gtest::gtest tc-core)
target_include_directories(unit-tests PRIVATE ${GMOCK_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

set(RUN_UNIT_TESTS ./unit-tests --gtest_color=yes)
add_test(
    NAME UNIT_TESTS
    COMMAND $<TARGET_FILE:unit-tests>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)