add_executable(onnx2mlir onnx2mlir.cpp)

target_link_libraries(onnx2mlir gtest::gtest tc-core)
target_include_directories(onnx2mlir PRIVATE ${GMOCK_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})

set(RUN_UNIT_TESTS ./onnx2mlir --gtest_color=yes)
add_test(
    NAME ONNX-to-MLIR
    COMMAND $<TARGET_FILE:onnx2mlir>
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../..)