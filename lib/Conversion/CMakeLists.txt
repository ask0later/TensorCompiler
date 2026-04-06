add_library(tc-conversion MLIRBuilder.cpp)

target_link_libraries(tc-conversion
    tc-graph
    tc-dialect
    onnx)

target_include_directories(tc-conversion PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../Dialect
    )