add_library(tc-converter MLIRBuilder.cpp GraphBuilder.cpp)

target_link_libraries(tc-converter
    onnx)

target_include_directories(tc-converter PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../Dialect
    )