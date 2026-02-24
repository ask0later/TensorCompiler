add_library(tc-driver MLIRBuilder.cpp)

target_link_libraries(tc-driver
    onnx)

target_include_directories(tc-driver PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../Dialect
    )