add_library(tc-frontend ONNXParser.cpp)

target_link_libraries(tc-frontend
    onnx
    onnx_proto
    protobuf::libprotobuf)

target_include_directories(tc-frontend PUBLIC
    ${PROJECT_SOURCE_DIR}/include)