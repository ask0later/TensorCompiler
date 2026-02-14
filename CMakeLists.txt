cmake_minimum_required(VERSION 3.27)
project(TensorCompiler)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED 20)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "CMAKE_TOOLCHAIN_FILE is not set. Run: conan install . --output-folder=build")
endif()

find_package(Protobuf CONFIG REQUIRED)
find_package(ONNX CONFIG REQUIRED)

add_subdirectory(lib)

add_executable(tensor-compiler tensor-compiler.cpp)
target_link_libraries(tensor-compiler PRIVATE tc-core)