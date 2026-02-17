cmake_minimum_required(VERSION 3.27)
project(TensorCompiler)

if (NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "CMAKE_TOOLCHAIN_FILE is not set. Run: conan install . --output-folder=build")
endif()

find_package(Protobuf CONFIG REQUIRED)
find_package(ONNX CONFIG REQUIRED)
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_compile_options(-fno-rtti) # llvm-project does not use RTTI

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
add_subdirectory(lib)

add_executable(tensor-compiler tensor-compiler.cpp)
target_link_libraries(tensor-compiler PRIVATE tc-core)