set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/TensorCompiler/Dialect/NNDialect.td)

mlir_tablegen(NNDialect.h.inc -gen-dialect-decls -I ${PROJECT_SOURCE_DIR}/include)
mlir_tablegen(NNDialect.cpp.inc -gen-dialect-defs -I ${PROJECT_SOURCE_DIR}/include)
add_public_tablegen_target(MLIRNNDialectIncGen)

set(LLVM_TARGET_DEFINITIONS ${PROJECT_SOURCE_DIR}/include/TensorCompiler/Dialect/NNOps.td)
mlir_tablegen(NNOps.h.inc -gen-op-decls -I ${PROJECT_SOURCE_DIR}/include)
mlir_tablegen(NNOps.cpp.inc -gen-op-defs -I ${PROJECT_SOURCE_DIR}/include)
add_public_tablegen_target(MLIRNNOpsIncGen)

add_library(tc-dialect
    NNDialect.cpp
)

target_link_libraries(tc-dialect PUBLIC
    MLIRIR
    MLIRSupport
    )

target_include_directories(tc-dialect PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../Dialect
    )

add_dependencies(tc-dialect
    MLIRNNDialectIncGen
    MLIRNNOpsIncGen
)