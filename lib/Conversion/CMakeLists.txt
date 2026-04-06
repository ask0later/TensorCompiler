add_library(tc-conversion MLIRBuilder.cpp ConvertNNDialectToLinalg.cpp LLVMLowering.cpp)

target_link_libraries(tc-conversion
    tc-graph
    tc-dialect
    onnx
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRTransforms
    MLIRSupport

    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRLinalgDialect
    MLIRSCFDialect
    MLIRMemRefDialect
    MLIRBufferizationDialect
    MLIRControlFlowDialect
    MLIRLLVMDialect

    MLIRTargetLLVMIRExport

    MLIRArithToLLVMConversion
    MLIRFuncToLLVMConversion
    MLIRControlFlowToLLVMConversion
    MLIRMemRefToLLVMConversion
    MLIRSCFToControlFlow
    MLIRLinalgTransforms

    LLVMCore
    LLVMSupport)

target_include_directories(tc-conversion PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}/../Dialect
    )