add_subdirectory(Dialect)
add_subdirectory(Frontend)
add_subdirectory(Graph)
add_subdirectory(Converter)

add_library(tc-core INTERFACE)

target_link_libraries(tc-core
    INTERFACE
        tc-dialect
        tc-frontend
        tc-graph
        tc-converter
        MLIRParser
        MLIRFuncDialect
        MLIRArithDialect
        )