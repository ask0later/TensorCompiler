add_subdirectory(IR)
add_subdirectory(Dialect)
add_subdirectory(Frontend)

add_library(tc-core INTERFACE)

target_link_libraries(tc-core
    INTERFACE
        tc-ir
        tc-dialect
        tc-frontend
        MLIRParser
        MLIRFuncDialect
        )