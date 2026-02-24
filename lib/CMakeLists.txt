add_subdirectory(Dialect)
add_subdirectory(Frontend)
add_subdirectory(Driver)

add_library(tc-core INTERFACE)

target_link_libraries(tc-core
    INTERFACE
        tc-dialect
        tc-frontend
        tc-driver
        MLIRParser
        MLIRFuncDialect
        MLIRArithDialect
        )