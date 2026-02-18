add_subdirectory(IR)
add_subdirectory(Dialect)
add_subdirectory(Frontend)
add_subdirectory(Driver)

add_library(tc-core INTERFACE)

target_link_libraries(tc-core
    INTERFACE
        tc-ir
        tc-driver
        tc-dialect
        tc-frontend
        MLIRParser
        MLIRFuncDialect
        MLIRArithDialect
        )