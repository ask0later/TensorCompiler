add_library(tc-graph Graph.cpp GraphDumper.cpp)

target_include_directories(tc-graph PUBLIC
    ${PROJECT_SOURCE_DIR}/include)