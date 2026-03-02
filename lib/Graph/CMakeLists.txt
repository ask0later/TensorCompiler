add_library(tc-graph IR.cpp Exporter.cpp)

target_include_directories(tc-graph PUBLIC
    ${PROJECT_SOURCE_DIR}/include)