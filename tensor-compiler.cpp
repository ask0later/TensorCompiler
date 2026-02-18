#include "TensorCompiler/Frontend/ONNXModel.hpp"
#include "TensorCompiler/Frontend/ONNXDumper.hpp"
#include <iostream>

int main(const int argc, const char *const argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }

    try {
        auto model = ONNXModel::QueryONNXModel(argv[1]);
        DumpVisitor dumper{std::cout};
        model.Parse(dumper);
    } catch (const std::exception &e) {
        std::cerr << "Compilation Failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}