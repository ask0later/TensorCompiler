#include "TensorCompiler/Frontend/ONNXParser.hpp"
#include <iostream>

int main(const int argc, const char *const argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
        return 1;
    }

    try {
        auto parser = ONNXParser::QueryONNXParser(argv[1]);
        parser.ParseModel();
    } catch (const std::exception &e) {
        std::cerr << "Compilation Failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}