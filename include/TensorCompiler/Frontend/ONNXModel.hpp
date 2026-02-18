#pragma once
#include "TensorCompiler/Frontend/ONNXVisitor.hpp"
#include <fstream>

class ONNXModel final {
    const std::string_view file_name_;
    onnx::ModelProto model_;

    ONNXModel(const std::string_view file_name) : file_name_(file_name) {
        std::ifstream input_model(file_name_.data(), std::ios::binary);
        if (!input_model.good()) {
            std::string err = std::string("Failed to open file: ") + std::string(file_name_);
            throw std::runtime_error(err);
        }

        model_.ParseFromIstream(&input_model);
    }

    void Parse(ONNXVisitor &visitor, const onnx::GraphProto &graph) {
        visitor.Visit(graph);
        
        for (auto &&input : graph.input())
            visitor.Visit(input);

        for (auto &&output : graph.output())
            visitor.Visit(output);

        for (auto &&tensor : graph.initializer())
            visitor.Visit(tensor);
        
        for (auto &&node : graph.node())
            Parse(visitor, node);
    }

    void Parse(ONNXVisitor &visitor, const onnx::NodeProto &node) {
        visitor.Visit(node);

        for (auto &&attr : node.attribute())
            visitor.Visit(attr);
    }

public:
    static ONNXModel &QueryONNXModel(const std::string_view file_name) {
        static ONNXModel parser{file_name};
        return parser;
    }

    void Parse(ONNXVisitor &visitor) {
        visitor.Visit(model_);

        if (model_.has_graph())
            Parse(visitor, model_.graph());
        else
            throw std::runtime_error("Error: Model has no graph.");

        visitor.Finalize(model_.graph());
    }
};