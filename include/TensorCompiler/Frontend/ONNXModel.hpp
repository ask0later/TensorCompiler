#pragma once
#include "TensorCompiler/Frontend/ONNXVisitor.hpp"
#include <fstream>

class ONNXModel final {
  const std::string_view fileName_;
  onnx::ModelProto model_;

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
  ONNXModel(const std::string_view fileName) : fileName_(fileName) {
    std::ifstream input_model(fileName_.data(), std::ios::binary);
    if (!input_model.good())
      throw std::runtime_error(std::string("Failed to open file: ") +
                               std::string(fileName_));

    if (!model_.ParseFromIstream(&input_model))
      throw std::runtime_error(std::string("Failed to parse ONNX model: ") +
                               std::string(fileName_));
  }

  void Parse(ONNXVisitor &visitor) {
    visitor.Visit(model_);

    if (!model_.has_graph())
      throw std::runtime_error("Error: Model has no graph.");

    Parse(visitor, model_.graph());
    visitor.Finalize(model_.graph());
  }
};