#pragma once
#include "TensorCompiler/Frontend/ONNXVisitor.hpp"
#include "TensorCompiler/Graph/Graph.hpp"

#include <set>
#include <string>
#include <vector>

namespace tc::frontend {
using tc::graph::AttrValue;
using tc::graph::EntityId;
using tc::graph::Graph;
using tc::graph::TensorData;

class GraphBuilder final : public tc::frontend::ONNXVisitor {
  Graph graph_;
  std::set<std::string> initializerNames_;
  std::set<std::string> inputNames_;
  std::set<std::string> outputNames_;

  TensorData ParseTensor(const onnx::TensorProto &tensor);
  AttrValue ParseAttribute(const onnx::AttributeProto &attr);

  EntityId EnsureTensor(const std::string &name,
                        const std::vector<int64_t> &shape, int32_t dtype);

public:
  const Graph &GetGraph() const { return graph_; }

  void Visit(const onnx::ModelProto &) override;
  void Visit(const onnx::GraphProto &graph) override;
  void Visit(const onnx::ValueInfoProto &value) override;
  void Visit(const onnx::TensorProto &tensor) override;
  void Visit(const onnx::NodeProto &node) override;
  void Visit(const onnx::AttributeProto &) override;

  void Finalize(const onnx::GraphProto &) override;
};
} // namespace tc::frontend