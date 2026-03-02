#pragma once
#include "TensorCompiler/Frontend/ONNXVisitor.hpp"
#include <ostream>

namespace tc::frontend::debug {

class DumpVisitor final : public tc::frontend::ONNXVisitor {
  std::ostream &os_;

public:
  explicit DumpVisitor(std::ostream &os);

  void Visit(const onnx::ModelProto &model) override;
  void Visit(const onnx::GraphProto &graph) override;
  void Visit(const onnx::TensorProto &tensor) override;
  void Visit(const onnx::NodeProto &node) override;
  void Visit(const onnx::AttributeProto &attr) override;
  void Visit(const onnx::ValueInfoProto &info) override;

  void Finalize(const onnx::GraphProto &graph) override;
};
} // namespace tc::frontend::debug