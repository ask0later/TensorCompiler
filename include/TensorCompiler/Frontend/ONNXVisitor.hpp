#pragma once
#include <onnx/onnx_pb.h>

namespace tc::frontend {
class ONNXVisitor {
public:
  virtual ~ONNXVisitor() = default;
  virtual void Visit(const onnx::ModelProto &model) = 0;
  virtual void Visit(const onnx::GraphProto &graph) = 0;
  virtual void Visit(const onnx::TensorProto &tensor) = 0;
  virtual void Visit(const onnx::NodeProto &node) = 0;
  virtual void Visit(const onnx::AttributeProto &attr) = 0;
  virtual void Visit(const onnx::ValueInfoProto &info) = 0;
  virtual void Finalize(const onnx::GraphProto &graph) = 0;
};
} // namespace tc::frontend