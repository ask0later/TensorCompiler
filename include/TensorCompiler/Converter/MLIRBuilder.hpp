#pragma once
#include "TensorCompiler/Frontend/ONNXVisitor.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <map>
#include <set>
#include <string>

namespace tc::converter::onnx_to_high_mlir {
class MLIRBuilder final : public tc::frontend::ONNXVisitor {
public:
  explicit MLIRBuilder(mlir::MLIRContext &ctx);

  const mlir::ModuleOp &GetModule() const;

  void Visit(const onnx::ModelProto &) override;
  void Visit(const onnx::GraphProto &graph) override;
  void Visit(const onnx::ValueInfoProto &) override;
  void Visit(const onnx::TensorProto &tensor) override;
  void Visit(const onnx::NodeProto &node) override;
  void Visit(const onnx::AttributeProto &) override;

  void Finalize(const onnx::GraphProto &graph) override;

private:
  mlir::Value FindValue(const std::string &name) const;

  mlir::Type ConvertElemType(int onnx_dtype);
  mlir::RankedTensorType ConvertTensorType(int onnx_dtype,
                                           llvm::ArrayRef<int64_t> shape);

  mlir::DenseElementsAttr ConvertTensor(const onnx::TensorProto &tensor);
  mlir::RankedTensorType ConvertValueInfo(const onnx::ValueInfoProto &info);

  void BuildRelu(const onnx::NodeProto &node, mlir::Location loc);
  void BuildAdd(const onnx::NodeProto &node, mlir::Location loc);
  void BuildMul(const onnx::NodeProto &node, mlir::Location loc);
  void BuildMatMul(const onnx::NodeProto &node, mlir::Location loc);
  void BuildGemm(const onnx::NodeProto &node, mlir::Location loc);
  void BuildConv(const onnx::NodeProto &node, mlir::Location loc);
  void BuildTranspose(const onnx::NodeProto &node, mlir::Location loc);

private:
  mlir::MLIRContext &ctx_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;

  std::map<std::string, mlir::Value> valueMap_;
  std::set<std::string> initializerNames_;
};
} // namespace tc::converter::onnx_to_high_mlir