#pragma once
#include "TensorCompiler/Frontend/ONNXVisitor.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

class MLIRBuilder final : public ONNXVisitor {
  mlir::MLIRContext &ctx_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;
  std::map<std::string, mlir::Value> valueMap_;
  std::set<std::string> initializerNames_;

  mlir::Value findValue(const std::string &name) const;

  mlir::Type convertElemType(int onnx_dtype);
  mlir::RankedTensorType convertTensorType(int onnx_dtype,
                                           llvm::ArrayRef<int64_t> shape);
  mlir::DenseElementsAttr convertTensor(const onnx::TensorProto &tensor);
  mlir::RankedTensorType convertValueInfo(const onnx::ValueInfoProto &info);

  void buildRelu(const onnx::NodeProto &node, mlir::Location loc);
  void buildAdd(const onnx::NodeProto &node, mlir::Location loc);
  void buildMul(const onnx::NodeProto &node, mlir::Location loc);
  void buildMatMul(const onnx::NodeProto &node, mlir::Location loc);
  void buildGemm(const onnx::NodeProto &node, mlir::Location loc);
  void buildConv(const onnx::NodeProto &node, mlir::Location loc);
  void buildTranspose(const onnx::NodeProto &node, mlir::Location loc);

public:
  MLIRBuilder(mlir::MLIRContext &ctx);
  const mlir::ModuleOp &GetModule() const;

  void Visit(const onnx::ModelProto &) override;
  void Visit(const onnx::GraphProto &graph) override;
  void Visit(const onnx::ValueInfoProto &) override;
  void Visit(const onnx::TensorProto &tensor) override;
  void Visit(const onnx::NodeProto &node) override;
  void Visit(const onnx::AttributeProto &) override;

  void Finalize(const onnx::GraphProto &graph) override;
};