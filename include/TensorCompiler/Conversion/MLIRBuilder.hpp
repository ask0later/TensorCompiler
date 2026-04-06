#pragma once
#include "TensorCompiler/Dialect/NNDialect.hpp"
#include "TensorCompiler/Graph/Graph.hpp"
#include "TensorCompiler/Graph/GraphVisitor.hpp"

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <unordered_map>
#include <unordered_set>

namespace tc::conversion {

class HighLevelMLIRBuilder : public tc::graph::GraphVisitor {
public:
  explicit HighLevelMLIRBuilder(mlir::MLIRContext &ctx);

  const mlir::ModuleOp &GetModule() const { return module_; }

  mlir::ModuleOp Build(const tc::graph::Graph &graph) {
    graph.Accept(*this);
    return module_;
  }

private:
  void Visit(const tc::graph::Graph &graph) override;
  void Visit(const tc::graph::TensorEntity &tensor) override;
  void Visit(const tc::graph::OperationEntity &op) override;
  void Visit(const tc::graph::ConstantEntity &constant) override;
  void Finalize(const tc::graph::Graph &graph) override;

  mlir::Value GetValue(tc::graph::EntityId id) const;
  mlir::Value GetValue(const std::string &name) const;

  void BuildRelu(const tc::graph::OperationEntity &op, mlir::Location loc);
  void BuildAdd(const tc::graph::OperationEntity &op, mlir::Location loc);
  void BuildMul(const tc::graph::OperationEntity &op, mlir::Location loc);
  void BuildMatMul(const tc::graph::OperationEntity &op, mlir::Location loc);
  void BuildGemm(const tc::graph::OperationEntity &op, mlir::Location loc);
  void BuildConv(const tc::graph::OperationEntity &op, mlir::Location loc);
  void BuildTranspose(const tc::graph::OperationEntity &op, mlir::Location loc);

  mlir::Type ConvertElemType(int32_t onnx_dtype);
  mlir::RankedTensorType ConvertTensorType(int32_t dtype,
                                           llvm::ArrayRef<int64_t> shape);
  mlir::DenseElementsAttr ConvertTensorData(const tc::graph::TensorData &data);

private:
  mlir::MLIRContext &ctx_;
  mlir::OpBuilder builder_;
  mlir::ModuleOp module_;

  std::unordered_map<tc::graph::EntityId, mlir::Value> valueMap_;
  std::unordered_map<std::string, tc::graph::EntityId> tensorNameToId_;
  std::unordered_set<tc::graph::EntityId> initializerIds_;
};
} // namespace tc::conversion