#pragma once

namespace tc::graph {
class Graph;
struct TensorEntity;
struct OperationEntity;
struct ConstantEntity;
} // namespace tc::graph

namespace tc::graph {

class GraphVisitor {
public:
  virtual ~GraphVisitor() = default;

  virtual void Visit(const tc::graph::Graph &graph) = 0;
  virtual void Visit(const tc::graph::TensorEntity &tensor) = 0;
  virtual void Visit(const tc::graph::OperationEntity &op) = 0;
  virtual void Visit(const tc::graph::ConstantEntity &constant) = 0;
  virtual void Finalize(const tc::graph::Graph &graph) = 0;
};

} // namespace tc::graph