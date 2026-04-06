#include "TensorCompiler/Graph/GraphDumper.hpp"
#include "TensorCompiler/Graph/GraphVisitor.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace tc::graph {
namespace {

static std::string indent(size_t n) { return std::string(n, ' '); }

static std::string ShapeToStr(const std::vector<int64_t> &shape) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i)
      ss << ", ";
    ss << shape[i];
  }
  ss << "]";
  return ss.str();
}

class DumpGraphVisitor final : public tc::graph::GraphVisitor {
public:
  void Visit(const Graph &graph) override { graph_ = &graph; }

  void Visit(const TensorEntity &tensor) override {
    tensors_.push_back(&tensor);
  }

  void Visit(const OperationEntity &op) override { ops_.push_back(&op); }

  void Visit(const ConstantEntity &) override { /* Do nothing */ }

  void Finalize(const Graph & /*graph*/) override {
    std::ostringstream oss;
    oss << "=== Graph ===\n";

    oss << "\nInputs:\n";
    for (auto id : graph_->Inputs()) {
      if (auto *t = graph_->GetTensor(id))
        oss << indent(2) << t->name << " " << ShapeToStr(t->shape) << "\n";
    }

    oss << "\nOperations:\n";
    for (const auto *op : ops_) {
      oss << "\n"
          << indent(2) << "[" << op->op_type << "] " << op->name << "\n";
      oss << indent(4) << "inputs:\n";
      for (auto in : op->inputs) {
        if (auto *t = graph_->GetTensor(in))
          oss << indent(6) << t->name << "\n";
      }
      oss << indent(4) << "outputs:\n";
      for (auto out : op->outputs) {
        if (auto *t = graph_->GetTensor(out))
          oss << indent(6) << t->name << "\n";
      }
    }

    oss << "\nOutputs:\n";
    for (auto id : graph_->Outputs()) {
      if (auto *t = graph_->GetTensor(id))
        oss << indent(2) << t->name << "\n";
    }

    oss << "\nConstants:\n";
    for (const auto *t : tensors_) {
      if (t->is_initializer)
        oss << indent(2) << t->name << " " << ShapeToStr(t->shape) << "\n";
    }

    oss << "\nTensors:\n";
    for (const auto *t : tensors_) {
      oss << indent(2) << t->name << " " << ShapeToStr(t->shape);
      if (t->is_initializer)
        oss << "  (init)";
      oss << "\n";
    }

    result_ = oss.str();
  }

  std::string GetResult() const { return result_; }

private:
  const Graph *graph_ = nullptr;
  std::vector<const TensorEntity *> tensors_;
  std::vector<const OperationEntity *> ops_;
  std::string result_;
};

class DotGraphVisitor final : public tc::graph::GraphVisitor {
public:
  void Visit(const Graph &graph) override {
    graph_ = &graph;
    oss_ << "digraph G {\n";
    oss_ << "  rankdir=LR;\n";
  }

  void Visit(const TensorEntity &tensor) override {
    int id = nextId();
    tensorIds_[&tensor] = id;
    oss_ << "  t" << id << " [shape=oval,label=\"" << tensor.name << "\"];\n";
  }

  void Visit(const OperationEntity &op) override {
    int id = nextId();
    opIds_[&op] = id;
    ops_.push_back(&op);
    oss_ << "  op" << id << " [shape=box,label=\"" << op.op_type << "\"];\n";
  }

  void Visit(const ConstantEntity &) override { /* Do nothing */ }

  void Finalize(const Graph & /*graph*/) override {
    for (const auto *op : ops_) {
      auto opIt = opIds_.find(op);
      if (opIt == opIds_.end())
        continue;
      int opId = opIt->second;

      for (auto inId : op->inputs) {
        auto *t = graph_->GetTensor(inId);
        if (!t)
          continue;
        auto tIt = tensorIds_.find(t);
        if (tIt != tensorIds_.end())
          oss_ << "  t" << tIt->second << " -> op" << opId << ";\n";
      }
      for (auto outId : op->outputs) {
        auto *t = graph_->GetTensor(outId);
        if (!t)
          continue;
        auto tIt = tensorIds_.find(t);
        if (tIt != tensorIds_.end())
          oss_ << "  op" << opId << " -> t" << tIt->second << ";\n";
      }
    }

    oss_ << "}\n";
    result_ = oss_.str();
  }

  std::string GetResult() const { return result_; }

private:
  int nextId() { return idCounter_++; }

  const Graph *graph_ = nullptr;
  std::ostringstream oss_;
  int idCounter_ = 0;
  std::vector<const OperationEntity *> ops_;
  std::unordered_map<const TensorEntity *, int> tensorIds_;
  std::unordered_map<const OperationEntity *, int> opIds_;
  std::string result_;
};

} // namespace

std::string DumpGraph(const Graph &g) {
  DumpGraphVisitor visitor;
  visitor.Visit(g);
  for (const auto &e : g.Entities()) {
    if (e.kind == EntityKind::Tensor)
      visitor.Visit(std::get<TensorEntity>(e.entity));
    else if (e.kind == EntityKind::Operation)
      visitor.Visit(std::get<OperationEntity>(e.entity));
    else if (e.kind == EntityKind::Constant)
      visitor.Visit(std::get<ConstantEntity>(e.entity));
  }
  visitor.Finalize(g);
  return visitor.GetResult();
}

std::string ToDot(const Graph &g) {
  DotGraphVisitor visitor;
  visitor.Visit(g);
  for (const auto &e : g.Entities()) {
    if (e.kind == EntityKind::Tensor)
      visitor.Visit(std::get<TensorEntity>(e.entity));
    else if (e.kind == EntityKind::Operation)
      visitor.Visit(std::get<OperationEntity>(e.entity));
    else if (e.kind == EntityKind::Constant)
      visitor.Visit(std::get<ConstantEntity>(e.entity));
  }
  visitor.Finalize(g);
  return visitor.GetResult();
}

bool SaveDot(const Graph &g, const std::string &file) {
  std::ofstream f(file);
  if (!f)
    return false;
  f << ToDot(g);
  return true;
}
} // namespace tc::graph