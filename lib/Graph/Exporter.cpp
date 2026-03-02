#include "TensorCompiler/Graph/Exporter.hpp"
#include <fstream>
#include <sstream>

namespace tc::graph {

static constexpr size_t PAD = 2;
static constexpr size_t OP_PAD = 4;
static constexpr size_t IO_PAD = 6;

static std::string indent(size_t n) {
  return std::string(n, ' ');
}

static std::string ShapeToStr(const std::vector<int64_t>& shape) {
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) ss << ", ";
    ss << shape[i];
  }
  ss << "]";
  return ss.str();
}

std::string DumpGraph(const Graph& g) {
  std::ostringstream os;

  os << "=== Graph ===\n";

  os << "\nInputs:\n";
  for (auto id : g.Inputs()) {
    if (auto* t = g.GetTensor(id))
      os << indent(PAD) << t->name << " " << ShapeToStr(t->shape) << "\n";
  }

  os << "\nOperations:\n";

  for (const auto& e : g.Entities()) {
    if (e.kind != EntityKind::Operation)
      continue;

    const auto& op = std::get<OperationEntity>(e.entity);

    os << "\n" << indent(PAD)
       << "[" << op.op_type << "] " << op.name << "\n";

    os << indent(OP_PAD) << "inputs:\n";
    for (auto in : op.inputs)
      if (auto* t = g.GetTensor(in))
        os << indent(IO_PAD) << t->name << "\n";

    os << indent(OP_PAD) << "outputs:\n";
    for (auto out : op.outputs)
      if (auto* t = g.GetTensor(out))
        os << indent(IO_PAD) << t->name << "\n";
  }

  os << "\nOutputs:\n";
  for (auto id : g.Outputs()) {
    if (auto* t = g.GetTensor(id))
      os << indent(PAD) << t->name << "\n";
  }

  os << "\nConstants:\n";
  for (const auto &e : g.Entities()) {
    if (e.kind != EntityKind::Tensor)
      continue;

    const auto &t = std::get<TensorEntity>(e.entity);

    if (!t.is_initializer)
      continue;

    os << indent(PAD)
      << t.name << " "
      << ShapeToStr(t.shape)
      << "\n";
  }

  os << "\nTensors:\n";
  for (const auto &e : g.Entities()) {
    if (e.kind != EntityKind::Tensor)
      continue;

    const auto &t = std::get<TensorEntity>(e.entity);

    os << indent(PAD)
      << t.name
      << " " << ShapeToStr(t.shape);

    if (t.is_initializer)
      os << "  (init)";

    os << "\n";
  }

  return os.str();
}

std::string ToDot(const Graph& g) {
  std::ostringstream os;
  os << "digraph G {\n";
  os << "  rankdir=LR;\n";

  for (size_t i = 0; i < g.Entities().size(); ++i) {
    const auto& e = g.Entities()[i];

    if (e.kind == EntityKind::Tensor) {
      const auto& t = std::get<TensorEntity>(e.entity);
      os << "  t" << i
         << " [shape=oval,label=\"" << t.name << "\"];\n";
    }

    if (e.kind == EntityKind::Operation) {
      const auto& op = std::get<OperationEntity>(e.entity);

      os << "  op" << i
         << " [shape=box,label=\"" << op.op_type << "\"];\n";

      for (auto in : op.inputs)
        os << "  t" << in << " -> op" << i << ";\n";

      for (auto out : op.outputs)
        os << "  op" << i << " -> t" << out << ";\n";
    }
  }

  os << "}\n";
  return os.str();
}

bool SaveDot(const Graph& g, const std::string& file) {
  std::ofstream f(file);
  if (!f)
    return false;
  f << ToDot(g);
  return true;
}


} // namespace tc::graph