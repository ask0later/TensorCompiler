#include "TensorCompiler/Converter/GraphBuilder.hpp"

#include <cstring>

namespace tc::converter::onnx_to_graph {
using tc::graph::AttrValue;
using tc::graph::DoubleList;
using tc::graph::EntityId;
using tc::graph::IntList;
using tc::graph::StringList;
using tc::graph::TensorData;

TensorData GraphBuilder::ParseTensor(const onnx::TensorProto &t) {
  TensorData td;
  td.name = t.name();

  td.dims.reserve(t.dims_size());
  for (auto d : t.dims())
    td.dims.push_back(d);

  if (!t.raw_data().empty()) {
    const auto &raw = t.raw_data();
    td.raw_data.assign(raw.begin(), raw.end());
    return td;
  }

  if (t.float_data_size()) {
    size_t bytes = t.float_data_size() * sizeof(float);
    td.raw_data.resize(bytes);
    std::memcpy(td.raw_data.data(), t.float_data().data(), bytes);
    return td;
  }

  if (t.int64_data_size()) {
    size_t bytes = t.int64_data_size() * sizeof(int64_t);
    td.raw_data.resize(bytes);
    std::memcpy(td.raw_data.data(), t.int64_data().data(), bytes);
    return td;
  }

  if (t.int32_data_size()) {
    size_t bytes = t.int32_data_size() * sizeof(int32_t);
    td.raw_data.resize(bytes);
    std::memcpy(td.raw_data.data(), t.int32_data().data(), bytes);
    return td;
  }

  return td;
}

AttrValue GraphBuilder::ParseAttribute(const onnx::AttributeProto &a) {
  switch (a.type()) {
  case onnx::AttributeProto::INT:
    return static_cast<int64_t>(a.i());
  case onnx::AttributeProto::FLOAT:
    return static_cast<double>(a.f());
  case onnx::AttributeProto::STRING:
    return std::string(a.s());
  case onnx::AttributeProto::INTS:
    return IntList(a.ints().begin(), a.ints().end());
  case onnx::AttributeProto::FLOATS:
    return DoubleList(a.floats().begin(), a.floats().end());
  case onnx::AttributeProto::STRINGS: {
    StringList v;
    v.reserve(a.strings_size());
    for (auto &s : a.strings())
      v.emplace_back(s);
    return v;
  }
  case onnx::AttributeProto::TENSOR:
    return ParseTensor(a.t());
  default:
    return std::monostate{};
  }
}

EntityId GraphBuilder::EnsureTensor(const std::string &name,
                                    const std::vector<int64_t> &shape) {
  if (name.empty())
    return -1;

  return graph_.AddTensor(name, shape, initializerNames_.count(name) != 0);
}

void GraphBuilder::Visit(const onnx::ModelProto &) { /* Do nothing */ }

void GraphBuilder::Visit(const onnx::GraphProto &g) {
  initializerNames_.clear();

  for (auto &init : g.initializer())
    initializerNames_.insert(init.name());

  for (auto &in : g.input())
    if (!initializerNames_.count(in.name()))
      graph_.AddInput(EnsureTensor(in.name()));

  for (auto &out : g.output())
    graph_.AddOutput(EnsureTensor(out.name()));
}

void GraphBuilder::Visit(const onnx::ValueInfoProto &value) {
  if (!value.name().empty() && value.has_type() &&
      value.type().has_tensor_type()) {

    std::vector<int64_t> shape;

    const auto &tt = value.type().tensor_type();
    if (tt.has_shape()) {
      for (auto &dim : tt.shape().dim())
        shape.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
    }

    EnsureTensor(value.name(), shape);
  }
}

void GraphBuilder::Visit(const onnx::TensorProto &tensor) {
  TensorData td = ParseTensor(tensor);

  graph_.AddConstant(td);

  EntityId tid = EnsureTensor(td.name, td.dims);

  if (auto *t = graph_.GetTensor(tid)) {
    t->is_initializer = true;
    t->data = td;
  }
}

void GraphBuilder::Visit(const onnx::NodeProto &node) {
  std::vector<EntityId> ins;
  std::vector<EntityId> outs;

  ins.reserve(node.input_size());
  outs.reserve(node.output_size());

  for (auto &i : node.input())
    if (!i.empty())
      ins.push_back(EnsureTensor(i));

  for (auto &o : node.output())
    if (!o.empty())
      outs.push_back(EnsureTensor(o));

  std::unordered_map<std::string, AttrValue> attrs;
  for (auto &a : node.attribute())
    attrs[a.name()] = ParseAttribute(a);

  graph_.AddOperation(node.op_type(), node.name(), ins, outs, attrs);
}

void GraphBuilder::Visit(const onnx::AttributeProto &) { /* Do nothing */ }

void GraphBuilder::Finalize(const onnx::GraphProto &) { /* Do nothing */ }

} // namespace tc::converter::onnx_to_graph