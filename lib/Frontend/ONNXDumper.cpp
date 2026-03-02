#include "TensorCompiler/Frontend/ONNXDumper.hpp"

namespace tc::frontend::debug {
static constexpr size_t GRAPH_PADDING = 4;
static constexpr size_t TENSOR_PADDING = 6;
static constexpr size_t NODE_PADDING = 6;
static constexpr size_t INFO_PADDING = 6;
static constexpr size_t ATTRIBUTE_PADDING = 8;

static void dumpTensor(const onnx::TensorProto &tensor, std::ostream &os,
                       size_t padding) {
  std::string pad(padding, ' ');
  os << pad << "Tensor: " << tensor.name() << "\n";
  os << pad << "  dims: [";
  for (size_t i = 0; i < static_cast<size_t>(tensor.dims_size()); ++i) {
    os << tensor.dims(static_cast<int>(i));
    if (i + 1 < static_cast<size_t>(tensor.dims_size()))
      os << ", ";
  }
  os << "]\n";
  os << pad << "  dtype: " << tensor.data_type() << "\n";

  if (tensor.float_data_size() > 0) {
    os << pad << "  float_data: [";
    for (size_t i = 0; i < static_cast<size_t>(tensor.float_data_size()); ++i) {
      os << tensor.float_data(static_cast<int>(i));
      if (i + 1 < static_cast<size_t>(tensor.float_data_size()))
        os << ", ";
    }
    os << "]\n";
  } else if (tensor.has_raw_data()) {
    const float *data =
        reinterpret_cast<const float *>(tensor.raw_data().data());
    size_t count = tensor.raw_data().size() / sizeof(float);

    os << pad << "  raw_data (float): [";
    for (size_t i = 0; i < count; ++i) {
      os << data[i];
      if (i + 1 < count)
        os << ", ";
    }
    os << "]\n";
  }
}

DumpVisitor::DumpVisitor(std::ostream &os) : os_(os) {}

void DumpVisitor::Visit(const onnx::ModelProto &model) {
  os_ << "=== ONNX Model ===\n";
  os_ << "  ir_version:    " << model.ir_version() << "\n";
  os_ << "  opset_version: ";

  for (const auto &opset : model.opset_import())
    os_ << opset.domain() << ":" << opset.version() << " ";
  os_ << "\n";

  if (!model.domain().empty())
    os_ << "  domain:        " << model.domain() << "\n";
  if (!model.model_version())
    os_ << "  model_version: " << model.model_version() << "\n";
  if (!model.doc_string().empty())
    os_ << "  doc_string:    " << model.doc_string() << "\n";
  if (!model.producer_name().empty())
    os_ << "  producer_name: " << model.producer_name() << "\n";
  if (!model.producer_version().empty())
    os_ << "  producer_ver:  " << model.producer_version() << "\n";
}

void DumpVisitor::Visit(const onnx::GraphProto &graph) {
  std::string pad(GRAPH_PADDING, ' ');
  os_ << pad << "=== Graph: " << graph.name() << " ===\n";
  os_ << pad << "  nodes:        " << graph.node_size() << "\n";
  os_ << pad << "  inputs:       " << graph.input_size() << "\n";
  os_ << pad << "  outputs:      " << graph.output_size() << "\n";
  os_ << pad << "  initializers: " << graph.initializer_size() << "\n";
}

void DumpVisitor::Visit(const onnx::TensorProto &tensor) {
  dumpTensor(tensor, os_, TENSOR_PADDING);
}

void DumpVisitor::Visit(const onnx::NodeProto &node) {
  std::string pad(NODE_PADDING, ' ');
  os_ << pad << "Node: " << node.op_type();
  if (!node.name().empty())
    os_ << " (" << node.name() << ")";
  os_ << "\n";

  os_ << pad << "  inputs:  [";
  for (size_t i = 0; i < static_cast<size_t>(node.input_size()); ++i) {
    os_ << node.input(static_cast<int>(i));
    if (i + 1 < static_cast<size_t>(node.input_size()))
      os_ << ", ";
  }
  os_ << "]\n";

  os_ << pad << "  outputs: [";
  for (size_t i = 0; i < static_cast<size_t>(node.output_size()); ++i) {
    os_ << node.output(static_cast<int>(i));
    if (i + 1 < static_cast<size_t>(node.output_size()))
      os_ << ", ";
  }
  os_ << "]\n";
}

void DumpVisitor::Visit(const onnx::AttributeProto &attr) {
  std::string pad(ATTRIBUTE_PADDING, ' ');
  os_ << pad << "Attribute: " << attr.name() << "\n";

  switch (attr.type()) {
  case onnx::AttributeProto::FLOAT:
    os_ << pad << "  float: " << attr.f() << "\n";
    break;
  case onnx::AttributeProto::INT:
    os_ << pad << "  int: " << attr.i() << "\n";
    break;
  case onnx::AttributeProto::STRING:
    os_ << pad << "  string: " << attr.s() << "\n";
    break;
  case onnx::AttributeProto::TENSOR:
    dumpTensor(attr.t(), os_, ATTRIBUTE_PADDING + 2);
    break;
  case onnx::AttributeProto::TENSORS:
    for (size_t i = 0; i < static_cast<size_t>(attr.tensors_size()); ++i)
      dumpTensor(attr.tensors(static_cast<int>(i)), os_, ATTRIBUTE_PADDING + 2);
    break;
  case onnx::AttributeProto::FLOATS:
    os_ << pad << "  floats: [";
    for (size_t i = 0; i < static_cast<size_t>(attr.floats_size()); ++i) {
      os_ << attr.floats(static_cast<int>(i));
      if (i + 1 < static_cast<size_t>(attr.floats_size()))
        os_ << ", ";
    }
    os_ << "]\n";
    break;
  case onnx::AttributeProto::INTS:
    os_ << pad << "  ints: [";
    for (size_t i = 0; i < static_cast<size_t>(attr.ints_size()); ++i) {
      os_ << attr.ints(static_cast<int>(i));
      if (i + 1 < static_cast<size_t>(attr.ints_size()))
        os_ << ", ";
    }
    os_ << "]\n";
    break;
  default:
    os_ << pad << "  type: " << attr.type() << " (unsupported dump)\n";
    break;
  }
}

void DumpVisitor::Visit(const onnx::ValueInfoProto &info) {
  std::string pad(INFO_PADDING, ' ');
  os_ << pad << "ValueInfo: " << info.name() << "\n";

  if (info.has_type() && info.type().has_tensor_type()) {
    const auto &tt = info.type().tensor_type();
    os_ << pad << "  dtype: " << tt.elem_type() << "\n";

    if (tt.has_shape()) {
      os_ << pad << "  shape: [";

      for (size_t i = 0; i < static_cast<size_t>(tt.shape().dim_size()); ++i) {
        const auto &dim = tt.shape().dim(static_cast<int>(i));

        if (dim.has_dim_value())
          os_ << dim.dim_value();
        else if (dim.has_dim_param())
          os_ << dim.dim_param();
        else
          os_ << "?";

        if (i + 1 < static_cast<size_t>(tt.shape().dim_size()))
          os_ << ", ";
      }

      os_ << "]\n";
    }
  }
}

void DumpVisitor::Finalize(const onnx::GraphProto &) { /* Do nothing*/ }
} // namespace tc::frontend::debug