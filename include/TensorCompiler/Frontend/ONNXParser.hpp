#pragma once

#include <onnx/onnx_pb.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <string>
#include <fstream>

class ONNXParser final {
    const std::string_view file_name_;
    onnx::ModelProto model_;

    ONNXParser(const std::string_view file_name) : file_name_(file_name) {
        std::ifstream input_model(file_name_.data(), std::ios::binary);
        if (!input_model.good()) {
            std::string err = std::string("Failed to open file: ") + std::string(file_name_);
            throw std::runtime_error(err);
        }

        model_.ParseFromIstream(&input_model);
    }

    void ParseGraph(const onnx::GraphProto &graph) {
        std::cout << "\n=== Graph ===\n";
        std::cout << "Name: " << graph.name() << "\n";
        std::cout << "Nodes: " << graph.node_size() << "\n";
        std::cout << "Inputs: " << graph.input_size() << "\n";
        std::cout << "Outputs: " << graph.output_size() << "\n";
        std::cout << "Initializers: " << graph.initializer_size() << "\n";

        for (auto &&tensor : graph.initializer()) {
            ParseTensor(tensor);
        }
        
        for (auto &&node : graph.node()) {
            ParseNode(node);
        }

        for (auto &&input : graph.input()) {
            ParseInput(input);
        }

        for (auto &&output : graph.output()) {
            ParseOutput(output);
        }
    }

    void ParseTensor(const onnx::TensorProto &tensor) {
        std::cout << tensor.name() << std::endl;

        for (size_t i = 0; i < tensor.dims_size(); i++) {
            std::cout << tensor.dims(i) << std::endl;
        }

        int dtype = tensor.data_type();

        if (tensor.float_data_size() > 0) {
            for (size_t i = 0; i < tensor.float_data_size(); i++) {
                float val = tensor.float_data(i);
                std::cout << val << std::endl;
            }
        } else if (tensor.has_raw_data()) {
            const std::string& raw = tensor.raw_data();
            const float* data = reinterpret_cast<const float*>(raw.data());
            std::cout << *data << std::endl;
        }
    }

    void ParseNode(const onnx::NodeProto &node) {
        std::string op_type = node.op_type();
        std::cout << op_type << std::endl;

        for (const auto& in : node.input())
            std::cout << "\t" << in << "\n";

        for (const auto& out : node.output())
            std::cout << "\t" << out << "\n";

        for (auto &&attr : node.attribute()) {
            ParseAttribute(attr);
        }
    }

    void ParseAttribute(const onnx::AttributeProto &attr) {
        std::string attr_name = attr.name();
        std::cout << attr_name << std::endl;
    }

    void ParseInput(const onnx::ValueInfoProto &input) {
        std::cout << input.name() << std::endl;

        if (input.has_type() && input.type().has_tensor_type()) {
            const auto &tensor_type = input.type().tensor_type();
            const auto &dtype = tensor_type.elem_type();

            if (tensor_type.has_shape()) {
                for (size_t i = 0; i < tensor_type.shape().dim_size(); i++) {
                    const auto& dim = tensor_type.shape().dim(i);
                    if (dim.has_dim_value()) {
                        std::cout << dim.dim_value() << std::endl;
                    } else if (dim.has_dim_param()) {
                        std::cout << dim.dim_param() << std::endl;
                    }
                }
            }
        }
    }

    void ParseOutput(const onnx::ValueInfoProto &output) {
        std::cout << output.name() << std::endl;

        if (output.has_type() && output.type().has_tensor_type()) {
            const auto &tensor_type = output.type().tensor_type();
            const auto &elem_type = tensor_type.elem_type();

            if (tensor_type.has_shape()) {
                for (size_t i = 0; i < tensor_type.shape().dim_size(); i++) {
                    const auto& dim = tensor_type.shape().dim(i);
                    if (dim.has_dim_value()) {
                        std::cout << dim.dim_value() << std::endl;
                    } else if (dim.has_dim_param()) {
                        std::cout << dim.dim_param() << std::endl;
                    }
                }
            }
        }
    }
    
public:
    static ONNXParser &QueryONNXParser(const std::string_view file_name) {
        static ONNXParser parser{file_name};
        return parser;
    }

    void ParseModel() {
        if (model_.has_graph()) {
            const onnx::GraphProto& graph = model_.graph();
            ParseGraph(graph);
        } else {
            throw std::runtime_error("Error: Model has no graph.");
        }
    }
};