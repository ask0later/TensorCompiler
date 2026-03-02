#include "TensorCompiler/Frontend/ONNXModel.hpp"
#include "TensorCompiler/Converter/GraphBuilder.hpp"
#include "TensorCompiler/Graph/IR.hpp"

#include <gtest/gtest.h>

using namespace tc;
using namespace tc::converter::onnx_to_graph;
using namespace tc::graph;

static onnx::ModelProto BuildSimpleModel() {
  onnx::ModelProto model;
  auto* graph = model.mutable_graph();
  graph->set_name("test_graph");

  auto* input = graph->add_input();
  input->set_name("x");

  auto* weight = graph->add_initializer();
  weight->set_name("w");
  weight->add_dims(2);
  weight->add_dims(2);
  weight->set_data_type(onnx::TensorProto::FLOAT);
  weight->add_float_data(1);
  weight->add_float_data(1);
  weight->add_float_data(1);
  weight->add_float_data(1);

  auto* node = graph->add_node();
  node->set_op_type("Relu");
  node->add_input("x");
  node->add_output("y");

  auto* output = graph->add_output();
  output->set_name("y");

  return model;
}

static Graph BuildGraphFromModel(const onnx::ModelProto& model) {
  GraphBuilder builder;
  builder.Visit(model);
  builder.Visit(model.graph());

  for (const auto& in : model.graph().input())
    builder.Visit(in);

  for (const auto& init : model.graph().initializer())
    builder.Visit(init);

  for (const auto& node : model.graph().node())
    builder.Visit(node);

  builder.Finalize(model.graph());
  return builder.GetGraph();
}

static Graph ONNXtoGraph(std::string_view fileName) {
  tc::frontend::ONNXModel model{fileName};
  tc::converter::onnx_to_graph::GraphBuilder builder;
  model.Parse(builder);
  return builder.GetGraph();
}

static bool hasOp(const Graph &g, const std::string &opType) {
  for (const auto &e : g.Entities()) {
    if (e.kind != EntityKind::Operation)
      continue;

    const auto &op = std::get<OperationEntity>(e.entity);
    if (op.op_type == opType)
      return true;
  }
  return false;
}

static size_t countConstants(const Graph &g) {
  size_t n = 0;
  for (auto &e : g.Entities())
    if (e.kind == EntityKind::Constant)
      ++n;
  return n;
}

TEST(ONNXGraphBuilderTest, BuildsGraphStructure) {
  auto model = BuildSimpleModel();
  Graph g = BuildGraphFromModel(model);

  EXPECT_EQ(g.Inputs().size(), 1);
  EXPECT_EQ(g.Outputs().size(), 1);

  bool foundTensorX = false;
  bool foundTensorY = false;
  bool foundInitializer = false;
  bool foundConstant = false;
  bool foundRelu = false;

  for (const auto& e : g.Entities()) {

    if (e.kind == EntityKind::Tensor) {
      const auto& t = std::get<TensorEntity>(e.entity);
      if (t.name == "x") foundTensorX = true;
      if (t.name == "y") foundTensorY = true;
      if (t.name == "w" && t.is_initializer) foundInitializer = true;
    }

    if (e.kind == EntityKind::Constant) {
      const auto& c = std::get<ConstantEntity>(e.entity);
      if (c.data.name == "w") foundConstant = true;
    }

    if (e.kind == EntityKind::Operation) {
      const auto& op = std::get<OperationEntity>(e.entity);
      if (op.op_type == "Relu") foundRelu = true;
    }
  }

  EXPECT_TRUE(foundTensorX);
  EXPECT_TRUE(foundTensorY);
  EXPECT_TRUE(foundInitializer);
  EXPECT_TRUE(foundConstant);
  EXPECT_TRUE(foundRelu);
}

TEST(ONNXGraphBuilderTest, DetectsInitializerTensor) {
  auto model = BuildSimpleModel();
  Graph g = BuildGraphFromModel(model);

  bool foundInitializer = false;

  for (const auto& e : g.Entities()) {
    if (e.kind != EntityKind::Tensor)
      continue;

    const auto& t = std::get<TensorEntity>(e.entity);
    if (t.name == "w") {
      foundInitializer = true;
      EXPECT_TRUE(t.is_initializer);
      ASSERT_TRUE(t.data.has_value());
      EXPECT_EQ(t.data->dims.size(), 2);
    }
  }

  EXPECT_TRUE(foundInitializer);
}

TEST(ONNXGraphBuilderTest, OperationCreatedCorrectly) {
  auto model = BuildSimpleModel();
  Graph g = BuildGraphFromModel(model);

  bool foundRelu = false;

  for (const auto& e : g.Entities()) {
    if (e.kind != EntityKind::Operation)
      continue;

    const auto& op = std::get<OperationEntity>(e.entity);

    if (op.op_type == "Relu") {
      foundRelu = true;
      EXPECT_EQ(op.inputs.size(), 1);
      EXPECT_EQ(op.outputs.size(), 1);
    }
  }

  EXPECT_TRUE(foundRelu);
}

TEST(ONNXGraphBuilderTest, HandlesBinaryOp) {
  onnx::ModelProto model;
  auto* graph = model.mutable_graph();

  graph->add_input()->set_name("a");
  graph->add_input()->set_name("b");

  auto* node = graph->add_node();
  node->set_op_type("Add");
  node->add_input("a");
  node->add_input("b");
  node->add_output("c");

  graph->add_output()->set_name("c");

  Graph g = BuildGraphFromModel(model);

  bool foundAdd = false;

  for (auto& e : g.Entities()) {
    if (e.kind != EntityKind::Operation)
      continue;

    auto& op = std::get<OperationEntity>(e.entity);

    if (op.op_type == "Add") {
      foundAdd = true;
      EXPECT_EQ(op.inputs.size(), 2);
      EXPECT_EQ(op.outputs.size(), 1);
    }
  }

  EXPECT_TRUE(foundAdd);
}

TEST(ONNXtoGraph, AddMul) {
  auto g = ONNXtoGraph("tests/data/add_mul.onnx");

  EXPECT_TRUE(hasOp(g, "Add"));
  EXPECT_TRUE(hasOp(g, "Mul"));
}

TEST(ONNXtoGraph, ConvRelu) {
  auto g = ONNXtoGraph("tests/data/conv_relu.onnx");

  EXPECT_TRUE(hasOp(g, "Conv"));
  EXPECT_TRUE(hasOp(g, "Relu"));
}

TEST(ONNXtoGraph, MatMulRelu) {
  auto g = ONNXtoGraph("tests/data/matmul_relu.onnx");

  EXPECT_TRUE(hasOp(g, "MatMul"));
  EXPECT_TRUE(hasOp(g, "Relu"));
}

TEST(ONNXtoGraph, SingleRelu) {
  auto g = ONNXtoGraph("tests/data/relu.onnx");

  EXPECT_TRUE(hasOp(g, "Relu"));
}

TEST(ONNXtoGraph, TransposeMatMul) {
  auto g = ONNXtoGraph("tests/data/transpose_matmul.onnx");

  EXPECT_TRUE(hasOp(g, "Transpose"));
  EXPECT_TRUE(hasOp(g, "MatMul"));
}

TEST(ONNXtoGraph, Gemm) {
  auto g = ONNXtoGraph("tests/data/gemm.onnx");

  EXPECT_TRUE(hasOp(g, "Gemm"));
}

TEST(ONNXtoGraph, Pipeline) {
  auto g = ONNXtoGraph("tests/data/test_0.onnx");

  EXPECT_TRUE(hasOp(g, "Transpose"));
  EXPECT_TRUE(hasOp(g, "MatMul"));
  EXPECT_TRUE(hasOp(g, "Relu"));
  EXPECT_TRUE(hasOp(g, "Conv"));
  EXPECT_TRUE(hasOp(g, "Add"));
  EXPECT_TRUE(hasOp(g, "Mul"));
  EXPECT_TRUE(hasOp(g, "Gemm"));

  EXPECT_FALSE(g.Inputs().empty());
  EXPECT_FALSE(g.Outputs().empty());
}

TEST(ONNXtoGraph, WeightsAndBias) {
  auto g = ONNXtoGraph("tests/data/test_1.onnx");

  bool foundWeights = false;
  bool foundBias = false;

  for (const auto &e : g.Entities()) {
    if (e.kind != EntityKind::Constant)
      continue;

    const auto &c = std::get<ConstantEntity>(e.entity);

    if (c.data.dims == std::vector<int64_t>{3,4}) {
      foundWeights = true;
      EXPECT_EQ(c.data.raw_data.size(), 3 * 4 * sizeof(float));
    }

    if (c.data.dims == std::vector<int64_t>{4}) {
      foundBias = true;
    }
  }

  EXPECT_TRUE(foundWeights);
  EXPECT_TRUE(foundBias);
}

TEST(ONNXtoGraph, InputsOutputsAreSet) {
  auto g = ONNXtoGraph("tests/data/add_mul.onnx");

  EXPECT_GT(g.Inputs().size(), 0u);
  EXPECT_GT(g.Outputs().size(), 0u);

  for (auto id : g.Inputs()) {
    auto *t = g.GetTensor(id);
    ASSERT_NE(t, nullptr);
    EXPECT_FALSE(t->name.empty());
  }

  for (auto id : g.Outputs()) {
    auto *t = g.GetTensor(id);
    ASSERT_NE(t, nullptr);
    EXPECT_FALSE(t->name.empty());
  }
}