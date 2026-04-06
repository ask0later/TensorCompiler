#include "TensorCompiler/Graph/Graph.hpp"
#include "TensorCompiler/Graph/GraphDumper.hpp"

#include <gtest/gtest.h>
#include <string>

using namespace tc::graph;

static constexpr int32_t FLOAT_DTYPE = 1; // onnx::TensorProto::FLOAT

static TensorData MakeTensorData(std::string name, std::vector<int64_t> dims,
                                 int32_t dtype = FLOAT_DTYPE) {
  TensorData td;
  td.name = std::move(name);
  td.dims = std::move(dims);
  td.dtype = dtype;
  return td;
}

template <typename T> const T &GetScalar(const AttrValue &v) {
  return std::get<T>(std::get<AttrScalar>(v));
}

TEST(GraphEntityTest, AddTensorAndGet) {
  Graph g;
  EntityId id = g.AddTensor("t0", {1, 2, 3}, FLOAT_DTYPE, false);
  EXPECT_GE(id, 0);

  TensorEntity *t = g.GetTensor(id);
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(t->name, "t0");
  EXPECT_EQ(t->shape, std::vector<int64_t>({1, 2, 3}));
  EXPECT_FALSE(t->is_initializer);
  EXPECT_EQ(t->dtype, FLOAT_DTYPE);
}

TEST(GraphEntityTest, AddConstantAndInitializerFlag) {
  Graph g;

  TensorData data = MakeTensorData("w0", {4, 4});
  g.AddConstant(data);

  EntityId tId = g.AddTensor("w0", {4, 4}, data.dtype, true);
  TensorEntity *t = g.GetTensor(tId);
  ASSERT_NE(t, nullptr);
  EXPECT_TRUE(t->is_initializer);
  bool foundConstant = false;
  for (const auto &e : g.Entities()) {
    if (e.kind != EntityKind::Constant)
      continue;

    const auto &c = std::get<ConstantEntity>(e.entity);
    if (c.data.name == "w0") {
      foundConstant = true;
      EXPECT_EQ(c.data.dims, std::vector<int64_t>({4, 4}));
      EXPECT_EQ(c.data.dtype, FLOAT_DTYPE);
    }
  }

  EXPECT_TRUE(foundConstant);
}

TEST(GraphOperationTest, AddOperationCreatesCorrectEntity) {
  Graph g;
  EntityId a = g.AddTensor("a", {}, FLOAT_DTYPE);
  EntityId b = g.AddTensor("b", {}, FLOAT_DTYPE);
  EntityId out = g.AddTensor("out", {}, FLOAT_DTYPE);

  std::unordered_map<std::string, AttrValue> attrs;
  attrs["alpha"] = int64_t(2);

  EntityId opId = g.AddOperation("Add", "add1", {a, b}, {out}, attrs);
  EXPECT_EQ(opId, static_cast<EntityId>(opId));

  const auto &ents = g.Entities();
  ASSERT_GT(ents.size(), 0u);

  bool found = false;

  for (const auto &e : ents) {
    if (e.kind != EntityKind::Operation)
      continue;

    const auto &op = std::get<OperationEntity>(e.entity);

    if (op.name == "add1") {
      found = true;

      EXPECT_EQ(op.op_type, "Add");
      EXPECT_EQ(op.inputs.size(), 2u);
      EXPECT_EQ(op.outputs.size(), 1u);

      EXPECT_EQ(op.inputs[0], a);
      EXPECT_EQ(op.inputs[1], b);

      auto it = op.attrs.find("alpha");
      ASSERT_NE(it, op.attrs.end());

      ASSERT_TRUE(std::holds_alternative<AttrScalar>(it->second));
      EXPECT_EQ(GetScalar<int64_t>(it->second), 2);
    }
  }

  EXPECT_TRUE(found);
}

TEST(GraphExporterTest, DumpGraphIncludesConstantsAndIO) {
  Graph g;

  EntityId image = g.AddTensor("image", {}, FLOAT_DTYPE);
  EntityId vec = g.AddTensor("vector", {128}, FLOAT_DTYPE);

  TensorData conv_w = MakeTensorData("conv_w", {64, 3, 7, 7});
  g.AddConstant(conv_w);
  EntityId conv_w_t = g.AddTensor("conv_w", {64, 3, 7, 7}, conv_w.dtype, true);

  TensorData conv_b = MakeTensorData("conv_b", {64});
  g.AddConstant(conv_b);
  EntityId conv_b_t = g.AddTensor("conv_b", {64}, conv_b.dtype, true);

  EntityId c = g.AddTensor("c", {}, FLOAT_DTYPE);
  g.AddOperation("Conv", "conv1", {image, conv_w_t, conv_b_t}, {c}, {});

  EntityId r = g.AddTensor("r", {}, FLOAT_DTYPE);
  g.AddOperation("Relu", "relu1", {c}, {r}, {});

  EntityId out = g.AddTensor("output", {}, FLOAT_DTYPE);
  g.AddOperation("Identity", "out_op", {r}, {out}, {});

  g.AddInput(image);
  g.AddOutput(out);

  std::string dump = DumpGraph(g);

  EXPECT_NE(dump.find("Inputs:"), std::string::npos);
  EXPECT_NE(dump.find("Operations:"), std::string::npos);
  EXPECT_NE(dump.find("Outputs:"), std::string::npos);

  EXPECT_NE(dump.find("conv_w"), std::string::npos);
  EXPECT_NE(dump.find("conv_b"), std::string::npos);

  EXPECT_NE(dump.find("image"), std::string::npos);
  EXPECT_NE(dump.find("vector"), std::string::npos);

  EXPECT_NE(dump.find("output"), std::string::npos);
}

TEST(GraphExporterTest, ToDotProducesEdges) {
  Graph g;

  EntityId t0 = g.AddTensor("t0", {}, FLOAT_DTYPE);
  EntityId t1 = g.AddTensor("t1", {}, FLOAT_DTYPE);
  EntityId out = g.AddTensor("out", {}, FLOAT_DTYPE);

  g.AddOperation("OpA", "opA", {t0, t1}, {out}, {});

  std::string dot = ToDot(g);

  EXPECT_NE(dot.find("t0"), std::string::npos);
  EXPECT_NE(dot.find("op"), std::string::npos);
  EXPECT_NE(dot.find("->"), std::string::npos);
}