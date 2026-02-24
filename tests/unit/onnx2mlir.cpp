#include "TensorCompiler/Frontend/ONNXModel.hpp"
#include "TensorCompiler/Frontend/ONNXDumper.hpp"
#include "TensorCompiler/Driver/MLIRBuilder.hpp"
#include "TensorCompiler/Dialect/NNDialect.hpp"

#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Verifier.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/Casting.h"

#include <gtest/gtest.h>

static mlir::ModuleOp
ONNXtoMLIR(mlir::MLIRContext &ctx, std::string_view fileName) {
  ctx.loadDialect<
      mlir::arith::ArithDialect,
      mlir::func::FuncDialect,
      mlir::nn::NNDialect>();

  ONNXModel model{fileName};
  MLIRBuilder builder(ctx);
  model.Parse(builder);

  return builder.GetModule();
}

static bool hasOp(mlir::ModuleOp &m, llvm::StringRef name) {
  bool found = false;
  m.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == name)
      found = true;
  });
  return found;
}

static bool hasFunc(mlir::ModuleOp &m, llvm::StringRef name) {
  bool found = false;
  m.walk([&](mlir::func::FuncOp func) {
    if (func.getName() == name)
      found = true;
  });
  return found;
}

TEST(ONNXtoMLIR, AddMul) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/add_mul.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  EXPECT_TRUE(hasFunc(mdl, "AddMulGraph"));
  EXPECT_TRUE(hasOp(mdl, "nn.add"));
  EXPECT_TRUE(hasOp(mdl, "nn.mul"));
}

TEST(ONNXtoMLIR, ConvRelu) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/conv_relu.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  EXPECT_TRUE(hasFunc(mdl, "ConvReluGraph"));
  EXPECT_TRUE(hasOp(mdl, "nn.conv"));
  EXPECT_TRUE(hasOp(mdl, "nn.relu"));
}

TEST(ONNXtoMLIR, MatMulRelu) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/matmul_relu.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  EXPECT_TRUE(hasFunc(mdl, "MatMulReluGraph"));
  EXPECT_TRUE(hasOp(mdl, "nn.matmul"));
  EXPECT_TRUE(hasOp(mdl, "nn.relu"));
}

TEST(ONNXtoMLIR, SingleRelu) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/relu.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  EXPECT_TRUE(hasFunc(mdl, "SingleRelu"));
  EXPECT_TRUE(hasOp(mdl, "nn.relu"));
}

TEST(ONNXtoMLIR, TransposeMatMul) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/transpose_matmul.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  EXPECT_TRUE(hasFunc(mdl, "TransposeMatMulGraph"));
  EXPECT_TRUE(hasOp(mdl, "nn.transpose"));
  EXPECT_TRUE(hasOp(mdl, "nn.matmul"));
}

TEST(ONNXtoMLIR, Gemm) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/gemm.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  EXPECT_TRUE(hasFunc(mdl, "GemmGraph"));
  EXPECT_TRUE(hasOp(mdl, "nn.gemm"));
}

TEST(ONNXtoMLIR, Pipeline) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/test_0.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  EXPECT_TRUE(hasFunc(mdl, "PipelineGraph"));
  EXPECT_TRUE(hasOp(mdl, "nn.transpose"));
  EXPECT_TRUE(hasOp(mdl, "nn.matmul"));
  EXPECT_TRUE(hasOp(mdl, "nn.relu"));
  EXPECT_TRUE(hasOp(mdl, "nn.conv"));
  EXPECT_TRUE(hasOp(mdl, "nn.add"));
  EXPECT_TRUE(hasOp(mdl, "nn.mul"));
  EXPECT_TRUE(hasOp(mdl, "nn.gemm"));
}

TEST(ONNXtoMLIR, WeightsAndBias) {
  mlir::MLIRContext ctx;
  auto mdl = ONNXtoMLIR(ctx, "tests/data/test_1.onnx");
  EXPECT_TRUE(mlir::succeeded(mdl.verify()));

  bool foundWeights = false;
  bool foundBias = false;

  mdl.walk([&](mlir::arith::ConstantOp cst) {
    auto type =
      llvm::cast<mlir::RankedTensorType>(cst.getType());
    auto dense =
      llvm::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue());
    
    ASSERT_TRUE(dense);

    if (type.getShape() == llvm::ArrayRef<int64_t>{3,4}) {
      std::vector<float> values(dense.getValues<float>().begin(),
                                dense.getValues<float>().end());
      for (size_t i = 0; i < 12; ++i) {
        EXPECT_EQ(values[i], i + 1);  
      }
      foundWeights = true;
    }

    if (type.getShape() == llvm::ArrayRef<int64_t>{4}) {
      EXPECT_TRUE(dense.isSplat());
      EXPECT_FLOAT_EQ(dense.getSplatValue<float>(), 1.0f);
      foundBias = true;
    }
  });

  EXPECT_TRUE(foundWeights);
  EXPECT_TRUE(foundBias);
}