#include "TensorCompiler/Converter/MLIRBuilder.hpp"
#include "TensorCompiler/Dialect/NNDialect.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace tc::converter::onnx_to_high_mlir {
static const onnx::AttributeProto *FindAttr(const onnx::NodeProto &node,
                                            std::string_view name) {
  for (const auto &attr : node.attribute())
    if (attr.name() == name)
      return &attr;
  return nullptr;
}

static int64_t GetAttrInt(const onnx::NodeProto &node, std::string_view name,
                          int64_t defaultVal) {
  const auto *a = FindAttr(node, name);
  return (a && a->type() == onnx::AttributeProto::INT) ? a->i() : defaultVal;
}

static float GetAttrFloat(const onnx::NodeProto &node, std::string_view name,
                          float defaultVal) {
  const auto *a = FindAttr(node, name);
  return (a && a->type() == onnx::AttributeProto::FLOAT) ? a->f() : defaultVal;
}

static std::string getAttrString(const onnx::NodeProto &node,
                                 std::string_view name,
                                 std::string_view defaultVal) {
  const auto *a = FindAttr(node, name);
  return (a && a->type() == onnx::AttributeProto::STRING)
             ? a->s()
             : std::string(defaultVal);
}

static llvm::SmallVector<int64_t> GetAttrInts(const onnx::NodeProto &node,
                                              std::string_view name,
                                              size_t fillLen = 0,
                                              int64_t fillVal = 0) {
  const auto *a = FindAttr(node, name);
  if (a && a->type() == onnx::AttributeProto::INTS)
    return {a->ints().begin(), a->ints().end()};
  return llvm::SmallVector<int64_t>(fillLen, fillVal);
}

mlir::Value MLIRBuilder::FindValue(const std::string &name) const {
  if (name.empty())
    return mlir::Value{};
  auto it = valueMap_.find(name);
  return (it != valueMap_.end()) ? it->second : mlir::Value{};
}

mlir::Type MLIRBuilder::ConvertElemType(int onnx_dtype) {
  switch (onnx_dtype) {
  case onnx::TensorProto::FLOAT:
    return builder_.getF32Type();
  case onnx::TensorProto::DOUBLE:
    return builder_.getF64Type();
  case onnx::TensorProto::INT32:
    return builder_.getI32Type();
  case onnx::TensorProto::INT64:
    return builder_.getI64Type();
  case onnx::TensorProto::INT8:
    return builder_.getI8Type();
  case onnx::TensorProto::UINT8:
    return builder_.getIntegerType(8, false);
  case onnx::TensorProto::BOOL:
    return builder_.getI1Type();
  default:
    llvm::errs() << "Unsupported dtype: " << onnx_dtype << "\n";
    llvm_unreachable("unsupported tensor dtype");
  }
}

mlir::RankedTensorType
MLIRBuilder::ConvertTensorType(int onnx_dtype, llvm::ArrayRef<int64_t> shape) {
  return mlir::RankedTensorType::get(shape, ConvertElemType(onnx_dtype));
}

mlir::DenseElementsAttr
MLIRBuilder::ConvertTensor(const onnx::TensorProto &tensor) {
  llvm::SmallVector<int64_t> shape;
  for (size_t i = 0; i < tensor.dims_size(); ++i)
    shape.push_back(tensor.dims(i));
  auto type = ConvertTensorType(tensor.data_type(), shape);

  if (tensor.float_data_size() > 0) {
    llvm::SmallVector<float> data(tensor.float_data().begin(),
                                  tensor.float_data().end());
    return mlir::DenseElementsAttr::get(type, llvm::ArrayRef(data));
  }
  if (tensor.has_raw_data()) {
    const auto &raw = tensor.raw_data();
    switch (tensor.data_type()) {
    case onnx::TensorProto::FLOAT: {
      auto *p = reinterpret_cast<const float *>(raw.data());
      return mlir::DenseElementsAttr::get(
          type, llvm::ArrayRef(p, raw.size() / sizeof(float)));
    }
    case onnx::TensorProto::INT64: {
      auto *p = reinterpret_cast<const int64_t *>(raw.data());
      return mlir::DenseElementsAttr::get(
          type, llvm::ArrayRef(p, raw.size() / sizeof(int64_t)));
    }
    case onnx::TensorProto::INT32: {
      auto *p = reinterpret_cast<const int32_t *>(raw.data());
      return mlir::DenseElementsAttr::get(
          type, llvm::ArrayRef(p, raw.size() / sizeof(int32_t)));
    }
    default:
      llvm_unreachable("unsupported raw data type");
    }
  }
  if (tensor.int64_data_size() > 0) {
    llvm::SmallVector<int64_t> data(tensor.int64_data().begin(),
                                    tensor.int64_data().end());
    return mlir::DenseElementsAttr::get(type, llvm::ArrayRef(data));
  }
  if (tensor.int32_data_size() > 0) {
    llvm::SmallVector<int32_t> data(tensor.int32_data().begin(),
                                    tensor.int32_data().end());
    return mlir::DenseElementsAttr::get(type, llvm::ArrayRef(data));
  }
  llvm_unreachable("unsupported tensor data format");
}

mlir::RankedTensorType
MLIRBuilder::ConvertValueInfo(const onnx::ValueInfoProto &info) {
  assert(info.has_type() && info.type().has_tensor_type());
  const auto &tt = info.type().tensor_type();
  llvm::SmallVector<int64_t> shape;
  if (tt.has_shape())
    for (const auto &dim : tt.shape().dim())
      shape.push_back(dim.has_dim_value() ? dim.dim_value() : -1);
  return ConvertTensorType(tt.elem_type(), shape);
}

void MLIRBuilder::BuildRelu(const onnx::NodeProto &node, mlir::Location loc) {
  mlir::Value in = valueMap_.at(node.input(0));
  auto op = mlir::nn::ReluOp::create(builder_, loc, in.getType(), in);
  valueMap_[node.output(0)] = op.getResult();
}

void MLIRBuilder::BuildAdd(const onnx::NodeProto &node, mlir::Location loc) {
  mlir::Value lhs = valueMap_.at(node.input(0));
  mlir::Value rhs = valueMap_.at(node.input(1));
  auto lhsTy = llvm::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsTy = llvm::cast<mlir::RankedTensorType>(rhs.getType());
  mlir::Type resultType =
      lhsTy.getRank() >= rhsTy.getRank() ? lhs.getType() : rhs.getType();
  auto op = mlir::nn::AddOp::create(builder_, loc, resultType, lhs, rhs);
  valueMap_[node.output(0)] = op.getResult();
}

void MLIRBuilder::BuildMul(const onnx::NodeProto &node, mlir::Location loc) {
  mlir::Value lhs = valueMap_.at(node.input(0));
  mlir::Value rhs = valueMap_.at(node.input(1));
  auto lhsTy = llvm::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsTy = llvm::cast<mlir::RankedTensorType>(rhs.getType());
  mlir::Type resultType =
      lhsTy.getRank() >= rhsTy.getRank() ? lhs.getType() : rhs.getType();
  auto op = mlir::nn::MulOp::create(builder_, loc, resultType, lhs, rhs);
  valueMap_[node.output(0)] = op.getResult();
}

void MLIRBuilder::BuildMatMul(const onnx::NodeProto &node, mlir::Location loc) {
  mlir::Value A = valueMap_.at(node.input(0));
  mlir::Value B = valueMap_.at(node.input(1));
  auto aTy = llvm::cast<mlir::RankedTensorType>(A.getType());
  auto bTy = llvm::cast<mlir::RankedTensorType>(B.getType());
  llvm::SmallVector<int64_t> shape(aTy.getShape());
  shape.back() = bTy.getShape().back();
  auto resultType = mlir::RankedTensorType::get(shape, aTy.getElementType());
  auto op = mlir::nn::MatMulOp::create(builder_, loc, resultType, A, B);
  valueMap_[node.output(0)] = op.getResult();
}

void MLIRBuilder::BuildGemm(const onnx::NodeProto &node, mlir::Location loc) {
  mlir::Value A = valueMap_.at(node.input(0));
  mlir::Value B = valueMap_.at(node.input(1));
  mlir::Value C =
      (node.input_size() > 2) ? FindValue(node.input(2)) : mlir::Value{};

  int64_t transA = GetAttrInt(node, "transA", 0);
  int64_t transB = GetAttrInt(node, "transB", 0);
  float alpha = GetAttrFloat(node, "alpha", 1.0f);
  float beta = GetAttrFloat(node, "beta", 1.0f);

  auto aTy = llvm::cast<mlir::RankedTensorType>(A.getType());
  auto bTy = llvm::cast<mlir::RankedTensorType>(B.getType());
  int64_t M = transA ? aTy.getDimSize(1) : aTy.getDimSize(0);
  int64_t N = transB ? bTy.getDimSize(0) : bTy.getDimSize(1);
  auto resultType = mlir::RankedTensorType::get({M, N}, aTy.getElementType());

  auto op = mlir::nn::GemmOp::create(
      builder_, loc, resultType, A, B, C, builder_.getI64IntegerAttr(transA),
      builder_.getI64IntegerAttr(transB), builder_.getF32FloatAttr(alpha),
      builder_.getF32FloatAttr(beta));
  valueMap_[node.output(0)] = op.getResult();
}

void MLIRBuilder::BuildConv(const onnx::NodeProto &node, mlir::Location loc) {
  mlir::Value input = valueMap_.at(node.input(0));
  mlir::Value weight = valueMap_.at(node.input(1));
  mlir::Value bias =
      (node.input_size() > 2) ? FindValue(node.input(2)) : mlir::Value{};

  auto inputTy = llvm::cast<mlir::RankedTensorType>(input.getType());
  auto weightTy = llvm::cast<mlir::RankedTensorType>(weight.getType());

  int64_t spatialDims = inputTy.getRank() - 2;

  auto strides = GetAttrInts(node, "strides", spatialDims, 1);
  auto dilations = GetAttrInts(node, "dilations", spatialDims, 1);
  auto pads = GetAttrInts(node, "pads", spatialDims * 2, 0);
  auto group = GetAttrInt(node, "group", 1);
  auto autoPad = getAttrString(node, "auto_pad", "NOTSET");

  int64_t N = inputTy.getDimSize(0);
  int64_t Cout = weightTy.getDimSize(0);

  auto inputShape = inputTy.getShape();
  auto kernelShape = weightTy.getShape();

  llvm::SmallVector<int64_t> outputShape = {N, Cout};

  for (int i = 0; i < spatialDims; ++i) {
    int64_t in = inputShape[2 + i];
    int64_t k = kernelShape[2 + i];
    int64_t pb = pads[i];
    int64_t pe = pads[i + spatialDims];
    int64_t s = strides[i];
    int64_t d = dilations[i];

    if (in == mlir::ShapedType::kDynamic || k == mlir::ShapedType::kDynamic) {
      outputShape.push_back(mlir::ShapedType::kDynamic);
      continue;
    }

    int64_t out = (in + pb + pe - d * (k - 1) - 1) / s + 1;
    outputShape.push_back(out);
  }

  auto resultType =
      mlir::RankedTensorType::get(outputShape, inputTy.getElementType());

  if (bias) {
    auto biasTy = llvm::cast<mlir::RankedTensorType>(bias.getType());
    if (biasTy.getRank() != 1 || biasTy.getDimSize(0) != Cout) {
      llvm::errs() << "Conv bias must have shape [" << Cout << "]\n";
      llvm::report_fatal_error("Invalid Conv bias shape");
    }
  }

  auto op = mlir::nn::ConvOp::create(
      builder_, loc, resultType, input, weight, bias,
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), strides),
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), dilations),
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), pads),
      builder_.getI64IntegerAttr(group), builder_.getStringAttr(autoPad));

  valueMap_[node.output(0)] = op.getResult();
}

void MLIRBuilder::BuildTranspose(const onnx::NodeProto &node,
                                 mlir::Location loc) {
  mlir::Value input = valueMap_.at(node.input(0));
  auto inputTy = llvm::cast<mlir::RankedTensorType>(input.getType());
  int64_t rank = inputTy.getRank();

  auto perm = GetAttrInts(node, "perm", 0, 0);
  if (perm.empty()) {
    perm.resize(rank);
    for (int64_t i = 0; i < rank; ++i)
      perm[i] = rank - 1 - i;
  }

  llvm::SmallVector<int64_t> outShape(rank);
  for (int64_t i = 0; i < rank; ++i)
    outShape[i] = inputTy.getDimSize(perm[i]);
  auto resultType =
      mlir::RankedTensorType::get(outShape, inputTy.getElementType());

  auto op = mlir::nn::TransposeOp::create(
      builder_, loc, resultType, input,
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), perm));
  valueMap_[node.output(0)] = op.getResult();
}

MLIRBuilder::MLIRBuilder(mlir::MLIRContext &ctx)
    : ctx_(ctx), builder_(&ctx_),
      module_(mlir::ModuleOp::create(builder_.getUnknownLoc())) {
  builder_.setInsertionPointToEnd(module_.getBody());
}

const mlir::ModuleOp &MLIRBuilder::GetModule() const { return module_; }

void MLIRBuilder::Visit(const onnx::ModelProto &) { /* Do nothing */ }

void MLIRBuilder::Visit(const onnx::GraphProto &graph) {
  for (const auto &init : graph.initializer())
    initializerNames_.insert(init.name());

  llvm::SmallVector<mlir::Type> inputTypes, outputTypes;
  for (const auto &in : graph.input())
    if (!initializerNames_.count(in.name()))
      inputTypes.push_back(ConvertValueInfo(in));
  for (const auto &out : graph.output())
    outputTypes.push_back(ConvertValueInfo(out));

  auto funcName = graph.name().empty() ? "main" : graph.name();
  auto func = mlir::func::FuncOp::create(
      builder_.getUnknownLoc(), funcName,
      builder_.getFunctionType(inputTypes, outputTypes));
  builder_.insert(func);

  auto *block = func.addEntryBlock();
  builder_.setInsertionPointToStart(block);

  unsigned argIdx = 0;
  for (const auto &in : graph.input())
    if (!initializerNames_.count(in.name()))
      valueMap_[in.name()] = block->getArgument(argIdx++);
}

void MLIRBuilder::Visit(const onnx::ValueInfoProto &) { /* Do nothing */ }

void MLIRBuilder::Visit(const onnx::TensorProto &tensor) {
  auto attr = ConvertTensor(tensor);
  auto op =
      mlir::arith::ConstantOp::create(builder_, builder_.getUnknownLoc(), attr);
  valueMap_[tensor.name()] = op.getResult();
}

void MLIRBuilder::Visit(const onnx::NodeProto &node) {
  auto loc = builder_.getUnknownLoc();
  const auto &opType = node.op_type();

  if (opType == "Relu")
    BuildRelu(node, loc);
  else if (opType == "Add")
    BuildAdd(node, loc);
  else if (opType == "Mul")
    BuildMul(node, loc);
  else if (opType == "MatMul")
    BuildMatMul(node, loc);
  else if (opType == "Gemm")
    BuildGemm(node, loc);
  else if (opType == "Conv")
    BuildConv(node, loc);
  else if (opType == "Transpose")
    BuildTranspose(node, loc);
  else
    llvm::errs() << "[warn] Unsupported op: " << opType << "\n";
}

void MLIRBuilder::Visit(const onnx::AttributeProto &) { /* Do nothing */ }

void MLIRBuilder::Finalize(const onnx::GraphProto &graph) {
  llvm::SmallVector<mlir::Value> outputs;
  for (size_t i = 0; i < graph.output_size(); ++i)
    outputs.push_back(valueMap_.at(graph.output(i).name()));
  mlir::func::ReturnOp::create(builder_, builder_.getUnknownLoc(), outputs);
}
} // namespace tc::converter::onnx_to_high_mlir