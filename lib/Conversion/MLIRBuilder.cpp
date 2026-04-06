#include "TensorCompiler/Conversion/MLIRBuilder.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>

#include <llvm/Support/ErrorHandling.h>

#include <onnx/onnx_pb.h>

namespace tc::conversion {
using tc::graph::AttrValue;
using tc::graph::DoubleList;
using tc::graph::EntityId;
using tc::graph::IntList;
using tc::graph::StringList;
using tc::graph::TensorData;

static const AttrValue *FindAttr(const tc::graph::OperationEntity &op,
                                 std::string_view name) {
  auto it = op.attrs.find(std::string(name));
  return (it != op.attrs.end()) ? &it->second : nullptr;
}

static int64_t GetAttrInt(const tc::graph::OperationEntity &op,
                          std::string_view name, int64_t defaultVal) {
  const auto *attr = FindAttr(op, name);
  if (attr && std::holds_alternative<tc::graph::AttrScalar>(*attr)) {
    const auto &scalar = std::get<tc::graph::AttrScalar>(*attr);
    if (std::holds_alternative<int64_t>(scalar))
      return std::get<int64_t>(scalar);
  }
  return defaultVal;
}

static float GetAttrFloat(const tc::graph::OperationEntity &op,
                          std::string_view name, float defaultVal) {
  const auto *attr = FindAttr(op, name);
  if (attr && std::holds_alternative<tc::graph::AttrScalar>(*attr)) {
    const auto &scalar = std::get<tc::graph::AttrScalar>(*attr);
    if (std::holds_alternative<double>(scalar))
      return static_cast<float>(std::get<double>(scalar));
    if (std::holds_alternative<int64_t>(scalar))
      return static_cast<float>(std::get<int64_t>(scalar));
  }
  return defaultVal;
}

static std::string GetAttrString(const tc::graph::OperationEntity &op,
                                 std::string_view name,
                                 std::string_view defaultVal) {
  const auto *attr = FindAttr(op, name);
  if (attr && std::holds_alternative<tc::graph::AttrScalar>(*attr)) {
    const auto &scalar = std::get<tc::graph::AttrScalar>(*attr);
    if (std::holds_alternative<std::string>(scalar))
      return std::get<std::string>(scalar);
  }
  return std::string(defaultVal);
}

static llvm::SmallVector<int64_t>
GetAttrInts(const tc::graph::OperationEntity &op, std::string_view name,
            size_t fillLen = 0, int64_t fillVal = 0) {
  const auto *attr = FindAttr(op, name);
  if (attr && std::holds_alternative<tc::graph::AttrList>(*attr)) {
    const auto &list = std::get<tc::graph::AttrList>(*attr);
    if (std::holds_alternative<tc::graph::IntList>(list)) {
      const auto &vec = std::get<tc::graph::IntList>(list);
      return {vec.begin(), vec.end()};
    }
  }
  return llvm::SmallVector<int64_t>(fillLen, fillVal);
}

HighLevelMLIRBuilder::HighLevelMLIRBuilder(mlir::MLIRContext &ctx)
    : ctx_(ctx), builder_(&ctx_),
      module_(mlir::ModuleOp::create(builder_.getUnknownLoc())) {
  builder_.setInsertionPointToEnd(module_.getBody());
}

mlir::Value HighLevelMLIRBuilder::GetValue(tc::graph::EntityId id) const {
  auto it = valueMap_.find(id);
  return (it != valueMap_.end()) ? it->second : mlir::Value{};
}

mlir::Value HighLevelMLIRBuilder::GetValue(const std::string &name) const {
  auto it = tensorNameToId_.find(name);
  if (it != tensorNameToId_.end())
    return GetValue(it->second);
  return mlir::Value{};
}

mlir::Type HighLevelMLIRBuilder::ConvertElemType(int32_t onnx_dtype) {
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
HighLevelMLIRBuilder::ConvertTensorType(int32_t dtype,
                                        llvm::ArrayRef<int64_t> shape) {
  return mlir::RankedTensorType::get(shape, ConvertElemType(dtype));
}

mlir::DenseElementsAttr
HighLevelMLIRBuilder::ConvertTensorData(const tc::graph::TensorData &data) {
  auto type = ConvertTensorType(data.dtype, data.dims);
  llvm::ArrayRef<uint8_t> raw(data.raw_data);

  switch (data.dtype) {
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

void HighLevelMLIRBuilder::BuildRelu(const tc::graph::OperationEntity &op,
                                     mlir::Location loc) {
  mlir::Value in = GetValue(op.inputs[0]);
  auto resultType = in.getType();
  auto mlirOp = mlir::nn::ReluOp::create(builder_, loc, resultType, in);
  valueMap_[op.outputs[0]] = mlirOp.getResult();
}

void HighLevelMLIRBuilder::BuildAdd(const tc::graph::OperationEntity &op,
                                    mlir::Location loc) {
  mlir::Value lhs = GetValue(op.inputs[0]);
  mlir::Value rhs = GetValue(op.inputs[1]);
  auto lhsTy = llvm::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsTy = llvm::cast<mlir::RankedTensorType>(rhs.getType());
  mlir::Type resultType =
      (lhsTy.getRank() >= rhsTy.getRank()) ? lhs.getType() : rhs.getType();
  auto mlirOp = mlir::nn::AddOp::create(builder_, loc, resultType, lhs, rhs);
  valueMap_[op.outputs[0]] = mlirOp.getResult();
}

void HighLevelMLIRBuilder::BuildMul(const tc::graph::OperationEntity &op,
                                    mlir::Location loc) {
  mlir::Value lhs = GetValue(op.inputs[0]);
  mlir::Value rhs = GetValue(op.inputs[1]);
  auto lhsTy = llvm::cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsTy = llvm::cast<mlir::RankedTensorType>(rhs.getType());
  mlir::Type resultType =
      (lhsTy.getRank() >= rhsTy.getRank()) ? lhs.getType() : rhs.getType();
  auto mlirOp = mlir::nn::MulOp::create(builder_, loc, resultType, lhs, rhs);
  valueMap_[op.outputs[0]] = mlirOp.getResult();
}

void HighLevelMLIRBuilder::BuildMatMul(const tc::graph::OperationEntity &op,
                                       mlir::Location loc) {
  mlir::Value A = GetValue(op.inputs[0]);
  mlir::Value B = GetValue(op.inputs[1]);
  auto aTy = llvm::cast<mlir::RankedTensorType>(A.getType());
  auto bTy = llvm::cast<mlir::RankedTensorType>(B.getType());
  llvm::SmallVector<int64_t> shape(aTy.getShape());
  shape.back() = bTy.getShape().back();
  auto resultType = mlir::RankedTensorType::get(shape, aTy.getElementType());
  auto mlirOp = mlir::nn::MatMulOp::create(builder_, loc, resultType, A, B);
  valueMap_[op.outputs[0]] = mlirOp.getResult();
}

void HighLevelMLIRBuilder::BuildGemm(const tc::graph::OperationEntity &op,
                                     mlir::Location loc) {
  mlir::Value A = GetValue(op.inputs[0]);
  mlir::Value B = GetValue(op.inputs[1]);
  mlir::Value C =
      (op.inputs.size() > 2) ? GetValue(op.inputs[2]) : mlir::Value{};

  int64_t transA = GetAttrInt(op, "transA", 0);
  int64_t transB = GetAttrInt(op, "transB", 0);
  float alpha = GetAttrFloat(op, "alpha", 1.0f);
  float beta = GetAttrFloat(op, "beta", 1.0f);

  auto aTy = llvm::cast<mlir::RankedTensorType>(A.getType());
  auto bTy = llvm::cast<mlir::RankedTensorType>(B.getType());
  int64_t M = transA ? aTy.getDimSize(1) : aTy.getDimSize(0);
  int64_t N = transB ? bTy.getDimSize(0) : bTy.getDimSize(1);
  auto resultType = mlir::RankedTensorType::get({M, N}, aTy.getElementType());

  auto mlirOp = mlir::nn::GemmOp::create(
      builder_, loc, resultType, A, B, C, builder_.getI64IntegerAttr(transA),
      builder_.getI64IntegerAttr(transB), builder_.getF32FloatAttr(alpha),
      builder_.getF32FloatAttr(beta));
  valueMap_[op.outputs[0]] = mlirOp.getResult();
}

void HighLevelMLIRBuilder::BuildConv(const tc::graph::OperationEntity &op,
                                     mlir::Location loc) {
  mlir::Value input = GetValue(op.inputs[0]);
  mlir::Value weight = GetValue(op.inputs[1]);
  mlir::Value bias =
      (op.inputs.size() > 2) ? GetValue(op.inputs[2]) : mlir::Value{};

  auto inputTy = llvm::cast<mlir::RankedTensorType>(input.getType());
  auto weightTy = llvm::cast<mlir::RankedTensorType>(weight.getType());

  int64_t spatialDims = inputTy.getRank() - 2;

  auto strides = GetAttrInts(op, "strides", spatialDims, 1);
  auto dilations = GetAttrInts(op, "dilations", spatialDims, 1);
  auto pads = GetAttrInts(op, "pads", spatialDims * 2, 0);
  auto group = GetAttrInt(op, "group", 1);
  auto autoPad = GetAttrString(op, "auto_pad", "NOTSET");

  int64_t N = inputTy.getDimSize(0);
  int64_t Cout = weightTy.getDimSize(0);

  llvm::SmallVector<int64_t> outputShape = {N, Cout};
  for (int i = 0; i < spatialDims; ++i) {
    int64_t in = inputTy.getDimSize(2 + i);
    int64_t k = weightTy.getDimSize(2 + i);
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

  auto mlirOp = mlir::nn::ConvOp::create(
      builder_, loc, resultType, input, weight, bias,
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), strides),
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), dilations),
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), pads),
      builder_.getI64IntegerAttr(group), builder_.getStringAttr(autoPad));
  valueMap_[op.outputs[0]] = mlirOp.getResult();
}

void HighLevelMLIRBuilder::BuildTranspose(const tc::graph::OperationEntity &op,
                                          mlir::Location loc) {
  mlir::Value input = GetValue(op.inputs[0]);
  auto inputTy = llvm::cast<mlir::RankedTensorType>(input.getType());
  int64_t rank = inputTy.getRank();

  auto perm = GetAttrInts(op, "perm", 0, 0);
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

  auto mlirOp = mlir::nn::TransposeOp::create(
      builder_, loc, resultType, input,
      mlir::DenseI64ArrayAttr::get(builder_.getContext(), perm));
  valueMap_[op.outputs[0]] = mlirOp.getResult();
}

void HighLevelMLIRBuilder::Visit(const tc::graph::Graph &graph) {
  llvm::SmallVector<mlir::Type> inputTypes, outputTypes;
  for (tc::graph::EntityId id : graph.Inputs()) {
    const auto *tensor = graph.GetTensor(id);
    if (tensor && !tensor->is_initializer) {
      inputTypes.push_back(ConvertTensorType(tensor->dtype, tensor->shape));
    }
  }
  for (tc::graph::EntityId id : graph.Outputs()) {
    const auto *tensor = graph.GetTensor(id);
    if (tensor) {
      outputTypes.push_back(ConvertTensorType(tensor->dtype, tensor->shape));
    }
  }

  std::string funcName = "main";
  auto func = mlir::func::FuncOp::create(
      builder_.getUnknownLoc(), funcName,
      builder_.getFunctionType(inputTypes, outputTypes));
  builder_.insert(func);

  auto *block = func.addEntryBlock();
  builder_.setInsertionPointToStart(block);

  size_t argIdx = 0;
  for (tc::graph::EntityId id : graph.Inputs()) {
    const auto *tensor = graph.GetTensor(id);
    if (tensor && !tensor->is_initializer) {
      valueMap_[id] = block->getArgument(argIdx++);
      tensorNameToId_[tensor->name] = id;
    }
  }
}

void HighLevelMLIRBuilder::Visit(const tc::graph::TensorEntity &tensor) {
  if (!tensor.is_initializer || !tensor.data.has_value())
    return;
  auto it = valueMap_.find(tensor.id);
  if (it != valueMap_.end())
    return;
  auto attr = ConvertTensorData(*tensor.data);
  auto op =
      mlir::arith::ConstantOp::create(builder_, builder_.getUnknownLoc(), attr);
  valueMap_[tensor.id] = op.getResult();
  tensorNameToId_[tensor.name] = tensor.id;
}

void HighLevelMLIRBuilder::Visit(const tc::graph::OperationEntity &op) {
  auto loc = builder_.getUnknownLoc();
  const auto &opType = op.op_type;

  if (opType == "Relu")
    BuildRelu(op, loc);
  else if (opType == "Add")
    BuildAdd(op, loc);
  else if (opType == "Mul")
    BuildMul(op, loc);
  else if (opType == "MatMul")
    BuildMatMul(op, loc);
  else if (opType == "Gemm")
    BuildGemm(op, loc);
  else if (opType == "Conv")
    BuildConv(op, loc);
  else if (opType == "Transpose")
    BuildTranspose(op, loc);
  else
    llvm::errs() << "[warn] Unsupported op: " << opType << "\n";
}

void HighLevelMLIRBuilder::Visit(const tc::graph::ConstantEntity &constant) {
  /* Do nothing */
}

void HighLevelMLIRBuilder::Finalize(const tc::graph::Graph &graph) {
  llvm::SmallVector<mlir::Value> outputs;
  for (tc::graph::EntityId id : graph.Outputs()) {
    auto it = valueMap_.find(id);
    if (it != valueMap_.end())
      outputs.push_back(it->second);
    else
      llvm::errs() << "Warning: output " << id << " not found in valueMap\n";
  }
  mlir::func::ReturnOp::create(builder_, builder_.getUnknownLoc(), outputs);
}
} // namespace tc::conversion