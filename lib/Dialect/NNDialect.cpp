#include "TensorCompiler/Dialect/NNDialect.hpp"
#include <mlir/IR/Builders.h>
#include <mlir/IR/OpImplementation.h>

using namespace mlir;
using namespace mlir::nn;

#include "NNDialect.cpp.inc"

void NNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NNOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "NNOps.cpp.inc"

static RankedTensorType getRankedTensor(Value v) {
  return llvm::dyn_cast<RankedTensorType>(v.getType());
}

static bool sameElementType(Type a, Type b) {
  return llvm::cast<ShapedType>(a).getElementType() ==
         llvm::cast<ShapedType>(b).getElementType();
}

static LogicalResult verifyBroadcastOp(Operation *op, RankedTensorType lhsTy,
                                       RankedTensorType rhsTy,
                                       RankedTensorType outTy) {
  if (!lhsTy || !rhsTy || !outTy)
    return op->emitOpError("requires ranked tensor types");

  if (!sameElementType(lhsTy, rhsTy) || !sameElementType(lhsTy, outTy))
    return op->emitOpError("element types must match");

  int64_t rL = lhsTy.getRank();
  int64_t rR = rhsTy.getRank();
  int64_t rO = outTy.getRank();

  int64_t rMax = std::max(rL, rR);
  if (rO != rMax)
    return op->emitOpError("output rank must equal broadcast rank");

  for (size_t i = 0; i < rMax; ++i) {
    int64_t dL = (i < rMax - rL) ? 1 : lhsTy.getDimSize(i - (rMax - rL));
    int64_t dR = (i < rMax - rR) ? 1 : rhsTy.getDimSize(i - (rMax - rR));
    int64_t dO = outTy.getDimSize(i);

    if (!(dL == dR || dL == 1 || dR == 1 || dL == ShapedType::kDynamic ||
          dR == ShapedType::kDynamic))
      return op->emitOpError("operands are not broadcast-compatible");

    int64_t exp = (dL == ShapedType::kDynamic || dR == ShapedType::kDynamic)
                      ? ShapedType::kDynamic
                      : std::max(dL, dR);

    if (exp != ShapedType::kDynamic && dO != ShapedType::kDynamic && dO != exp)
      return op->emitOpError("incorrect broadcast dimension at axis ") << i;
  }

  return success();
}

LogicalResult ReluOp::verify() {
  auto inTy = getRankedTensor(getInput());
  auto outTy = getRankedTensor(getOutput());

  if (!inTy || !outTy)
    return emitOpError("requires ranked tensor types");

  if (inTy != outTy)
    return emitOpError("input and output types must match");

  return success();
}

LogicalResult AddOp::verify() {
  auto lhsTy = getRankedTensor(getLhs());
  auto rhsTy = getRankedTensor(getRhs());
  auto outTy = getRankedTensor(getOutput());

  return verifyBroadcastOp(*this, lhsTy, rhsTy, outTy);
}

LogicalResult MulOp::verify() {
  auto lhsTy = getRankedTensor(getLhs());
  auto rhsTy = getRankedTensor(getRhs());
  auto outTy = getRankedTensor(getOutput());

  return verifyBroadcastOp(*this, lhsTy, rhsTy, outTy);
}

LogicalResult MatMulOp::verify() {
  auto aTy = getRankedTensor(getA());
  auto bTy = getRankedTensor(getB());
  auto outTy = getRankedTensor(getOutput());

  if (!aTy || !bTy || !outTy)
    return emitOpError("requires ranked tensor types");

  if (aTy.getRank() < 2 || bTy.getRank() < 2)
    return emitOpError("operands must be at least 2D");

  if (!sameElementType(aTy, bTy) || !sameElementType(aTy, outTy))
    return emitOpError("element types must match");

  int64_t kA = aTy.getDimSize(aTy.getRank() - 1);
  int64_t kB = bTy.getDimSize(bTy.getRank() - 2);

  if (kA != ShapedType::kDynamic && kB != ShapedType::kDynamic && kA != kB)
    return emitOpError("inner dimensions must match");

  return success();
}

LogicalResult GemmOp::verify() {
  auto aTy = getRankedTensor(getA());
  auto bTy = getRankedTensor(getB());
  auto outTy = getRankedTensor(getOutput());

  if (!aTy || !bTy || !outTy)
    return emitOpError("requires ranked tensor types");

  if (aTy.getRank() != 2 || bTy.getRank() != 2)
    return emitOpError("A and B must be 2D");

  if (!sameElementType(aTy, bTy) || !sameElementType(aTy, outTy))
    return emitOpError("element types must match");

  int64_t K1 = getTransA() ? aTy.getDimSize(0) : aTy.getDimSize(1);
  int64_t K2 = getTransB() ? bTy.getDimSize(1) : bTy.getDimSize(0);

  if (K1 != ShapedType::kDynamic && K2 != ShapedType::kDynamic && K1 != K2)
    return emitOpError("inner dimensions mismatch");

  if (auto bias = getC()) {
    auto cTy = getRankedTensor(bias);
    if (!cTy)
      return emitOpError("bias must be ranked tensor");
  }

  return success();
}

LogicalResult ConvOp::verify() {
  auto xTy = getRankedTensor(getInput());
  auto wTy = getRankedTensor(getWeight());
  auto yTy = getRankedTensor(getOutput());

  if (!xTy || !wTy || !yTy)
    return emitOpError("requires ranked tensor types");

  if (xTy.getRank() < 3)
    return emitOpError("input rank must be >= 3");

  if (!sameElementType(xTy, wTy) || !sameElementType(xTy, yTy))
    return emitOpError("element types must match");

  int64_t spatial = xTy.getRank() - 2;

  if ((int64_t)getStrides().size() != spatial)
    return emitOpError("strides size mismatch");

  if ((int64_t)getDilations().size() != spatial)
    return emitOpError("dilations size mismatch");

  StringRef autoPad = getAutoPad();
  if (autoPad != "NOTSET" && autoPad != "SAME_UPPER" &&
      autoPad != "SAME_LOWER" && autoPad != "VALID")
    return emitOpError("unknown auto_pad value: ") << autoPad;

  if (autoPad == "NOTSET" && getPads().size() != spatial * 2)
    return emitOpError("pads size must equal 2 * spatial");

  if (getGroup() <= 0)
    return emitOpError("group must be positive");

  return success();
}

LogicalResult TransposeOp::verify() {
  auto inTy = getRankedTensor(getInput());
  auto outTy = getRankedTensor(getOutput());

  if (!inTy || !outTy)
    return emitOpError("requires ranked tensor types");

  int64_t rank = inTy.getRank();
  auto perm = getPerm();

  if (perm.empty())
    return success();

  if (perm.size() != rank)
    return emitOpError("perm size must equal rank");

  llvm::SmallVector<bool> seen(rank, false);
  for (auto p : perm) {
    if (p < 0 || p >= rank)
      return emitOpError("perm index out of range");
    if (seen[p])
      return emitOpError("duplicate perm index");
    seen[p] = true;
  }

  for (size_t i = 0; i < rank; ++i) {
    int64_t inDim = inTy.getDimSize(perm[i]);
    int64_t outDim = outTy.getDimSize(i);

    if (inDim != ShapedType::kDynamic && outDim != ShapedType::kDynamic &&
        inDim != outDim)
      return emitOpError("output shape mismatch at axis ") << i;
  }

  return success();
}