#include "TensorCompiler/Conversion/ConvertNNDialectToLinalg.hpp"
#include "TensorCompiler/Dialect/NNDialect.hpp"

#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace mlir::nn;

namespace tc::conversion {
namespace {

struct AddOpPattern : public OpRewritePattern<AddOp> {
  using OpRewritePattern<AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AddOp op, PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto type = op.getType();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(type.getRank()));
    maps.push_back(rewriter.getMultiDimIdentityMap(type.getRank()));

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, type, ValueRange{lhs, rhs}, ValueRange{},
        maps,
        SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, ValueRange{sum.getResult()});
        });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct MulOpPattern : public OpRewritePattern<MulOp> {
  using OpRewritePattern<MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MulOp op, PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto type = op.getType();
    auto lhs = op.getLhs();
    auto rhs = op.getRhs();

    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(type.getRank()));
    maps.push_back(rewriter.getMultiDimIdentityMap(type.getRank()));

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, type, ValueRange{lhs, rhs}, ValueRange{},
        maps,
        SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto prod = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, ValueRange{prod.getResult()});
        });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct ReluOpPattern : public OpRewritePattern<ReluOp> {
  using OpRewritePattern<ReluOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluOp op, PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto type = op.getType();
    auto input = op.getInput();

    auto zeroAttr = rewriter.getZeroAttr(type.getElementType());
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    auto zeroTensor = rewriter.create<tensor::SplatOp>(
        loc, zero.getResult(), type);

    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(type.getRank()));
    maps.push_back(rewriter.getMultiDimIdentityMap(type.getRank()));

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, type, ValueRange{input, zeroTensor}, ValueRange{},
        maps,
        SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto cmp = b.create<arith::CmpFOp>(
              loc, arith::CmpFPredicate::OGT, args[0], args[1]);
          auto sel = b.create<arith::SelectOp>(
              loc, cmp.getResult(), args[0], args[1]);
          b.create<linalg::YieldOp>(loc, ValueRange{sel.getResult()});
        });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct MatMulOpPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto type = op.getType();

    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, type, ValueRange{op.getA(), op.getB()}, ValueRange{});

    rewriter.replaceOp(op, matmul.getResult(0));
    return success();
  }
};

struct GemmOpPattern : public OpRewritePattern<GemmOp> {
  using OpRewritePattern<GemmOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GemmOp op, PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto type = op.getType();
    auto a = op.getA();
    auto b = op.getB();
    Value c = op.getC();

    auto matmul = rewriter.create<linalg::MatmulOp>(
        loc, type, ValueRange{a, b}, ValueRange{});

    Value matmulResult = matmul.getResult(0);

    if (!c) {
      rewriter.replaceOp(op, matmulResult);
      return success();
    }

    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(type.getRank()));
    maps.push_back(rewriter.getMultiDimIdentityMap(type.getRank()));

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, type, ValueRange{matmulResult, c}, ValueRange{},
        maps,
        SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, ValueRange{sum.getResult()});
        });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct ConvOpPattern : public OpRewritePattern<ConvOp> {
  using OpRewritePattern<ConvOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConvOp op, PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto type = op.getType();
    auto input = op.getInput();
    auto weight = op.getWeight();
    auto bias = op.getBias();

    auto conv = rewriter.create<linalg::Conv2DNchwFchwOp>(
        loc, type, ValueRange{input, weight}, ValueRange{});

    Value convResult = conv.getResult(0);

    if (!bias) {
      rewriter.replaceOp(op, convResult);
      return success();
    }

    SmallVector<AffineMap> maps(2, rewriter.getMultiDimIdentityMap(type.getRank()));
    maps.push_back(rewriter.getMultiDimIdentityMap(type.getRank()));

    auto generic = rewriter.create<linalg::GenericOp>(
        loc, type, ValueRange{convResult, bias}, ValueRange{},
        maps,
        SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          auto sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, ValueRange{sum.getResult()});
        });

    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct TransposeOpPattern : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern<TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op, PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto type = op.getType();
    auto perm = op.getPerm();

    auto init = rewriter.create<tensor::EmptyOp>(
        loc, type.getShape(), type.getElementType());

    auto transpose = rewriter.create<linalg::TransposeOp>(
        loc, input, init, perm);

    rewriter.replaceOp(op, transpose.getResult().front());
    return success();
  }
};

class ConvertNNDialectToLinalgPass
    : public PassWrapper<ConvertNNDialectToLinalgPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertNNDialectToLinalgPass)

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<AddOpPattern, MulOpPattern, ReluOpPattern, MatMulOpPattern,
                 GemmOpPattern, ConvOpPattern, TransposeOpPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertNNDialectToLinalgPass() {
  return std::make_unique<ConvertNNDialectToLinalgPass>();
}

} // namespace tc::conversion