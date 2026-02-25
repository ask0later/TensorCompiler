#include "TensorCompiler/Driver/MLIRBuilder.hpp"
#include "TensorCompiler/Frontend/ONNXDumper.hpp"
#include "TensorCompiler/Frontend/ONNXModel.hpp"

#include "TensorCompiler/Dialect/NNDialect.hpp"
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <iostream>

int main(const int argc, const char *const argv[]) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <model.onnx>" << std::endl;
    return 1;
  }

  try {
    ONNXModel model{argv[1]};
    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::nn::NNDialect>();

    MLIRBuilder builder(ctx);
    model.Parse(builder);
    const mlir::ModuleOp &mdl = builder.GetModule();
    mdl->print(llvm::outs());
    llvm::outs() << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Compilation Failed: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}