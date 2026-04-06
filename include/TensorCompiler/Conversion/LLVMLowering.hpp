#pragma once

#include <memory>
#include <mlir/IR/BuiltinOps.h>

namespace llvm {
class Module;
}

namespace tc::conversion {

std::unique_ptr<llvm::Module> lowerToLLVM(mlir::ModuleOp module,
                                          mlir::MLIRContext &ctx);

} // namespace tc::conversion