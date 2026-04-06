#pragma once
#include <memory>
#include <mlir/Pass/Pass.h>

namespace tc::conversion {
std::unique_ptr<mlir::Pass> createConvertNNDialectToLinalgPass();
} // namespace tc::conversion