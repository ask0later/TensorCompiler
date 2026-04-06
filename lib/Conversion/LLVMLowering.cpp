#include "TensorCompiler/Conversion/LLVMLowering.hpp"
#include "TensorCompiler/Conversion/ConvertNNDialectToLinalg.hpp"

#include <mlir/Pass/PassManager.h>
#include <mlir/Conversion/Passes.h>
#include <mlir/Transforms/Passes.h>

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>

#include <mlir/Dialect/Bufferization/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>

#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <llvm/IR/Module.h>
#include <llvm/IR/LLVMContext.h>

namespace tc::conversion {

std::unique_ptr<llvm::Module> lowerToLLVM(mlir::ModuleOp module,
                                          mlir::MLIRContext &ctx) {
  mlir::PassManager pm(&ctx);

  pm.addPass(createConvertNNDialectToLinalgPass());

  pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createSCFToControlFlowPass());

  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (failed(pm.run(module)))
    return nullptr;

  mlir::registerLLVMDialectTranslation(ctx);

  llvm::LLVMContext llvmContext;
  return mlir::translateModuleToLLVMIR(module, llvmContext);
}

} // namespace tc::conversion