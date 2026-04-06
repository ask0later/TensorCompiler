#include "TensorCompiler/Conversion/MLIRBuilder.hpp"
#include "TensorCompiler/Conversion/LLVMLowering.hpp"
#include "TensorCompiler/Dialect/NNDialect.hpp"
#include "TensorCompiler/Frontend/GraphBuilder.hpp"
#include "TensorCompiler/Frontend/ONNXModel.hpp"
#include "TensorCompiler/Graph/GraphDumper.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/Module.h>

#include <iostream>

using namespace llvm;

static cl::OptionCategory TCOptions("Tensor Compiler Options");

static cl::opt<std::string> InputModel(cl::Positional, cl::desc("<model.onnx>"),
                                       cl::Required, cl::cat(TCOptions));

static cl::opt<bool> DumpGraph("graph-dump", cl::desc("Dump graph"),
                               cl::init(false), cl::cat(TCOptions));

static cl::opt<bool> DumpGraphDot("graph-dot-dump",
                                  cl::desc("Dump graph to graph.dot"),
                                  cl::init(false), cl::cat(TCOptions));

static cl::opt<bool> DumpHIR("high-dialect-dump",
                             cl::desc("Dump high-level MLIR dialect"),
                             cl::init(false), cl::cat(TCOptions));

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions(TCOptions);
  cl::ParseCommandLineOptions(argc, argv, "Tensor Compiler\n");

  try {
    tc::frontend::ONNXModel model{InputModel};
    tc::frontend::GraphBuilder graphBuilder;
    model.Parse(graphBuilder);
    const auto &graph = graphBuilder.GetGraph();

    if (DumpGraph) {
      llvm::outs() << tc::graph::DumpGraph(graph);
    }

    if (DumpGraphDot) {
      if (!tc::graph::SaveDot(graph, "graph.dot")) {
        llvm::errs() << "Failed to save graph.dot\n";
        return 1;
      }
      llvm::outs() << "DOT graph saved to graph.dot\n";
    }

    mlir::MLIRContext ctx;
    ctx.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                    mlir::nn::NNDialect>();

    tc::conversion::HighLevelMLIRBuilder builder(ctx);
    const mlir::ModuleOp &module = builder.Build(graph);

    if (DumpHIR) {
      llvm::outs() << "HIR Dialect Dump:\n";
      module->print(llvm::outs());
    }

    auto llvmModule = tc::conversion::lowerToLLVM(module, ctx);
    if (!llvmModule) {
      llvm::errs() << "Failed to lower MLIR to LLVM IR\n";
      return 1;
    }

    llvm::outs() << "LLVM IR Dump:\n";
    llvmModule->print(llvm::outs(), nullptr);
    llvm::outs() << "\n";
  } catch (const std::exception &e) {
    llvm::errs() << "Compilation failed: " << e.what() << "\n";
    return 1;
  }

  return 0;
}