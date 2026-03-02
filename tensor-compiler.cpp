#include "TensorCompiler/Converter/GraphBuilder.hpp"
#include "TensorCompiler/Converter/MLIRBuilder.hpp"
#include "TensorCompiler/Dialect/NNDialect.hpp"
#include "TensorCompiler/Frontend/ONNXModel.hpp"
#include "TensorCompiler/Graph/Exporter.hpp"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/MLIRContext.h>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <iostream>

using namespace llvm;

static cl::OptionCategory TCOptions("Tensor Compiler Options");

static cl::opt<std::string> InputModel(cl::Positional, cl::desc("<model.onnx>"),
                                       cl::Required, cl::cat(TCOptions));

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
    tc::converter::onnx_to_graph::GraphBuilder graphBuilder;
    model.Parse(graphBuilder);
    const auto &graph = graphBuilder.GetGraph();

    std::cout << tc::graph::DumpGraph(graph);

    if (DumpGraphDot) {
      if (!tc::graph::SaveDot(graph, "graph.dot")) {
        llvm::errs() << "Failed to save graph.dot\n";
        return 1;
      }
      llvm::outs() << "DOT graph saved to graph.dot\n";
    }

    if (DumpHIR) {
      mlir::MLIRContext ctx;
      ctx.loadDialect<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                      mlir::nn::NNDialect>();

      tc::converter::onnx_to_high_mlir::MLIRBuilder builder(ctx);
      model.Parse(builder);

      const mlir::ModuleOp &module = builder.GetModule();

      llvm::outs() << "\n=== HIR Dialect Dump ===\n";
      module->print(llvm::outs());
      llvm::outs() << "\n";
    }

  } catch (const std::exception &e) {
    llvm::errs() << "Compilation failed: " << e.what() << "\n";
    return 1;
  }

  return 0;
}