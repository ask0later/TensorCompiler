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

Attribute NNDialect::parseAttribute(DialectAsmParser &parser,
                                    Type type) const {
  llvm_unreachable("unknown type");
  return {};
}

void NNDialect::printAttribute(Attribute attr,
                               DialectAsmPrinter &printer) const {
  llvm_unreachable("unknown attribute");
}

Type NNDialect::parseType(DialectAsmParser &parser) const {
  llvm_unreachable("unknown type");
  return {};
}

void NNDialect::printType(Type type,
                          DialectAsmPrinter &printer) const {
  llvm_unreachable("unknown type");
}

#define GET_OP_CLASSES
#include "NNOps.cpp.inc"