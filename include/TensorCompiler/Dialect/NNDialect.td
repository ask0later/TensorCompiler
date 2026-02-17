#ifndef NN_DIALECT
#define NN_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// NN Dialect Definition
//===----------------------------------------------------------------------===//

def NN_Dialect : Dialect {
  let name = "nn";
  let summary = "Neural Network Dialect";
  
  let description = [{
    Dialect for high-level representation of neural networks.
    Supports the following operations: relu.
  }];
  
  let cppNamespace = "::mlir::nn";

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // NN_DIALECT
