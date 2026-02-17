#ifndef NN_OPS
#define NN_OPS

include "TensorCompiler/Dialect/NNDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Base NN operation definition
//===----------------------------------------------------------------------===//

class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ReLU Operation
//===----------------------------------------------------------------------===//

def NN_ReluOp : NN_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  
  let description = [{
    Applies ReLU (Rectified Linear Unit) activation: max(x, 0)
    
    Example:
    ```mlir
    %result = nn.relu %input : tensor<4xf32>
    ```
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

#endif // NN_OPS