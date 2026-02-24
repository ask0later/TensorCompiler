#ifndef NN_OPS
#define NN_OPS

include "TensorCompiler/Dialect/NNDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

class NN_Op<string mnemonic, list<Trait> traits = []> :
    Op<NN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ReluOp
//
// Element-wise ReLU activation function.
//
// %y = nn.relu %x : tensor<4x64xf32>
//===----------------------------------------------------------------------===//

def NN_ReluOp : NN_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation function";
  let description = [{
    Applies ReLU (Rectified Linear Unit) activation: out = max(input, 0).
    Result type is identical to input type.

    Example:
    ```mlir
    %y = nn.relu %x : tensor<4xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$input);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($output)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AddOp
//
// Element-wise addition with NumPy-style broadcasting.
//
// %z = nn.add %x, %y : tensor<4x3xf32>, tensor<1x3xf32> -> tensor<4x3xf32>
//===----------------------------------------------------------------------===//

def NN_AddOp : NN_Op<"add", [Pure]> {
  let summary = "Element-wise addition with broadcasting";
  let description = [{
    Adds two tensors element-wise following NumPy broadcasting rules.
    The result shape is the broadcast-compatible shape of lhs and rhs.

    Example:
    ```mlir
    %z = nn.add %x, %y : tensor<4x3xf32>, tensor<1x3xf32> -> tensor<4x3xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MulOp
//
// Element-wise multiplication with NumPy-style broadcasting.
//
// %z = nn.mul %x, %y : tensor<4x3xf32>, tensor<1x3xf32> -> tensor<4x3xf32>
//===----------------------------------------------------------------------===//

def NN_MulOp : NN_Op<"mul", [Pure]> {
  let summary = "Element-wise multiplication with broadcasting";
  let description = [{
    Multiplies two tensors element-wise following NumPy broadcasting rules.

    Example:
    ```mlir
    %z = nn.mul %x, %y : tensor<4x3xf32>, tensor<4x3xf32> -> tensor<4x3xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$lhs, AnyRankedTensor:$rhs);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($output)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MatMulOp
//
// Batched matrix multiplication (ONNX MatMul).
// Supports broadcasting over batch dimensions.
//
// %c = nn.matmul %a, %b : tensor<4x8xf32>, tensor<8x16xf32>
//                      -> tensor<4x16xf32>
//===----------------------------------------------------------------------===//

def NN_MatMulOp : NN_Op<"matmul", [Pure]> {
  let summary = "Batched matrix multiplication";
  let description = [{
    Computes A @ B. Supports N-dimensional tensors: the last two dimensions
    are the matrix dimensions, the rest are batch dimensions (broadcast-compatible).

    Example:
    ```mlir
    %c = nn.matmul %a, %b : tensor<4x8xf32>, tensor<8x16xf32>
                         -> tensor<4x16xf32>
    ```
  }];

  let arguments = (ins AnyRankedTensor:$A, AnyRankedTensor:$B);
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $A `,` $B attr-dict
    `:` type($A) `,` type($B) `->` type($output)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GemmOp
//
// General matrix multiplication (ONNX Gemm):
//   Y = alpha * A' * B' + beta * C
//
// A' = transA==1 ? transpose(A) : A
// B' = transB==1 ? transpose(B) : B
// C  is optional bias.
//
// Attributes:
//   transA : i64  — transpose A before multiply (default 0)
//   transB : i64  — transpose B before multiply (default 0)
//   alpha : f32  — scale for A*B result (default 1.0)
//   beta : f32  — scale for bias C (default 1.0)
//
// %y = nn.gemm %a, %b [%c] {transA=0, transB=1, alpha=1.0, beta=1.0}
//          : tensor<4x8xf32>, tensor<16x8xf32>, tensor<16xf32>
//         -> tensor<4x16xf32>
//===----------------------------------------------------------------------===//

def NN_GemmOp : NN_Op<"gemm", [Pure]> {
  let summary = "General matrix multiplication (ONNX Gemm)";
  let description = [{
    Computes `Y = alpha * A' * B' + beta * C where A' and B' are
    optionally transposed. C (bias) is optional.

    Example:
    ```mlir
    // With bias
    %y = nn.gemm %a, %b [%c] {transA=0, transB=1, alpha=1.0, beta=1.0}
             : tensor<4x8xf32>, tensor<16x8xf32>, tensor<16xf32>
            -> tensor<4x16xf32>
    // Without bias
    %y = nn.gemm %a, %b {transA=0, transB=0, alpha=1.0, beta=1.0}
             : tensor<4x8xf32>, tensor<8x16xf32>
            -> tensor<4x16xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$A,
    AnyRankedTensor:$B,
    Optional<AnyRankedTensor>:$C,
    DefaultValuedAttr<I64Attr, "0">:$transA,
    DefaultValuedAttr<I64Attr, "0">:$transB,
    DefaultValuedAttr<F32Attr, "1.0f">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0f">:$beta
  );
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $A `,` $B ( `[` $C^ `:` type($C) `]` )? attr-dict
    `:` type($A) `,` type($B) `->` type($output)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ConvOp
//
// N-dimensional convolution (ONNX Conv):
//   rank(input)=3 → Conv1D
//   rank(input)=4 → Conv2D
//   rank(input)=5 → Conv3D
//
// Attributes:
//   strides : i64[]  — stride per spatial dim (default: all 1)
//   dilations : i64[]  — dilation per spatial dim (default: all 1)
//   pads : i64[]  — [d0_begin,..,dN_begin,
//                    d0_end,..,dN_end] (default: all 0)
//               ignored when auto_pad ≠ "NOTSET"
//   group : i64  — number of groups (depthwise)  (default: 1)
//   auto_pad : str  — "NOTSET"|"SAME_UPPER"|"SAME_LOWER"|"VALID"
//
// %y = nn.conv %x, %w [%b]
//          {strides=[2,2], dilations=[1,1], pads=[1,1,1,1],
//           group=1, auto_pad="NOTSET"}
//          : tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>
//         -> tensor<1x64x112x112xf32>
//===----------------------------------------------------------------------===//

def NN_ConvOp : NN_Op<"conv", [Pure]> {
  let summary = "N-dimensional convolution (ONNX Conv semantics)";
  let description = [{
    Applies convolution of `input` with `weight` kernel.
    `bias` is optional. Spatial dimensionality is inferred from input rank.

    `pads` layout (ONNX format): all begin values first, then all end values.
    For 2D: [pad_top, pad_left, pad_bottom, pad_right].
    `pads` is ignored when `auto_pad` is not "NOTSET".

    Example:
    ```mlir
    %y = nn.conv %x, %w [%b]
             {strides=[1,1], dilations=[1,1], pads=[0,0,0,0],
              group=1, auto_pad="NOTSET"}
             : tensor<1x3x224x224xf32>, tensor<64x3x3x3xf32>, tensor<64xf32>
            -> tensor<1x64x222x222xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    AnyRankedTensor:$weight,
    Optional<AnyRankedTensor>:$bias,

    DenseI64ArrayAttr:$strides,
    DenseI64ArrayAttr:$dilations,
    DenseI64ArrayAttr:$pads,
    DefaultValuedAttr<I64Attr, "1">:$group,
    DefaultValuedAttr<StrAttr, "\"NOTSET\"">:$auto_pad
  );
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight ( `[` $bias^ `:` type($bias) `]` )? attr-dict
    `:` type($input) `,` type($weight) `->` type($output)
  }];
  let hasVerifier = 1;
}

// TransposeOp
//
// Permutes the axes of a tensor (ONNX Transpose semantics).
//
// rank(input) = N
// rank(output) = N
//
// Attribute:
//   perm : i64[]  — permutation of axes.
//                   Must be a permutation of [0..N-1].
//                   Empty array = reverse all axes (ONNX default).
//
//   %y = nn.transpose %x {perm=[0,2,3,1]}
//          : tensor<1x3x8x8xf32>
//         -> tensor<1x8x8x3xf32>
//
//   %y = nn.transpose %x {perm=[1,0]}
//          : tensor<4x8xf32> -> tensor<8x4xf32>
//
//   Default ONNX behaviour
//   %y = nn.transpose %x
//          : tensor<2x3x4xf32> -> tensor<4x3x2xf32>
//
//===----------------------------------------------------------------------===//

def NN_TransposeOp : NN_Op<"transpose", [Pure]> {
  let summary = "Tensor axis permutation (ONNX Transpose semantics)";
  let description = [{
    Permutes the axes of the input tensor according to `perm`.
    Empty `perm` means reverse all axes (same as ONNX default).

    Example:
    ```mlir
    %y = nn.transpose %x {perm=[0,2,3,1]}
             : tensor<1x3x8x8xf32> -> tensor<1x8x8x3xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    DenseI64ArrayAttr:$perm
  );
  let results = (outs AnyRankedTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];
  let hasVerifier = 1;
}

#endif // NN_OPS