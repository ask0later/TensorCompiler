# Tensor Compiler

**Tensor Compiler** is a tool that converts models in the **ONNX** format into an intermediate representation (**IR**) using an **MLIR dialect**.

Currently, the project supports core neural network operations including:
- Conv
- Relu
- Add
- Mul
- MatMul
- Gemm
- Transpose

---

## Requirements

* LLVM + MLIR **version 20 or newer**  
  - either built from source, or  
  - installed system-wide 

---

## Build Instructions

### 1. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Conan

```bash
pip install conan
```

### 3. Install dependencies

```bash
conan install . --build=missing -s build_type=Release
```

### 4. Configure CMake

```bash
cmake --preset conan-release \
  -DMLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  -DLLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm
```

❗ **Important:**
You must provide paths to your built LLVM and MLIR installations:

* `MLIR_DIR` — path to the `mlir` directory inside the LLVM build
* `LLVM_DIR` — path to the `llvm` directory inside the LLVM build

---

### 5. Build the project

```bash
cmake --build --preset conan-release
```

---

### 6. Run tests

```bash
ctest --test-dir build/Release
```

---

## Usage

After building, you can pass an ONNX model to the compiler to generate MLIR IR:

```bash
./build/Release/tensor-compiler <model.onnx>
```
By default, the computation graph is printed to stdout.

---

## Options
### --graph-dot-dump
Export the computation graph to graph.dot.

### --high-dialect-dump
Print the generated high-level MLIR dialect.

---

## Maintainer
Developed and maintained by [ask0later](https://t.me/ask0later#). Feel free to open issues and contribute.

---