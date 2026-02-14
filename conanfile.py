from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps, cmake_layout

class TensorCompilerConan(ConanFile):
    name = "TensorCompiler"

    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("protobuf/3.21.12")
        self.requires("onnx/1.15.0")

    def build_requirements(self):
        self.tool_requires("cmake/[>=3.27]")

    def test_requirements(self):
        self.test_requires("gtest/1.17.0")

    def layout(self):
        cmake_layout(self)