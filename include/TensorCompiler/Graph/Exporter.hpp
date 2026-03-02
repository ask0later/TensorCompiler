#pragma once
#include "TensorCompiler/Graph/IR.hpp"
#include <string>

namespace tc::graph {

std::string DumpGraph(const Graph &g);

std::string ToDot(const Graph &g);
bool SaveDot(const Graph &g, const std::string &file);

} // namespace tc::graph