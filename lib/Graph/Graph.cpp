#include "TensorCompiler/Graph/Graph.hpp"
#include "TensorCompiler/Graph/GraphVisitor.hpp"

namespace tc::graph {

EntityId Graph::AddTensor(const std::string &name, std::vector<int64_t> shape,
                          int32_t dtype, bool is_init) {
  if (auto it = tensorByName_.find(name); it != tensorByName_.end())
    return it->second;

  EntityId id = entities_.size();
  TensorEntity t{id, name, shape, is_init, dtype, std::nullopt};
  entities_.emplace_back(EntityKind::Tensor, t);
  tensorByName_[name] = id;
  return id;
}

EntityId Graph::AddConstant(const TensorData &data) {
  EntityId id = entities_.size();
  ConstantEntity c{id, data};
  entities_.emplace_back(EntityKind::Constant, c);
  return id;
}

EntityId
Graph::AddOperation(const std::string &op, const std::string &name,
                    const std::vector<EntityId> &ins,
                    const std::vector<EntityId> &outs,
                    const std::unordered_map<std::string, AttrValue> &attrs) {
  EntityId id = entities_.size();
  OperationEntity o{id, name, op, ins, outs, attrs};
  entities_.emplace_back(EntityKind::Operation, o);
  return id;
}

TensorEntity *Graph::GetTensor(EntityId id) {
  if (id >= entities_.size() || entities_[id].kind != EntityKind::Tensor)
    return nullptr;
  return &std::get<TensorEntity>(entities_[id].entity);
}

const TensorEntity *Graph::GetTensor(EntityId id) const {
  if (id >= entities_.size() || entities_[id].kind != EntityKind::Tensor)
    return nullptr;
  return &std::get<TensorEntity>(entities_[id].entity);
}

void Graph::Accept(GraphVisitor &visitor) const {
  visitor.Visit(*this);
  for (const auto &entity : Entities()) {
    switch (entity.kind) {
    case EntityKind::Tensor:
      visitor.Visit(std::get<TensorEntity>(entity.entity));
      break;
    case EntityKind::Operation:
      visitor.Visit(std::get<OperationEntity>(entity.entity));
      break;
    case EntityKind::Constant:
      visitor.Visit(std::get<ConstantEntity>(entity.entity));
      break;
    }
  }
  visitor.Finalize(*this);
}
} // namespace tc::graph