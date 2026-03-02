#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tc::graph {
using IntList = std::vector<int64_t>;
using DoubleList = std::vector<double>;
using StringList = std::vector<std::string>;

struct TensorData {
  std::string name;
  std::vector<int64_t> dims;
  std::vector<uint8_t> raw_data;
};

using AttrScalar = std::variant<int64_t, double, std::string, bool>;
using AttrList = std::variant<IntList, DoubleList, StringList>;
using AttrValue =
    std::variant<std::monostate, AttrScalar, AttrList, TensorData>;

enum class EntityKind { Tensor, Operation, Constant };
using EntityId = int;

struct TensorEntity {
  EntityId id;
  std::string name;
  std::vector<int64_t> shape;
  bool is_initializer = false;
  std::optional<TensorData> data;
};

struct OperationEntity {
  EntityId id;
  std::string name;
  std::string op_type;
  std::vector<EntityId> inputs;
  std::vector<EntityId> outputs;
  std::unordered_map<std::string, AttrValue> attrs;
};

struct ConstantEntity {
  EntityId id;
  TensorData data;
};

struct Entity {
  EntityKind kind;
  std::variant<TensorEntity, OperationEntity, ConstantEntity> entity;

  Entity(EntityKind k, OperationEntity op)
      : kind(k), entity(std::move(op)) {}

  Entity(EntityKind k, TensorEntity t)
      : kind(k), entity(std::move(t)) {}

  Entity(EntityKind k, ConstantEntity c)
      : kind(k), entity(std::move(c)) {}
};

class Graph {
public:
  EntityId AddTensor(const std::string &name, std::vector<int64_t> shape = {},
                     bool is_init = false);

  EntityId AddConstant(const TensorData &data);

  EntityId
  AddOperation(const std::string &op, const std::string &name,
               const std::vector<EntityId> &ins,
               const std::vector<EntityId> &outs,
               const std::unordered_map<std::string, AttrValue> &attrs);

  TensorEntity *GetTensor(EntityId id);
  const TensorEntity *GetTensor(EntityId id) const;

  const std::vector<Entity> &Entities() const { return entities_; }
  const std::vector<EntityId> &Inputs() const { return inputs_; }
  const std::vector<EntityId> &Outputs() const { return outputs_; }
  void AddInput(EntityId id) { inputs_.push_back(id); }
  void AddOutput(EntityId id) { outputs_.push_back(id); }

private:
  std::vector<EntityId> inputs_;
  std::vector<EntityId> outputs_;

  std::vector<Entity> entities_;
  std::unordered_map<std::string, EntityId> tensorByName_;
};
} // namespace tc::graph