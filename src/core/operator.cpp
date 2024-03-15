// 2020-2024 Philipp Ruppel

#include <tractor/core/operator.h>

#include <tractor/core/log.h>

#include <algorithm>
#include <stdarg.h>
#include <unordered_map>
#include <unordered_set>

namespace tractor {

struct OperatorRegistry {
  std::unordered_map<std::string, const Operator *> name_map;
  std::unordered_map<OpGroup, OperatorModeMap *> group_map;
  std::unordered_map<
      OpMode, std::unordered_map<OpType, std::unordered_set<const Operator *>>>
      op_map;
  static OperatorRegistry *instance() {
    static OperatorRegistry *instance = new OperatorRegistry();
    return instance;
  }
};

const Operator *Operator::find(const std::string &name) {
  auto *op = tryFind(name);
  if (!op) {
    throw std::runtime_error(std::string() + "operator not found: " + name);
  }
  return op;
}

void Operator::callIndirect(void *base, uintptr_t *offsets) const {
  if (_functions.context.empty()) {
    _functions.indirect(base, offsets);
  } else {
    std::vector<uintptr_t> args = _functions.context;
    for (size_t i = 0; i < _arguments.size(); i++) {
      args.push_back(offsets[i]);
    }
    _functions.indirect(base, args.data());
  }
}

void Operator::callIndirect(void **args) const {
  callIndirect(nullptr, reinterpret_cast<uintptr_t *>(args));
}

void Operator::invoke(const void *first, ...) const {
  va_list va;
  va_start(va, first);
  std::vector<uintptr_t> args = _functions.context;
  args.push_back((uintptr_t)first);
  for (size_t i = 0; i < _arguments.size(); i++) {
    args.push_back(va_arg(va, uintptr_t));
  }
  _functions.indirect(nullptr, args.data());
}

Operator::Operator(const std::string &name, const std::string &label,
                   const OpMode &mode, const OpType &op, const OpGroup &group)
    : _name(name), _label(label), _op(op), _mode(mode) {
  TRACTOR_DEBUG("new operator type " << name);
  auto *registry = OperatorRegistry::instance();
  registry->name_map[name] = this;
  auto &map = registry->group_map[group];
  if (!map) {
    map = new OperatorModeMap();
  }
  _map = map;
  map->put(OperatorModeMap::index(mode), this);
  registry->op_map[mode][op].insert(this);
}

Operator::~Operator() {}

const Operator *Operator::tryFind(
    const OpMode &mode, const OpType &op,
    const std::initializer_list<TypeInfo> &types) {
  auto *registry = OperatorRegistry::instance();
  auto it_mode = registry->op_map.find(mode);
  if (it_mode == registry->op_map.end()) {
    return nullptr;
  }
  auto it_op = it_mode->second.find(op);
  if (it_op == it_mode->second.end()) {
    return nullptr;
  }
  for (auto *op : it_op->second) {
    auto it_type = types.begin();
    auto it_arg = op->arguments().begin();
    while (true) {
      /*if (it_arg != op->arguments().end() && it_arg->isOutput()) {
        ++it_arg;
        continue;
      }*/
      if (it_type == types.end() &&
          (it_arg == op->arguments().end() || it_arg->isOutput())) {
        return op;
      }
      if (it_type == types.end()) {
        break;
      }
      if (it_arg == op->arguments().end()) {
        break;
      }
      if (it_arg->typeInfo() == *it_type) {
        ++it_arg;
        ++it_type;
        continue;
      } else {
        break;
      }
    }
  }
  return nullptr;
}

const Operator *Operator::find(const OpMode &opmode, const OpType &optype,
                               const std::initializer_list<TypeInfo> &args) {
  auto *op = tryFind(opmode, optype, args);
  if (!op) {
    std::stringstream msg;
    msg << "operator not found mode:" << opmode.name()
        << " type:" << optype.name();
    for (auto &arg : args) {
      msg << " arg:" << arg.name();
    }
    throw std::runtime_error(msg.str());
  }
  return op;
}

const Operator *Operator::tryFind(const OpMode &mode, const OpGroup &group) {
  auto *registry = OperatorRegistry::instance();
  auto it_group = registry->group_map.find(group);
  if (it_group != registry->group_map.end()) {
    return it_group->second->at(OperatorModeMap::index(mode));
  } else {
    return nullptr;
  }
}

const Operator *Operator::tryFind(const std::string &name) {
  auto *registry = OperatorRegistry::instance();
  auto iter = registry->name_map.find(name);
  if (iter != registry->name_map.end()) {
    return iter->second;
  } else {
    return nullptr;
  }
}

std::vector<const Operator *> Operator::all() {
  auto *registry = OperatorRegistry::instance();
  std::vector<const Operator *> ret;
  for (auto &pair : registry->name_map) {
    ret.push_back(pair.second);
  }
  return ret;
}

size_t OperatorModeMap::index(const OpMode &type) {
  static std::unordered_map<OpMode, size_t> map;
  auto &i = map[type];
  if (!i) {
    i = map.size();
  }
  return i;
}

}  // namespace tractor
