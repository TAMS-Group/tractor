// (c) 2020-2022 Philipp Ruppel

#include <tractor/core/list.h>

namespace tractor {

OpGroup makeOpGroup(const std::string &name) {
  static std::unordered_map<std::string, std::shared_ptr<int>> map;
  if (map.find(name) == map.end()) {
    map[name] = std::make_shared<int>(1);
  }
  return OpGroup(map[name].get());
}

const Operator *
makeListOperator(const std::string &name, const std::string &label,
                 const OpMode &mode, const OpGroup &group,
                 const std::vector<Operator::Argument> &args,
                 void (*callback)(void *base, const uintptr_t *offsets)) {
  static std::unordered_map<std::string, std::shared_ptr<Operator>> map;
  if (map.find(name) == map.end()) {
    map[name] = std::make_shared<ListOperator>(name, label, mode, group, args,
                                               callback);
  }
  return map[name].get();
}

std::vector<Operator::Argument>
makeForwardArgs(const ArrayRef<const Operator::Argument> &args) {
  std::vector<Operator::Argument> ret;
  for (auto &a : args) {
    ret.push_back(a.makeInput());
  }
  for (auto &a : args) {
    ret.push_back(a);
  }
  return ret;
}

std::vector<Operator::Argument>
makeReverseArgs(const ArrayRef<const Operator::Argument> &args) {
  std::vector<Operator::Argument> ret;
  for (auto &a : args) {
    ret.push_back(a.makeInput());
  }
  for (auto &a : args) {
    ret.push_back(a.makeReverse());
  }
  return ret;
}

const Operator *
makeListOperator(const std::string &name, const std::string &label,
                 const std::vector<Operator::Argument> &args,
                 void (*fun_compute)(void *base, const uintptr_t *offsets),
                 void (*fun_forward)(void *base, const uintptr_t *offsets),
                 void (*fun_reverse)(void *base, const uintptr_t *offsets)) {
  auto group = makeOpGroup(name);
  auto *ret = makeListOperator(name, label, OpMode(typeid(compute *)), group,
                               args, fun_compute);
  makeListOperator("forward_" + name, label, OpMode(typeid(forward *)), group,
                   makeForwardArgs(args), fun_forward);
  makeListOperator("reverse_" + name, label, OpMode(typeid(reverse *)), group,
                   makeReverseArgs(args), fun_reverse);
  return ret;
}

} // namespace tractor
