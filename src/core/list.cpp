// 2020-2024 Philipp Ruppel

#include <tractor/core/factory.h>
#include <tractor/core/list.h>

namespace tractor {

OpGroup makeOpGroup(const std::string &name) {
  static std::unordered_map<std::string, std::shared_ptr<int>> map;
  if (map.find(name) == map.end()) {
    map[name] = std::make_shared<int>(1);
  }
  return OpGroup(map[name].get());
}

OpType makeOpType(const std::string &name) {
  static std::unordered_map<std::string, std::shared_ptr<int>> map;
  if (map.find(name) == map.end()) {
    map[name] = std::make_shared<int>(1);
  }
  return OpType(map[name].get());
}

class ListOperator : public Operator {
  std::function<void(void *, const uintptr_t *)> _fun;

 public:
  ListOperator(const std::string &name, const std::string &label,
               const OpMode &mode, const OpType &type, const OpGroup &group,
               const std::vector<Argument> &args,
               const std::function<void(void *, const uintptr_t *)> &fun)
      : Operator(name, label, mode, type, group), _fun(fun) {
    _arguments = args;
    _functions.indirect = [](void *base, const uintptr_t *offsets) {
      const ListOperator *_this = (const ListOperator *)offsets[0];
      _this->_fun(base, offsets + 1);
    };
    _functions.context.push_back((uintptr_t)this);
  }
};

const Operator *makeListOperator(
    const std::string &name, const std::string &label, const OpMode &mode,
    const OpType &type, const OpGroup &group,
    const std::vector<Operator::Argument> &args,
    const std::function<void(void *, const uintptr_t *)> &fun) {
  static std::unordered_map<std::string, std::shared_ptr<Operator>> map;
  if (map.find(name) == map.end()) {
    map[name] = std::make_shared<ListOperator>(name, label, mode, type, group,
                                               args, fun);
  }
  return map[name].get();
}

const Operator *makeListOperator(
    const std::string &name, const std::string &label, const OpMode &mode,
    const OpType &type, const OpGroup &group,
    const std::vector<Operator::Argument> &args,
    const std::function<void(const ArgList &)> &fun) {
  return makeListOperator(name, label, mode, type, group, args,
                          [fun](void *base, const uintptr_t *offsets) {
                            fun(ArgList(base, offsets));
                          });
}

std::vector<Operator::Argument> makeForwardArgs(
    const ArrayRef<const Operator::Argument> &args) {
  std::vector<Operator::Argument> ret;
  for (auto &a : args) {
    ret.push_back(a.makeInput());
  }
  for (auto &a : args) {
    ret.push_back(a);
  }
  return ret;
}

std::vector<Operator::Argument> makeReverseArgs(
    const ArrayRef<const Operator::Argument> &args) {
  std::vector<Operator::Argument> ret;
  for (auto &a : args) {
    ret.push_back(a.makeInput());
  }
  for (auto &a : args) {
    ret.push_back(a.makeReverse());
  }
  return ret;
}

const Operator *makeListOperator(
    const std::string &name, const std::string &label,
    const std::vector<Operator::Argument> &args,
    const std::function<void(void *, const uintptr_t *)> &fun_compute,
    const std::function<void(void *, const uintptr_t *)> &fun_forward,
    const std::function<void(void *, const uintptr_t *)> &fun_reverse) {
  auto group = makeOpGroup(name);
  auto *ret = makeListOperator(name, label, OpMode(typeid(compute *)),
                               makeOpType(name), group, args, fun_compute);
  makeListOperator("forward_" + name, label, OpMode(typeid(forward *)),
                   makeOpType("forward_" + name), group, makeForwardArgs(args),
                   fun_forward);
  makeListOperator("reverse_" + name, label, OpMode(typeid(reverse *)),
                   makeOpType("reverse_" + name), group, makeReverseArgs(args),
                   fun_reverse);
  return ret;
}

const Operator *makeListOperator(
    const std::string &name, const std::string &label,
    const std::vector<Operator::Argument> &args,
    const std::function<void(const ArgList &)> &fun_compute,
    const std::function<void(const ArgList &)> &fun_forward,
    const std::function<void(const ArgList &)> &fun_reverse) {
  return makeListOperator(
      name, label, args,
      [fun_compute](void *base, const uintptr_t *offsets) {
        fun_compute(ArgList(base, offsets));
      },
      [fun_forward](void *base, const uintptr_t *offsets) {
        fun_forward(ArgList(base, offsets));
      },
      [fun_reverse](void *base, const uintptr_t *offsets) {
        fun_reverse(ArgList(base, offsets));
      });
}

}  // namespace tractor
