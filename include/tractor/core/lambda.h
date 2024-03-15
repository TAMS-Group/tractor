// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/list.h>
#include <tractor/core/log.h>
#include <tractor/core/operator.h>
#include <unordered_map>
namespace tractor {

template <class Functor>
struct PointerOp : Operator {
  Functor _functor;
  template <class F, class... Args, size_t... Indices>
  void __init(void (F ::*f)(Args...) const,
              const std::integer_sequence<size_t, Indices...> &indices) {
    _functions.indirect = [](void *base, const uintptr_t *offsets) {
      const PointerOp *_this = (const PointerOp *)offsets[0];
      _this->_functor(
          (Args)(void *)((uint8_t *)base + offsets[Indices + 1])...);
    };
  }
  template <class F, class... Args>
  void __init(void (F ::*f)(Args...) const) {
    __init(f, std::make_index_sequence<sizeof...(Args)>());
  }
  PointerOp(const std::string &name, const std::string &label,
            const OpMode &mode, const OpType &op, const OpGroup &group,
            const std::vector<Operator::Argument> &args, const Functor &functor)
      : Operator(name, label, mode, op, group), _functor(functor) {
    TRACTOR_DEBUG("make op " << name);
    _arguments = args;
    __init(&Functor::operator());
    _functions.context.push_back((uintptr_t)this);
  }
};

template <class Functor>
const Operator *makePointerOp(const std::string &name, const std::string &label,
                              const OpMode &mode, const OpType &op,
                              const OpGroup &group,
                              const std::vector<Operator::Argument> &args,
                              const Functor &functor) {
  static std::unordered_map<std::string, const Operator *> map;
  if (!map[name]) {
    map[name] =
        new PointerOp<Functor>(name, label, mode, op, group, args, functor);
  }
  return map[name];
}

template <class Compute, class Forward, class Reverse>
const Operator *makePointerOp(const std::string &name, const std::string &label,
                              const OpType &type,
                              const std::vector<Operator::Argument> &args,
                              const Compute &fun_compute,
                              const Forward &fun_forward,
                              const Reverse &fun_reverse) {
  auto group = makeOpGroup(name);

  auto *op = makePointerOp(name, label, OpMode(typeid(compute *)), type, group,
                           args, fun_compute);

  {
    std::vector<Operator::Argument> aa;
    for (auto &a : args) aa.push_back(a.makeInput());
    for (auto &a : args) aa.push_back(a);
    makePointerOp("forward_" + name, label, OpMode(typeid(forward *)), type,
                  group, aa, fun_forward);
  }

  {
    std::vector<Operator::Argument> aa;
    for (auto &a : args) aa.push_back(a.makeInput());
    for (auto &a : args) aa.push_back(a.makeReverse());
    makePointerOp("reverse_" + name, label, OpMode(typeid(reverse *)), type,
                  group, aa, fun_reverse);
  }

  return op;
}

template <class Compute, class Forward, class Reverse>
const Operator *makePointerOp(const std::string &op_name,
                              const std::string &type_name,
                              const std::vector<Operator::Argument> &args,
                              const Compute &fun_compute,
                              const Forward &fun_forward,
                              const Reverse &fun_reverse) {
  auto *op =
      makePointerOp(op_name + "_" + type_name, op_name,
                    OpMode(typeid(compute *)), makeOpType(op_name),
                    makeOpGroup(op_name + "_" + type_name), args, fun_compute);

  {
    std::vector<Operator::Argument> aa;
    for (auto &a : args) aa.push_back(a.makeInput());
    for (auto &a : args) aa.push_back(a);
    makePointerOp("forward_" + op_name + "_" + type_name, op_name,
                  OpMode(typeid(forward *)), makeOpType(op_name),
                  makeOpGroup(op_name + "_" + type_name), aa, fun_forward);
  }

  {
    std::vector<Operator::Argument> aa;
    for (auto &a : args) aa.push_back(a.makeInput());
    for (auto &a : args) aa.push_back(a.makeReverse());
    makePointerOp("reverse_" + op_name + "_" + type_name, op_name,
                  OpMode(typeid(reverse *)), makeOpType(op_name),
                  makeOpGroup(op_name + "_" + type_name), aa, fun_reverse);
  }

  return op;
}

}  // namespace tractor
