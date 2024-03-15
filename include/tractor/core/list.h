// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>

namespace tractor {

OpGroup makeOpGroup(const std::string &name);

OpType makeOpType(const std::string &name);

class ArgList {
  uint8_t *_base = nullptr;
  const uintptr_t *_offsets = nullptr;

 public:
  inline ArgList(void *base, const uintptr_t *offsets)
      : _base((uint8_t *)base), _offsets(offsets) {}
  template <class T>
  inline T *argp(size_t i) const {
    return (T *)(void *)(_base + _offsets[i]);
  }
  template <class T>
  inline T &arg(size_t i) const {
    return *argp<T>(i);
  }
};

const Operator *makeListOperator(
    const std::string &name, const std::string &label, const OpMode &mode,
    const OpType &type, const OpGroup &group,
    const std::vector<Operator::Argument> &args,
    const std::function<void(void *, const uintptr_t *)> &fun);

const Operator *makeListOperator(
    const std::string &name, const std::string &label, const OpMode &mode,
    const OpType &type, const OpGroup &group,
    const std::vector<Operator::Argument> &args,
    const std::function<void(const ArgList &)> &fun);

std::vector<Operator::Argument> makeForwardArgs(
    const ArrayRef<const Operator::Argument> &args);

std::vector<Operator::Argument> makeReverseArgs(
    const ArrayRef<const Operator::Argument> &args);

const Operator *makeListOperator(
    const std::string &name, const std::string &label,
    const std::vector<Operator::Argument> &args,
    const std::function<void(void *, const uintptr_t *)> &fun_compute,
    const std::function<void(void *, const uintptr_t *)> &fun_forward,
    const std::function<void(void *, const uintptr_t *)> &fun_reverse);

const Operator *makeListOperator(
    const std::string &name, const std::string &label,
    const std::vector<Operator::Argument> &args,
    const std::function<void(const ArgList &)> &fun_compute,
    const std::function<void(const ArgList &)> &fun_forward,
    const std::function<void(const ArgList &)> &fun_reverse);

template <class T>
T &bindArg(void *base, const uintptr_t **offsets) {
  T *ret = (T *)(void *)((uint8_t *)(void *)base + (*offsets)[0]);
  (*offsets)++;
  return *ret;
}

}  // namespace tractor
