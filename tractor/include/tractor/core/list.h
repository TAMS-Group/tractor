// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>

namespace tractor {

OpGroup makeOpGroup(const std::string &name);

OpType makeOpType(const std::string &name);

const Operator *
makeListOperator(const std::string &name, const std::string &label,
                 const OpMode &mode, const OpType& type, const OpGroup &group,
                 const std::vector<Operator::Argument> &args,
                 const std::function<void(void *, const uintptr_t *)> &fun);

std::vector<Operator::Argument>
makeForwardArgs(const ArrayRef<const Operator::Argument> &args);

std::vector<Operator::Argument>
makeReverseArgs(const ArrayRef<const Operator::Argument> &args);

// const Operator *makeListOperator(
//     const std::string &name, const std::string &label,
//     const std::vector<Operator::Argument> &args,
//     const std::function<void(void *, const uintptr_t *)> &fun_compute,
//     const std::function<void(void *, const uintptr_t *)> &fun_forward,
//     const std::function<void(void *, const uintptr_t *)> &fun_reverse);

class ArgList {
  void *_base = nullptr;
  const uintptr_t *_offsets = nullptr;

public:
  ArgList(void *base, const uintptr_t *offsets)
      : _base(base), _offsets(offsets) {}
  template <class T> T &arg(size_t i) const {
    return *(T *)(void *)((uint8_t *)_base + _offsets[i]);
  }
};

template <class T> T &bindArg(void *base, const uintptr_t **offsets) {
  T *ret = (T *)(void *)((uint8_t *)(void *)base + (*offsets)[0]);
  (*offsets)++;
  return *ret;
}

} // namespace tractor
