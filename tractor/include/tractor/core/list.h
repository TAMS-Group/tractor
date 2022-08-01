// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>

namespace tractor {

OpGroup makeOpGroup(const std::string &name);

OpType makeOpType(const std::string &name);

class ListOperator : public Operator {

public:
  ListOperator(const std::string &name, const std::string &label,
               const OpMode &mode, const OpGroup &group,
               const std::vector<Argument> &args,
               void (*callback)(void *base, const uintptr_t *offsets))
      : Operator(name, label, mode, this, group) {
    _arguments = args;
    //_argument_count = args.size();
    _functions.indirect = callback;
  }
};

const Operator *
makeListOperator(const std::string &name, const std::string &label,
                 const OpMode &mode, const OpGroup &group,
                 const std::vector<Operator::Argument> &args,
                 void (*callback)(void *base, const uintptr_t *offsets));

std::vector<Operator::Argument>
makeForwardArgs(const ArrayRef<const Operator::Argument> &args);

std::vector<Operator::Argument>
makeReverseArgs(const ArrayRef<const Operator::Argument> &args);

const Operator *
makeListOperator(const std::string &name, const std::string &label,
                 const std::vector<Operator::Argument> &args,
                 void (*fun_compute)(void *base, const uintptr_t *offsets),
                 void (*fun_forward)(void *base, const uintptr_t *offsets),
                 void (*fun_reverse)(void *base, const uintptr_t *offsets));

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
