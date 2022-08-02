// (c) 2020-2022 Philipp Ruppel

#include <tractor/core/any.h>

#include <tractor/core/ops.h>
#include <tractor/core/recorder.h>

#include <stdarg.h>

namespace tractor {

// Any operator+(const Any &a, const Any &b) {
//   return Any::call(Operator::find<compute, op_add>({a.type(), b.type()}), a,
//   b);
// }
// Any operator-(const Any &a, const Any &b) {
//   return Any::call(Operator::find<compute, op_sub>({a.type(), b.type()}), a,
//   b);
// }
// Any operator*(const Any &a, const Any &b) {
//   return Any::call(Operator::find<compute, op_mul>({a.type(), b.type()}), a,
//   b);
// }
// Any operator/(const Any &a, const Any &b) {
//   return Any::call(Operator::find<compute, op_div>({a.type(), b.type()}), a,
//   b);
// }

// Any Any::_call(const Operator *op, size_t n, ...) {
//   va_list va;
//   va_start(va, n);
//   Any ret;
//   std::vector<void *> args;
//   for (size_t i = 0; i < n; i++) {
//     args.push_back(va_arg(va, Any *)->data());
//   }
//   if (args.size() < op->argumentCount()) {
//     ret = Any(op->arg(args.size()).typeInfo());
//     args.push_back(ret.data());
//   }
//   if (args.size() != op->argumentCount()) {
//     throw std::runtime_error("function signature mismatch");
//   }
//   op->callIndirect(args.data());
//   if (auto *rec = Recorder::instance()) {
//     rec->op(op);
//     for (auto &a : args) {
//       rec->push((uintptr_t)a);
//     }
//   }
//   return ret;
// }

void Any::_check() const {
  if (empty()) {
    throw std::runtime_error("variable is null");
  }
}

Any::Any(const TypeInfo &type) {
  _type = type;
  _data.resize(type.size(), 0);
  if (auto *rec = Recorder::instance()) {
    rec->constant(type, _data.data());
  }
}

Any::Any(const TypeInfo &type, const void *data) {
  _type = type;
  _data.resize(type.size(), 0);
  std::memcpy(_data.data(), data, type.size());
  if (auto *rec = Recorder::instance()) {
    rec->constant(type, _data.data());
  }
}

Any::Any(const Any &other) {
  _type = other._type;
  _data = other._data;
  if (!other.empty()) {
    _copy(_type, other._data.data(), _data.data());
  }
}

Any &Any::operator=(const Any &other) {
  _type = other._type;
  _data = other._data;
  if (!other.empty()) {
    _copy(_type, other._data.data(), _data.data());
  }
  return *this;
}

void Any::_copy(const TypeInfo &type, const void *from, void *to) {
  const Operator *move_op = Operator::find(OpMode(typeid(compute *)),
                                           OpType(typeid(op_move *)), {type});
  std::memcpy(to, from, type.size());
  if (auto *rec = Recorder::instance()) {
    rec->op(move_op);
    rec->push((uintptr_t)from);
    rec->push((uintptr_t)to);
  }
}

} // namespace tractor
