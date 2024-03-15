// 2020-2024 Philipp Ruppel

#include <tractor/core/any.h>

#include <tractor/core/ops.h>
#include <tractor/core/recorder.h>

#include <stdarg.h>

namespace tractor {

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
    rec->arg(type, from);
    rec->arg(type, to);
  }
}

}  // namespace tractor
