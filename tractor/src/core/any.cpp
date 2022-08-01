// (c) 2020-2022 Philipp Ruppel

#include <tractor/core/any.h>

#include <tractor/core/ops.h>
#include <tractor/core/recorder.h>

namespace tractor {

const TypeInfo &Any::type() const { return _type; }

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
  _copy(_type, other._data.data(), _data.data());
}

Any &Any::operator=(const Any &other) {
  _type = other._type;
  _data = other._data;
  _copy(_type, other._data.data(), _data.data());
  return *this;
}

void Any::_copy(const TypeInfo &type, const void *from, void *to) {
  // std::cout << "begin move" << std::endl;
  const Operator *move_op = Operator::find(OpMode(typeid(compute *)),
                                           OpType(typeid(op_move *)), {type});
  std::memcpy(to, from, type.size());
  if (auto *rec = Recorder::instance()) {
    rec->op(move_op);
    rec->push((uintptr_t)from);
    rec->push((uintptr_t)to);
  }
  // std::cout << "end move" << std::endl;
}

} // namespace tractor
