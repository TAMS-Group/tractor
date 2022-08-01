// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/type.h>
#include <tractor/core/var.h>

#include <vector>

namespace tractor {

class Any {
  TypeInfo _type;
  std::vector<uint8_t> _data;
  static void _copy(const TypeInfo &type, const void *from, void *to);
  void _check() const;
  static Any _call(const Operator *op, size_t n, ...);
  static TypeInfo _getType(const Any &any) { return any.type(); }
  template <class Op, class... Args>
  static const Operator *_findOp(Args &&...args) {
    // return Operator::find<compute, Op>({TypeInfo::get<Args>()...});
    std::cout << "find op" << std::endl;
    auto *op = Operator::find<compute, Op>({_getType(args)...});
    std::cout << "found " << op->name() << std::endl;
    return op;
  }

public:
  Any() {}
  explicit Any(const TypeInfo &type);
  explicit Any(const TypeInfo &type, const void *data);
  Any(const Any &other);
  Any &operator=(const Any &other);
  template <class T> explicit Any(const Var<T> &v) {
    _type = TypeInfo::get<T>();
    _data.resize(sizeof(T));
    _copy(_type, &v.value(), _data.data());
  }
  const TypeInfo &type() const { return _type; }
  const void *data() const {
    _check();
    return _data.data();
  }
  void *data() {
    _check();
    return _data.data();
  }
  template <class T> const T &value() const {
    _check();
    return *(const T *)data();
  }
  template <class T> T &value() {
    _check();
    return *(T *)data();
  }
  template <class T> bool is() const {
    if (empty()) {
      return false;
    }
    return _type == TypeInfo::get<T>();
  }
  inline bool empty() const { return _data.empty(); }
  template <class Op, class... Args> static Any call(Args &&...args) {
    return _call(_findOp<Op>(args...), sizeof...(Args), &args...);
  }
};

Any operator+(const Any &a, const Any &b);
Any operator-(const Any &a, const Any &b);
Any operator*(const Any &a, const Any &b);
Any operator/(const Any &a, const Any &b);

} // namespace tractor
