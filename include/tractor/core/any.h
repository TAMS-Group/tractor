// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/allocator.h>
#include <tractor/core/type.h>

#include <iostream>
#include <vector>

namespace tractor {

class Operator;

class Any {
  TypeInfo _type;
  AlignedStdVector<uint8_t> _data;
  static void _copy(const TypeInfo &type, const void *from, void *to);
  void _check() const;
  static TypeInfo _getType(const Any &any) { return any.type(); }

 public:
  Any() {}
  explicit Any(const TypeInfo &type);
  explicit Any(const TypeInfo &type, const void *data);
  Any(const Any &other);
  Any &operator=(const Any &other);
  const TypeInfo &type() const { return _type; }
  const void *data() const {
    _check();
    return _data.data();
  }
  void *data() {
    _check();
    return _data.data();
  }
  template <class T>
  const T &value() const {
    _check();
    return *(const T *)data();
  }
  template <class T>
  T &value() {
    _check();
    return *(T *)data();
  }
  template <class T>
  bool is() const {
    if (empty()) {
      return false;
    }
    return _type == TypeInfo::get<T>();
  }
  inline bool empty() const { return _data.empty(); }
};

}  // namespace tractor
