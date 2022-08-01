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

public:
  Any() {}
  explicit Any(const TypeInfo &type);
  explicit Any(const TypeInfo &type, const void *data);
  Any(const Any &other);
  Any &operator=(const Any &other);
  template <class T> explicit Any(const Var<T> &v) {
    _type = TypeInfo::get<T>();
    _data.resize(sizeof(T));
    // std::memcpy(_data.data(), &v.value(), sizeof(T));
    _copy(_type, &v.value(), _data.data());
  }
  const TypeInfo &type() const;
  const void *data() const { return _data.data(); }
  void *data() { return _data.data(); }
  template <class T> const T &value() const { return *(const T *)data(); }
  template <class T> T &value() { return *(T *)data(); }
  template <class T> bool is() const { return _type == TypeInfo::get<T>(); }
};

} // namespace tractor
