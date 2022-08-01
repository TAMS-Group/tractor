// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/allocator.h>
#include <tractor/core/any.h>
#include <tractor/core/lambda.h>
#include <tractor/core/list.h>
#include <tractor/core/var.h>

namespace tractor {

class TensorShape {
  std::vector<size_t> _data;

public:
  TensorShape() {}
  TensorShape(const std::vector<size_t> &s) : _data(s) {}
  template <class... Args>
  TensorShape(const Args &&...args) : _data({args...}) {}
  size_t dimensions() const { return _data.size(); }
  size_t operator[](size_t i) const { return _data[i]; }
  size_t at(size_t i) const { return _data[i]; }
  size_t elementCount() const {
    if (_data.empty()) {
      return 0;
    }
    size_t n = 1;
    for (auto &v : _data) {
      n *= v;
    }
    return n;
  }
  auto begin() const { return _data.begin(); }
  auto end() const { return _data.end(); }
  bool operator==(const TensorShape &other) const {
    return _data == other._data;
  }
  bool operator!=(const TensorShape &other) const {
    return _data != other._data;
  }
  bool empty() const { return _data.empty(); }
};

class TensorOperators {
  const Operator *_add = nullptr;
  const Operator *_zero = nullptr;
  const Operator *_move = nullptr;

public:
  TensorOperators() {}
  TensorOperators(const TypeInfo &element_type, const TypeInfo &tensor_type,
                  const TensorShape &tensor_shape,
                  void (*add)(size_t, const void *, const void *, void *));
  const Operator *add() const { return _add; }
  const Operator *zero() const { return _zero; }
  const Operator *move() const { return _move; }
};

class TensorInfo {
  const std::string _name;
  TensorShape _shape;
  TypeInfo _type;
  TensorOperators _operators;
  TensorInfo(const std::string &name, const TypeInfo &element_type,
             const TensorShape &shape,
             void (*add)(size_t, const void *, const void *, void *));
  static const TensorInfo *
  _make(const TypeInfo &element, const TensorShape &shape,
        void (*add)(size_t, const void *, const void *, void *));

public:
  const std::string &name() const { return _name; }
  const TensorShape &shape() const { return _shape; }
  const TypeInfo &type() const { return _type; }
  template <class T> static const TensorInfo *make(const TensorShape &shape) {
    return _make(TypeInfo::get<T>(), shape,
                 [](size_t n, const void *va, const void *vb, void *vx) {
                   const T *a = (const T *)va;
                   const T *b = (const T *)vb;
                   T *x = (T *)vx;
                   for (size_t i = 0; i < n; i++) {
                     x[i] = a[i] + b[i];
                   }
                 });
  }
  const TensorOperators &operators() const { return _operators; }
};

template <class T> class Tensor2 {

  const TensorInfo *_tensor_info = nullptr;
  Any _data;

  bool _throwIfEmpty() const {
    if (empty()) {
      throw std::runtime_error("tensor not initialized");
    }
  }

  // template <class... Indices> size_t _index(const Indices &...indices) const
  // {
  //   auto &shape = this->shape();
  //   if (shape.dimensions() != sizeof...(Indices)) {
  //     throw std::runtime_error("incorrect number of tensor index
  //     dimensions");
  //   }
  //   std::array<size_t, sizeof...(Indices)> ii = {indices...};
  //   size_t ret = 0;
  //   for (size_t i = 0; i < sizeof...(Indices); i++) {
  //     if (ii.at(i) >= shape.at(i)) {
  //       throw std::runtime_error("tensor index out of range");
  //     }
  //     ret *= shape[i];
  //     ret += ii[i];
  //   }
  //   return ret;
  // }

public:
  inline bool empty() const { return _tensor_info == nullptr; }
  const T *data() const {
    _throwIfEmpty();
    return (const T *)_data.data();
  }
  T *data() {
    _throwIfEmpty();
    return (T *)_data.data();
  }
  TensorShape shape() const {
    if (empty()) {
      return TensorShape();
    } else {
      return _tensor_info->shape();
    }
  }
  Tensor2() {}
  Tensor2(const TensorShape &shape) {
    if (!shape.empty()) {
      _tensor_info = TensorInfo::make<T>(shape);
      _data = Any(type());
    }
  }
  Tensor2(const TensorShape &shape, const T *data) {
    if (!shape.empty()) {
      _tensor_info = TensorInfo::make<T>(shape);
      _data = Any(type(), data);
    }
  }
  TypeInfo type() const {
    if (empty()) {
      return TypeInfo();
    } else {
      return _tensor_info->type();
    }
  }
  const TensorInfo &info() const {
    if (!_tensor_info) {
      throw std::runtime_error("tensor is empty");
    }
    return *_tensor_info;
  }
};

template <class T>
void add(const Tensor2<T> &a, const Tensor2<T> &b, Tensor2<T> &x) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("tensor shape mismatch");
  }
  x = Tensor2<T>(a.shape());
  auto *op_add = a.info().operators().add();
  op_add->invoke(a.data(), b.data(), x.data());
  if (auto *rec = Recorder::instance()) {
    rec->op(op_add);
    rec->push((uintptr_t)a.data());
    rec->push((uintptr_t)b.data());
    rec->push((uintptr_t)x.data());
  }
}

} // namespace tractor
