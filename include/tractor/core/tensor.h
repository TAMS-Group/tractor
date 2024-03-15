// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/any.h>
#include <tractor/core/recorder.h>

#include <boost/container/small_vector.hpp>

namespace tractor {

class TensorShape {
  boost::container::small_vector<size_t, 4> _data;

 public:
  TensorShape() {}
  explicit TensorShape(const std::initializer_list<size_t> &s) : _data(s) {}
  explicit TensorShape(const std::vector<size_t> &s)
      : _data(s.begin(), s.end()) {}
  template <class... Args>
  explicit TensorShape(size_t s, Args &&...args) : _data({s, args...}) {}
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
  size_t hash() const noexcept;
  size_t last(size_t i = 0) const { return _data.at(_data.size() - 1 - i); }
  friend size_t hash_value(const TensorShape &s) noexcept { return s.hash(); }
};

std::ostream &operator<<(std::ostream &s, const TensorShape &v);

class TensorOperators {
  const Operator *_add = nullptr;
  const Operator *_zero = nullptr;
  const Operator *_move = nullptr;

 public:
  TensorOperators() {}
  TensorOperators(const TypeInfo &element_type, const TypeInfo &tensor_type,
                  const TensorShape &tensor_shape);
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
             const TensorShape &shape);
  static const TensorInfo *_make(const TypeInfo &element,
                                 const TensorShape &shape);

 public:
  const std::string &name() const { return _name; }
  const TensorShape &shape() const { return _shape; }
  const TypeInfo &type() const { return _type; }
  static const TensorInfo *make(const TypeInfo &element_type,
                                const TensorShape &shape) {
    return _make(element_type, shape);
  }
  template <class T>
  static const TensorInfo *make(const TensorShape &shape) {
    return _make(TypeInfo::get<T>(), shape);
  }
  const TensorOperators &operators() const { return _operators; }
};

template <class T>
class Tensor {
  const TensorInfo *_tensor_info = nullptr;
  Any _data;
  bool _throwIfEmpty() const {
    if (empty()) {
      throw std::runtime_error("tensor not initialized");
    }
  }

 public:
  inline bool empty() const { return _tensor_info == nullptr; }
  const T *data() const {
    if (!empty()) {
      return (const T *)_data.data();
    } else {
      return nullptr;
    }
  }
  T *data() {
    if (!empty()) {
      return (T *)_data.data();
    } else {
      return nullptr;
    }
  }
  const TensorShape &shape() const {
    if (empty()) {
      static TensorShape empty_shape;
      return empty_shape;
    } else {
      return _tensor_info->shape();
    }
  }
  Tensor() {}
  Tensor(const TensorShape &shape) {
    if (!shape.empty()) {
      _tensor_info = TensorInfo::make<T>(shape);
      _data = Any(type());
    }
  }
  Tensor(const TensorShape &shape, const T *data) {
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
  const TensorInfo *info() const { return _tensor_info; }
  size_t sizeInBytes() const { return shape().elementCount() * sizeof(T); }
};

template <class Scalar>
std::ostream &operator<<(std::ostream &s, const Tensor<Scalar> &tensor) {
  return s << "Tensor(" << tensor.shape() << ")";
}

void emitTensorOpImpl(
    const Operator *op,
    const std::initializer_list<const TensorInfo *> &tensor_infos,
    const std::initializer_list<void *> &tensor_data);

template <class... Args>
void runTensorOpImpl(const Operator *op, Args &&...args) {
  emitTensorOpImpl(op, {args.info()...}, {(void *)args.data()...});
}

template <class Ret>
struct TensorOpCaller {
  template <class... Args>
  static Tensor<Ret> call(const Operator *op, Args &&...args) {
    const TensorShape &shape = (..., args).shape();
    Tensor<Ret> ret(shape);
    runTensorOpImpl(op, args..., ret);
    return ret;
  }
};
template <>
struct TensorOpCaller<void> {
  template <class... Args>
  static void call(const Operator *op, Args &&...args) {
    runTensorOpImpl(op, args...);
  }
};

template <class... Args>
static void checkAllTensorStatic(const Tensor<Args> &...) {}

template <class T>
struct MakeTensor {
  typedef const Tensor<T> Type;
};
template <class T>
struct MakeTensor<const T> {
  typedef const Tensor<T> Type;
};
template <class T>
struct MakeTensor<const T &> {
  typedef const Tensor<T> Type;
};
template <class T>
struct MakeTensor<T &> {
  typedef Tensor<T> Type;
};
template <class T>
struct MakeTensor<T *> {
  typedef const Tensor<T *> Type;
};

template <class T>
void variable(Tensor<T> &tensor) {
  if (auto *rec = Recorder::instance()) {
    rec->input(tensor.type(), tensor.data(), tensor.data());
  }
}
template <class T>
void parameter(Tensor<T> &tensor) {
  if (auto *rec = Recorder::instance()) {
    rec->parameter(tensor.type(), tensor.data(), tensor.data());
  }
}
template <class T>
void output(Tensor<T> &tensor) {
  if (auto *rec = Recorder::instance()) {
    rec->output(tensor.type(), tensor.data(), tensor.data());
  }
}
template <class T>
void goal(const Tensor<T> &tensor) {
  if (auto *rec = Recorder::instance()) {
    rec->goal(tensor.type(), tensor.data(), 0, "");
  }
}

}  // namespace tractor
