// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/allocator.h>
#include <tractor/core/any.h>
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
};

TypeInfo makeTensorType(const TypeInfo &element, const TensorShape &shape);

template <class T> class Tensor2 {
  TensorShape _shape;
  Any _data;
  template <class... Indices> size_t _index(const Indices &...indices) const {
    if (_shape.dimensions() != sizeof...(Indices)) {
      throw std::runtime_error("incorrect number of tensor index dimensions");
    }
    std::array<size_t, sizeof...(Indices)> ii = {indices...};
    size_t ret = 0;
    for (size_t i = 0; i < sizeof...(Indices); i++) {
      if (ii.at(i) >= _shape.at(i)) {
        throw std::runtime_error("tensor index out of range");
      }
      ret *= _shape[i];
      ret += ii[i];
    }
    return ret;
  }

public:
  const T *data() const { return (const T *)_data.data(); }
  T *data() { return (T *)_data.data(); }
  const TensorShape &shape() const { return _shape; }
  Tensor2() {}
  Tensor2(const TensorShape &shape) {
    _shape = shape;
    _data = Any(makeTensorType(TypeInfo::get<T>(), shape));
  }
  Tensor2(const TensorShape &shape, const T *data) {
    _shape = shape;
    _data = Any(makeTensorType(TypeInfo::get<T>(), shape), data);
  }
  template <class... Indices>
  auto &operator()(const Indices &...indices) const {
    return data()[_index(indices...)];
  }
  template <class... Indices> auto &operator()(const Indices &...indices) {
    return data()[_index(indices...)];
  }
  TypeInfo type() const { return makeTensorType(TypeInfo::get<T>(), shape()); }
};

template <class... Args> struct PointerOp : Operator {
  std::function<void(Args...)> functor;
  template <size_t... Indices>
  void init(const std::integer_sequence<size_t, Indices...> &indices) {
    _functions.indirect = [](void *base, const uintptr_t *offsets) {
      const PointerOp *_this = (const PointerOp *)offsets[0];
      _this->functor((Args)(void *)((uint8_t *)base + offsets[Indices + 1])...);
    };
  }
  PointerOp(const std::string &name, const std::string &label,
            const OpMode &mode, const OpType &op, const OpGroup &group,
            const std::vector<Operator::Argument> &args,
            const std::function<void(Args...)> &functor)
      : Operator(name, label, mode, op, group), functor(functor) {
    std::cout << "make op " << name << std::endl;
    _arguments = args;
    _argument_count = _arguments.size();
    init(std::make_index_sequence<sizeof...(Args)>());
    std::vector<uintptr_t> context;
    context.push_back((uintptr_t)this);
    _functions.context = context;
  }
};
template <class... Args>
const Operator *makePointerOp(const std::string &name, const std::string &label,
                              const OpMode &mode, const OpType &op,
                              const OpGroup &group,
                              const std::vector<Operator::Argument> &args,
                              const std::function<void(Args...)> &functor) {
  static std::unordered_map<std::string, const Operator *> map;
  if (!map[name]) {
    map[name] =
        new PointerOp<Args...>(name, label, mode, op, group, args, functor);
  }
  return map[name];
}

template <class T>
void add(const Tensor2<T> &a, const Tensor2<T> &b, Tensor2<T> &x) {
  auto tensor_type = makeTensorType(TypeInfo::get<T>(), a.shape());
  size_t size = a.shape().elementCount();
  auto *op =
      makePointerOp(std::string() + "add_" + tensor_type.name(), "add",
                    OpMode(typeid(compute *)), OpType(typeid(op_add *)),
                    makeOpGroup(std::string() + "add_" + tensor_type.name()),
                    {
                        Operator::Argument::makeInput(tensor_type),
                        Operator::Argument::makeInput(tensor_type),
                        Operator::Argument::makeOutput(tensor_type),
                    },
                    std::function<void(const T *, const T *, T *)>(
                        [size](const T *a, const T *b, T *x) {
                          std::cout << " > tensor add " << size << std::endl;
                          for (size_t i = 0; i < size; i++) {
                            x[i] = a[i] + b[i];
                          }
                        }));
  x = Tensor2<T>(a.shape());
  op->invoke(a.data(), b.data(), x.data());
  if (auto *rec = Recorder::instance()) {
    rec->op(op);
    rec->push((uintptr_t)a.data());
    rec->push((uintptr_t)b.data());
    rec->push((uintptr_t)x.data());
  }
}

// template <class T> class TensorStorage2 {
//   TensorShape _shape;
//   std::vector<T> _data;
//
// protected:
//   void _create(const TensorShape &shape) {
//     _shape = shape;
//     _data.clear();
//     _data.resize(shape.elementCount(), 0);
//   }
//
// public:
//   const T *data() const { return _data.data(); }
//   T *data() { return _data.data(); }
//   const TensorShape &shape() const { return _shape; }
// };
//
// template <class T> class TensorStorage2<Var<T>> {
//   TensorShape _shape;
//   Any _data;
//
// protected:
//   void _create(const TensorShape &shape) {
//     _shape = shape;
//     _data = Any(makeTensorType(TypeInfo::get<T>(), shape));
//   }
//
// public:
//   const T *data() const { return _data.data(); }
//   T *data() { return _data.data(); }
//   const TensorShape &shape() const { return _shape; }
// };

} // namespace tractor
