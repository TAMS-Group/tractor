// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/list.h>
#include <tractor/core/operator.h>
#include <tractor/tensor/tensor.h>

namespace tractor {

template <class T, size_t Dimensions> class TensorArg : public ArgList {
  std::array<size_t, Dimensions> _shape;
  template <class... Indices> size_t _index(const Indices &...indices) const {
    if (_shape.size() != sizeof...(Indices)) {
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
  TensorArg(void *base, const uintptr_t *offsets,
            const std::array<size_t, Dimensions> &shape)
      : ArgList(base, offsets), _shape(shape) {}
  size_t dimensions() const { return _shape.size(); }
  const std::array<size_t, Dimensions> &shape() const { return _shape; }
  size_t size() const {
    if (_shape.empty()) {
      return 0;
    }
    size_t s = 1;
    for (auto &v : _shape) {
      s *= v;
    }
    return s;
  }
  template <class... Indices>
  auto &operator()(const Indices &...indices) const {
    return arg<T>(_index(indices...));
  }
};

template <class T, size_t N>
TensorArg<T, N> bindTensorArg(void *base, const uintptr_t **offsets,
                              const std::array<size_t, N> &shape) {
  TensorArg<T, N> ret(base, *offsets, shape);
  (*offsets) += ret.size();
  return ret;
}

template <class ActivationScalar, class WeightScalar>
Tensor<ActivationScalar> dense_mul_vec_mat(const Tensor<ActivationScalar> &a,
                                           const Tensor<WeightScalar> &b) {

  uint64_t rows = b.shape()[0];
  uint64_t cols = b.shape()[1];

  Tensor<ActivationScalar> r;
  r.resize(cols);

  typedef typename std::decay<decltype(value(a[0]))>::type Batch;
  typedef typename std::decay<decltype(value(b[0]))>::type Weight;

  if (auto rec = Recorder::instance()) {

    std::string label = "dense_mul_vec_mat";

    std::string name = label + "_" + std::to_string(rows) + "_" +
                       std::to_string(cols) + "_" +
                       typeid(ActivationScalar).name();

    std::vector<Operator::Argument> arguments;
    arguments.push_back(Operator::Argument::make<const uint64_t &>());
    arguments.push_back(Operator::Argument::make<const uint64_t &>());
    for (size_t row = 0; row < rows; row++) {
      arguments.push_back(Operator::Argument::make<const Batch &>());
    }
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        arguments.push_back(Operator::Argument::make<const Weight &>());
      }
    }
    for (size_t col = 0; col < cols; col++) {
      arguments.push_back(Operator::Argument::make<Batch &>());
    }

    const Operator *op = makeListOperator(
        name, label, arguments,
        [](void *base, const uintptr_t *offsets) {
          auto rows = bindArg<uint64_t>(base, &offsets);
          auto cols = bindArg<uint64_t>(base, &offsets);
          auto a = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto b = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto x = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          for (size_t col = 0; col < cols; col++) {
            Batch s = Batch(0);
            for (size_t row = 0; row < rows; row++) {
              s += a(row) * Batch(b(row, col));
            }
            x(col) = s;
          }
        },
        [](void *base, const uintptr_t *offsets) {
          auto rows = bindArg<uint64_t>(base, &offsets);
          auto cols = bindArg<uint64_t>(base, &offsets);
          auto a = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto b = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto x = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          auto drows = bindArg<uint64_t>(base, &offsets);
          auto dcols = bindArg<uint64_t>(base, &offsets);
          auto da = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto db = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto dx = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          for (size_t col = 0; col < cols; col++) {
            Batch s = Batch(0);
            for (size_t row = 0; row < rows; row++) {
              s += da(row) * Batch(b(row, col)) + a(row) * Batch(db(row, col));
            }
            dx(col) = s;
          }
        },
        [](void *base, const uintptr_t *offsets) {
          auto rows = bindArg<uint64_t>(base, &offsets);
          auto cols = bindArg<uint64_t>(base, &offsets);
          auto a = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto b = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto x = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          auto drows = bindArg<uint64_t>(base, &offsets);
          auto dcols = bindArg<uint64_t>(base, &offsets);
          auto da = bindTensorArg<Batch, 1>(base, &offsets, {rows});
          auto db = bindTensorArg<Weight, 2>(base, &offsets, {rows, cols});
          auto dx = bindTensorArg<Batch, 1>(base, &offsets, {cols});
          for (size_t row = 0; row < rows; row++) {
            Batch s = Batch(0);
            for (size_t col = 0; col < cols; col++) {
              db(row, col) = batchSum(a(row) * dx(col));
              s += dx(col) * Batch(b(row, col));
            }
            da(row) = s;
          }
        });

    Var<uint64_t> vrows(rows);
    Var<uint64_t> vcols(cols);
    rec->op(op);
    rec->arg(&vrows);
    rec->arg(&vcols);
    for (size_t row = 0; row < rows; row++) {
      rec->arg(&a(row));
    }
    for (size_t row = 0; row < rows; row++) {
      for (size_t col = 0; col < cols; col++) {
        rec->arg(&b(row, col));
      }
    }
    for (size_t col = 0; col < cols; col++) {
      rec->arg(&r(col));
    }
  }

  for (size_t col = 0; col < cols; col++) {
    Batch s = Batch(0);
    for (size_t row = 0; row < rows; row++) {
      s += value(a(row)) * Batch(value(b(row, col)));
    }
    value(r(col)) = s;
  }

  return r;
}

} // namespace tractor
