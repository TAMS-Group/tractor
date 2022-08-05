// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/factory.h>
#include <tractor/core/list.h>
#include <tractor/core/ops.h>
#include <tractor/core/profiler.h>

#define TRACTOR_CHECK_TENSOR_DIMENSIONS(module, input, dims)                   \
  {                                                                            \
    if (input.shape().dimensions() != dims) {                                  \
      throw std::invalid_argument(                                             \
          std::string() + module +                                             \
          ": wrong number of dimensions, expected:" + std::to_string(dims) +   \
          ", found:" + std::to_string(input.shape().dimensions()));            \
    }                                                                          \
  }

namespace tractor {

template <class T> Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b) {
  return add(a, b);
}
template <class T> Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b) {
  return sub(a, b);
}
template <class T> Tensor<T> operator*(const Tensor<T> &a, const Tensor<T> &b) {
  return mul(a, b);
}
template <class T> Tensor<T> operator/(const Tensor<T> &a, const Tensor<T> &b) {
  return div(a, b);
}

template <class T> Tensor<T> &operator+=(Tensor<T> &a, const Tensor<T> &b) {
  a = add(a, b);
  return a;
}
template <class T> Tensor<T> &operator-=(Tensor<T> &a, const Tensor<T> &b) {
  a = sub(a, b);
  return a;
}
template <class T> Tensor<T> &operator*=(Tensor<T> &a, const Tensor<T> &b) {
  a = mul(a, b);
  return a;
}
template <class T> Tensor<T> &operator/=(Tensor<T> &a, const Tensor<T> &b) {
  a = div(a, b);
  return a;
}

// ------------------------------------------

template <class Value>
void unpack(const Tensor<Value> &tensor, std::vector<Var<Value>> &vector) {

  static Factory<const TensorInfo *, const Operator *> factory{
      [](const TensorInfo *tensor_info) {
        std::vector<Operator::Argument> args;
        args.push_back(Operator::Argument::makeInput(tensor_info->type()));
        for (size_t i = 0; i < tensor_info->shape().elementCount(); i++) {
          args.push_back(
              Operator::Argument::makeOutput(TypeInfo::get<Value>()));
        }

        size_t element_count = tensor_info->shape().elementCount();
        const Operator *op = makeListOperator(
            std::string() + "unpack_" + TypeInfo::get<Value>().name() + "_" +
                std::to_string(element_count),
            "unpack", args,
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor unpack nl");
              for (size_t i = 0; i < element_count; i++) {
                args.arg<Value>(i + 1) = args.argp<Value>(0)[i];
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor unpack f");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.arg<Value>(d + i + 1) = args.argp<Value>(d)[i];
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor unpack r");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.argp<Value>(d)[i] = args.arg<Value>(d + i + 1);
              }
            });
        return op;
      }};

  auto *op = factory[tensor.info()];

  vector.resize(tensor.info()->shape().elementCount());

  std::vector<void *> args;
  {
    args.push_back((void *)tensor.data());
    for (auto &e : vector) {
      args.push_back((void *)&e);
    }
  }

  callAndRecord(op, args.data());
}

template <class Value>
std::vector<Var<Value>> unpack(const Tensor<Value> &tensor) {
  std::vector<Var<Value>> ret;
  unpack(tensor, ret);
  return ret;
}

// ------------------------------------------

template <class Value>
Tensor<Value> pack_tensor(const Var<Value> *data, const TensorShape &shape) {

  static Factory<const TensorInfo *, const Operator *> factory{
      [](const TensorInfo *tensor_info) {
        std::vector<Operator::Argument> args(
            tensor_info->shape().elementCount(),
            Operator::Argument::makeInput(TypeInfo::get<Value>()));
        args.push_back(Operator::Argument::makeOutput(tensor_info->type()));

        size_t element_count = tensor_info->shape().elementCount();
        const Operator *op = makeListOperator(
            std::string() + "pack_" + TypeInfo::get<Value>().name() + "_" +
                std::to_string(element_count),
            "pack", args,
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor pack nl");
              for (size_t i = 0; i < element_count; i++) {
                args.argp<Value>(element_count)[i] = args.arg<Value>(i);
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor pack f");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.argp<Value>(d + element_count)[i] = args.arg<Value>(d + i);
              }
            },
            [element_count](const ArgList &args) {
              TRACTOR_PROFILER("tensor pack r");
              size_t d = element_count + 1;
              for (size_t i = 0; i < element_count; i++) {
                args.arg<Value>(d + i) = args.argp<Value>(d + element_count)[i];
              }
            });
        return op;
      }};

  Tensor<Value> ret(shape);

  auto *op = factory[ret.info()];

  std::vector<void *> args;
  {
    size_t n = shape.elementCount();
    for (size_t i = 0; i < n; i++) {
      args.push_back((void *)(data + i));
    }
    args.push_back(ret.data());
  }

  callAndRecord(op, args.data());

  return ret;
}

template <class Value>
Tensor<Value> pack_tensor(const std::vector<Var<Value>> &vector,
                          const TensorShape &shape) {
  if (vector.size() != shape.elementCount()) {
    throw std::runtime_error("pack failed, shapes not compatible");
  }
  return pack_tensor(vector.data(), shape);
}

template <class Value>
Tensor<Value> pack_tensor(const std::vector<Var<Value>> &vector) {
  return pack_tensor(vector.data(), TensorShape(vector.size()));
}

template <class Value>
Tensor<Value> make_tensor(const TensorShape &shape, const Value &v) {
  std::vector<Var<Value>> vector(shape.elementCount(), Var<Value>(v));
  return pack_tensor(vector, shape);
}

} // namespace tractor
