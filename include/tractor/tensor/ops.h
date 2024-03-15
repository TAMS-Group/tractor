// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/error.h>
#include <tractor/core/factory.h>
#include <tractor/core/list.h>
#include <tractor/core/ops.h>
#include <tractor/core/profiler.h>
#include <tractor/tensor/matmul.h>

#define TRACTOR_CHECK_TENSOR_DIMENSIONS(module, input, dims)                 \
  {                                                                          \
    if (input.shape().dimensions() != dims) {                                \
      throw std::invalid_argument(                                           \
          std::string() + module +                                           \
          ": wrong number of dimensions, expected:" + std::to_string(dims) + \
          ", found:" + std::to_string(input.shape().dimensions()));          \
    }                                                                        \
  }

namespace tractor {

template <class T>
Tensor<T> operator+(const Tensor<T> &a, const Tensor<T> &b) {
  return add(a, b);
}
template <class T>
Tensor<T> operator-(const Tensor<T> &a, const Tensor<T> &b) {
  return sub(a, b);
}
template <class T>
Tensor<T> operator*(const Tensor<T> &a, const Tensor<T> &b) {
  return mul(a, b);
}
template <class T>
Tensor<T> operator/(const Tensor<T> &a, const Tensor<T> &b) {
  return div(a, b);
}

template <class T>
Tensor<T> &operator+=(Tensor<T> &a, const Tensor<T> &b) {
  a = add(a, b);
  return a;
}
template <class T>
Tensor<T> &operator-=(Tensor<T> &a, const Tensor<T> &b) {
  a = sub(a, b);
  return a;
}
template <class T>
Tensor<T> &operator*=(Tensor<T> &a, const Tensor<T> &b) {
  a = mul(a, b);
  return a;
}
template <class T>
Tensor<T> &operator/=(Tensor<T> &a, const Tensor<T> &b) {
  a = div(a, b);
  return a;
}

// ------------------------------------------

template <class Value>
void unpack(const Tensor<Value> &tensor, std::vector<Var<Value>> &vector) {
  static Factory::Key<const TensorInfo *>::Value<const Operator *> factory{
      [](const TensorInfo *tensor_info) {
        std::vector<Operator::Argument> args;
        args.push_back(Operator::Argument::makeInput(tensor_info->type()));
        for (size_t i = 0; i < tensor_info->shape().elementCount(); i++) {
          args.push_back(
              Operator::Argument::makeOutput(TypeInfo::get<Value>()));
        }

        size_t element_count = tensor_info->shape().elementCount();
        const Operator *op = makeListOperator(
            std::string() + "unpack_" + args.front().typeInfo().name(),
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

  auto *op = factory.get(tensor.info());

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
  static Factory::Key<const TensorInfo *>::Value<const Operator *> factory{
      [](const TensorInfo *tensor_info) {
        std::vector<Operator::Argument> args(
            tensor_info->shape().elementCount(),
            Operator::Argument::makeInput(TypeInfo::get<Value>()));
        args.push_back(Operator::Argument::makeOutput(tensor_info->type()));

        size_t element_count = tensor_info->shape().elementCount();
        const Operator *op = makeListOperator(
            std::string() + "pack_" + args.back().typeInfo().name(), "pack",
            args,
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

  auto *op = factory.get(ret.info());

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

// ------------------------------------------

template <class A, class B,
          class X = decltype(std::declval<A>() * std::declval<B>())>
Tensor<X> matmul(const Tensor<A> &tensor_a, const Tensor<B> &tensor_b) {
  if (tensor_b.shape().dimensions() != 2) {
    throw std::runtime_error(
        "matmul invalid number of dimensions for weight matrix");
  }
  size_t a_cols_b_rows = tensor_b.shape()[0];
  size_t b_cols = tensor_b.shape()[1];

  size_t a_rows = 0;
  size_t input_vector_neurons = 0;
  TensorShape output_shape;
  switch (tensor_a.shape().dimensions()) {
    case 1:
      a_rows = 1;
      input_vector_neurons = tensor_a.shape()[0];
      output_shape = TensorShape({b_cols});
      break;
    case 2:
      a_rows = tensor_a.shape()[0];
      input_vector_neurons = tensor_a.shape()[1];
      output_shape = TensorShape({a_rows, b_cols});
      break;
    default:
      throw std::runtime_error(
          "matmul invalid input shape for "
          "activation vector or batch of vectors");
  }
  if (input_vector_neurons != a_cols_b_rows) {
    throw std::runtime_error(
        "matmul input vector length does not "
        "match weight matrix size");
  }

  Tensor<X> tensor_x(output_shape);

  static Factory::Key<const TensorInfo *, const TensorInfo *,
                      const TensorInfo *, size_t, size_t,
                      size_t>::Value<const Operator *>
      factory{[](const TensorInfo *input_info, const TensorInfo *weight_info,
                 const TensorInfo *output_info, size_t a_cols_b_rows,
                 size_t b_cols, size_t a_rows) {
        std::vector<Operator::Argument> args = {
            Operator::Argument::makeInput(input_info->type()),
            Operator::Argument::makeInput(weight_info->type()),
            Operator::Argument::makeOutput(output_info->type()),
        };
        for (auto &a : args) {
          TRACTOR_DEBUG("tensor mul arg " << a.typeInfo().name());
        }

        std::string base_name = "matmul";
        std::string variant_name = std::string() + input_info->type().name() +
                                   "_" + weight_info->type().name();

        std::shared_ptr<ProfilerTrack> profiler_nonlinear =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_n"));

        std::shared_ptr<ProfilerTrack> profiler_forward =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_f"));

        std::shared_ptr<ProfilerTrack> profiler_reverse =
            Profiler::instance()->track(std::make_shared<ProfilerTrack>(
                __PRETTY_FUNCTION__, base_name + "_" + variant_name + "_r"));

        const Operator *op = makePointerOp(
            base_name, variant_name, args,

            [a_rows, a_cols_b_rows, b_cols, profiler_nonlinear](
                const A *a, const B *b, X *x) TRACTOR_FAST {
              ProfilerScope profiler_scope(*profiler_nonlinear);
              matmul_compute(a_rows, a_cols_b_rows, b_cols, a, b, x);
            },

            [a_rows, a_cols_b_rows, b_cols, profiler_forward](
                const A *a, const B *b, const X *x, const A *da, const B *db,
                X *dx) TRACTOR_FAST {
              ProfilerScope profiler_scope(*profiler_forward);
              matmul_forward(a_rows, a_cols_b_rows, b_cols, a, b, x, da, db,
                             dx);
            },

            [a_rows, a_cols_b_rows, b_cols, profiler_reverse](
                const A *a, const B *b, const X *x, A *da, B *db, const X *dx)
                TRACTOR_FAST {
                  ProfilerScope profiler_scope(*profiler_reverse);
                  matmul_reverse(a_rows, a_cols_b_rows, b_cols, a, b, x, da, db,
                                 dx);
                });

        return op;
      }};

  auto *op = factory.get(tensor_a.info(), tensor_b.info(), tensor_x.info(),
                         a_cols_b_rows, b_cols, a_rows);

  std::array<void *, 3> args = {
      (void *)tensor_a.data(),
      (void *)tensor_b.data(),
      (void *)tensor_x.data(),
  };
  callAndRecord(op, args.data());

  return tensor_x;
}

}  // namespace tractor
