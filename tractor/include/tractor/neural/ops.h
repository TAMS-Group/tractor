// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/factory.h>
#include <tractor/core/lambda.h>
#include <tractor/core/operator.h>

#include <random>

namespace tractor {

template <class Activation, class Weight>
Tensor<Activation> dense_mul_vec_mat(const Tensor<Activation> &activations,
                                     const Tensor<Weight> &weights) {

  if (activations.shape().dimensions() != 1) {
    throw std::runtime_error(
        "invalid number of dimensions for dense_mul_vec_mat");
  }

  if (weights.shape().dimensions() != 2) {
    throw std::runtime_error(
        "invalid number of dimensions for dense_mul_vec_mat");
  }

  if (activations.shape()[0] != weights.shape()[0]) {
    throw std::runtime_error(
        "incompatible tensor shapes for dense_mul_vec_mat");
  }

  Tensor<Activation> output(TensorShape(weights.shape()[1]));

  static Factory<const TensorInfo *, const Operator *> factory{
      [](const TensorInfo *weights_info) {
        auto act_type = TypeInfo::get<Activation>();
        size_t rows = weights_info->shape()[0];
        size_t cols = weights_info->shape()[1];
        std::vector<Operator::Argument> args = {
            Operator::Argument::makeInput(
                TensorInfo::make(act_type, TensorShape(rows))->type()),
            Operator::Argument::makeInput(weights_info->type()),
            Operator::Argument::makeOutput(
                TensorInfo::make(act_type, TensorShape(cols))->type()),
        };
        for (auto &a : args) {
          std::cout << "tensor mul arg " << a.typeInfo().name() << std::endl;
        }
        const Operator *op = makePointerOp(
            "dense_mul_vec_mat",
            std::string() + TypeInfo::get<Activation>().name() + "_" +
                weights_info->name(),
            args,
            [rows, cols](const Activation *a, const Weight *b, Activation *x) {
              for (size_t col = 0; col < cols; col++) {
                Activation v = Activation(0);
                for (size_t row = 0; row < rows; row++) {
                  v += a[row] * Activation(b[row * cols + col]);
                }
                x[col] = v;
              }
            },
            [rows, cols](const Activation *a, const Weight *b, Activation *x,
                         const Activation *da, const Weight *db,
                         Activation *dx) {
              for (size_t col = 0; col < cols; col++) {
                Activation dv = Activation(0);
                for (size_t row = 0; row < rows; row++) {
                  dv += da[row] * Activation(b[row * cols + col]) +
                        a[row] * Activation(db[row * cols + col]);
                }
                dx[col] = dv;
              }
            },
            [rows, cols](const Activation *a, const Weight *b, Activation *x,
                         Activation *da, Weight *db, const Activation *dx) {
              for (size_t row = 0; row < rows; row++) {
                Activation dv = Activation(0);
                for (size_t col = 0; col < cols; col++) {
                  dv += dx[col] * Activation(b[row * cols + col]);
                }
                da[row] = dv;
              }
              for (size_t row = 0; row < rows; row++) {
                for (size_t col = 0; col < cols; col++) {
                  batch_sum(dx[col] * a[row], db[row * cols + col]);
                }
              }
            });
        return op;
      }};

  auto *op = factory[weights.info()];

  std::array<void *, 3> args = {
      (void *)activations.data(),
      (void *)weights.data(),
      (void *)output.data(),
  };
  callAndRecord(op, args.data());

  return output;
}

// ------------------------------------------

template <class T> T add_random_normal(const T &a, const T &s) {
  static thread_local std::mt19937 rng{std::mt19937::result_type(rand())};
  std::normal_distribution<double> dist;
  return a + T(dist(rng)) * s;
}
template <class T, size_t S>
Batch<T, S> add_random_normal(const Batch<T, S> &a, const T &s) {
  Batch<T, S> ret;
  for (size_t i = 0; i < S; i++) {
    ret[i] = add_random_normal(a[i], s);
  }
  return ret;
}
TRACTOR_OP(add_random_normal, (const T &a, const S &s),
           { return add_random_normal(a, s); })
TRACTOR_D(prepare, add_random_normal, (const T &a, const S &s, const T &x), {})
TRACTOR_D(forward, add_random_normal, (const T &da, const S &ds, T &dx),
          { dx = da; })
TRACTOR_D(reverse, add_random_normal, (T & da, S &ds, const T &dx), {
  da = dx;
  ds = S(0);
})

// ------------------------------------------

template <class T> T add_random_uniform(const T &a, const T &l, const T &h) {
  static thread_local std::mt19937 rng{std::mt19937::result_type(rand())};
  std::uniform_real_distribution<double> dist(l, h);
  return a + T(dist(rng));
}
template <class T, size_t S>
Batch<T, S> add_random_uniform(const Batch<T, S> &a, const T &l, const T &h) {
  Batch<T, S> ret;
  for (size_t i = 0; i < S; i++) {
    ret[i] = add_random_uniform(a[i], l, h);
  }
  return ret;
}
TRACTOR_OP(add_random_uniform, (const T &a, const S &l, const S &h),
           { return add_random_uniform(a, l, h); })
TRACTOR_D(prepare, add_random_uniform,
          (const T &a, const S &l, const S &h, const T &x), {})
TRACTOR_D(forward, add_random_uniform,
          (const T &da, const S &dl, const S &dh, T &dx), { dx = da; })
TRACTOR_D(reverse, add_random_uniform, (T & da, S &dl, S &dh, const T &dx), {
  da = dx;
  dl = S(0);
  dh = S(0);
})

// ------------------------------------------

// template <class T> inline void dropout3(const T &a, const T &b, T &x, T &y) {
//   static thread_local std::mt19937 rng{std::mt19937::result_type(rand())};
//   std::uniform_real_distribution<double> dist;
//   y = (dist(rng) < b) ? T(0) : T(1.0 / (1.0 - b));
//   x = a * y;
// }
// template <class T, size_t S>
// inline void dropout3(const Batch<T, S> &a, const T &b, Batch<T, S> &x,
//                      Batch<T, S> &y) {
//   for (size_t i = 0; i < S; i++) {
//     dropout2(a[i], b, x[i], y[i]);
//   }
// }
// TRACTOR_OP(dropout2, (const T &a, const S &b, T &x, T &y),
//            { dropout3(a, b, x, y); })
// TRACTOR_D(prepare, dropout2,
//           (const T &a, const S &b, const T &x, const T &y, T &s), { s = y; })
// TRACTOR_D(forward, dropout2,
//           (const T &s, const T &da, const S &db, T &dx, T &dy), {
//             dx = da * s;
//             dy = T(0);
//           })
// TRACTOR_D(reverse, dropout2,
//           (const T &s, T &da, S &db, const T &dx, const T &dy), {
//             da = dx * s;
//             db = S(0);
//           })
//
// template <class A, class B> auto dropout(const A &a, const B &b) {
//   A x, y;
//   dropout2(a, b, x, y);
//   return x;
// }

// ------------------------------------------

TRACTOR_OP(relu, (const T &a), { return std::max(T(0), a); })
TRACTOR_D_LOOP(forward, relu, (const T &a, const T &x, const T &da, T &dx),
               (a, x, da, dx), {
                 if (a >= T(0)) {
                   dx = da;
                 } else {
                   dx = T(0);
                 }
               })
TRACTOR_D_LOOP(reverse, relu, (const T &a, const T &x, T &da, const T &dx),
               (a, x, da, dx), {
                 if (a >= T(0)) {
                   da = dx;
                 } else {
                   da = T(0);
                 }
               })

} // namespace tractor
