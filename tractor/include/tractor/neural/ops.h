// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>

namespace tractor {

template <class T, size_t S> struct DenseArg {
  std::array<T, S> values;
  template <class... Args> inline void set(const Args &...vv) {
    values = std::array<T, S>({vv...});
  }
  auto &operator[](size_t i) { return values[i]; }
  auto &operator[](size_t i) const { return values[i]; }
};
template <class A, class B, size_t N> struct DenseLin {
  DenseArg<A, N> a;
  DenseArg<B, N> b;
};

template <class T> inline T dense_batch_sum(const T &v) { return v; }
template <class T, size_t S> inline T dense_batch_sum(const Batch<T, S> &v) {
  T ret = 0;
  for (size_t i = 0; i < S; i++) {
    ret += v[i];
  }
  return ret;
}

// template <class T, class S, size_t N>
// inline void dense_compute(const std::array<T, N> &a, const std::array<S, N>
// &b,
//                           const T &add, T &x) {
//   x = add;
//   for (size_t i = 0; i < N; i++) {
//     x += a[i] * T(b[i]);
//   }
// }

template <class T, class S> inline T dense_compute_ab(const T &a, const S &b) {
  return a * T(b);
}
template <class T, class S, class... Args>
inline T dense_compute_ab(const T &a, const S &b, const Args &...args) {
  return a * T(b) + dense_compute_ab(args...);
}

template <class T, class S, size_t N>
inline void dense_forward(const DenseLin<T, S, N> &p, const std::array<T, N> &a,
                          const std::array<S, N> &b, const T &add, T &x) {
  x = add;
  for (size_t i = 0; i < N; i++) {
    x += a[i] * T(p.b[i]) + p.a[i] * T(b[i]);
  }
}

template <class T, class S, size_t N>
inline void dense_reverse(const DenseLin<T, S, N> &p, size_t i, const T &x,
                          T &a, S &b) {
  a = x * T(p.b[i]);
  b = dense_batch_sum(x * p.a[i]);
}
template <class T, class S, size_t N, class... Args>
inline void dense_reverse(const DenseLin<T, S, N> &p, size_t i, const T &x,
                          T &a, S &b, Args &...args) {
  a = x * T(p.b[i]);
  b = dense_batch_sum(x * p.a[i]);
  dense_reverse(p, i + 1, x, args...);
}

#define DENSE4ARGS(m)                                                          \
  m T &a0, m T &a1, m T &a2, m T &a3, m S &b0, m S &b1, m S &b2, m S &b3

TRACTOR_OP(dense4, (DENSE4ARGS(const), const T &add),
           { return dense_compute_ab(a0, b0, a1, b1, a2, b2, a3, b3) + add; })
TRACTOR_D(prepare, dense4,
          (DENSE4ARGS(const), const T &add, const T &x, DenseLin<T, S, 4> &p), {
            p.a.set(a0, a1, a2, a3);
            p.b.set(b0, b1, b2, b3);
          })
TRACTOR_D(forward, dense4,
          (const DenseLin<T, S, 4> &p, DENSE4ARGS(const), const T &add, T &x), {
            dense_forward(p, {a0, a1, a2, a3}, {b0, b1, b2, b3}, add, x);
          })
TRACTOR_D(reverse, dense4,
          (const DenseLin<T, S, 4> &p, DENSE4ARGS(), T &add, const T &x), {
            dense_reverse(p, 0, x, a0, b0, a1, b1, a2, b2, a3, b3);
            add = x;
          })

// template <class T, class S, size_t N>
// inline void dense_reverse(const DenseLin<T, S, N> &p,
//                           const std::array<T *, N> &a,
//                           const std::array<S *, N> &b, T &add, const T &x) {
//   for (size_t i = 0; i < N; i++) {
//     *a[i] = x * T(p.b[i]);
//     *b[i] = dense_batch_sum(x * p.a[i]);
//   }
//   add = x;
// }

// TRACTOR_D(
//     reverse, dense4,
//     (const DenseLin<T, S, 4> &p, DENSE4ARGS(), T &add, const T &x), {
//       dense_reverse(p, {&a0, &a1, &a2, &a3}, {&b0, &b1, &b2, &b3}, add, x);
//     })

// a0 = x * T(p.b[0]);
// a1 = x * T(p.b[1]);
// a2 = x * T(p.b[2]);
// a3 = x * T(p.b[3]);
// b0 = dense_batch_sum(x * p.a[0]);
// b1 = dense_batch_sum(x * p.a[1]);
// b2 = dense_batch_sum(x * p.a[2]);
// b3 = dense_batch_sum(x * p.a[3]);
// add = x;

// TRACTOR_OP(dense4,
//            (const T &a0, const T &a1, const T &a2, const T &a3, const S &b0,
//             const S &b1, const S &b2, const S &b3, const T &add),
//            {
//              // return T(0);
//              return a0 * T(b0) + a1 * T(b1) + a2 * T(b2) + a3 * T(b3) + add;
//            })

// TRACTOR_OP(dense4, (DENSE4ARGS(const), const T &add), {
//   return batch(a0, 4) * b0 + batch(a1, 4) * b1 + batch(a2) * b2 +
//          batch(a3) * b3 + add;
// })
/*
TRACTOR_D(reverse, dense4,
          (const DenseLin<S, T, 4> &p, DENSE4ARGS(), T &add, const T &x), {
            // a0 = x * p.b[0];
            // a1 = x * p.b[1];
            // a2 = x * p.b[2];
            // a3 = x * p.b[3];
            b0 = x * p.a[0];
            b1 = x * p.a[1];
            b2 = x * p.a[2];
            b3 = x * p.a[3];
            add = x;
          })
*/
// ------------------------------------------

template <class T> auto relu(const T &a) { return std::max(T(0), a); }
TRACTOR_OP(relu, (const T &a), { return relu(a); })
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
