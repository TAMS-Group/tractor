// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/ops.h>
#include <tractor/core/recorder.h>
#include <tractor/core/var.h>

namespace tractor {

#define VAR_OP(op, fn)                                              \
                                                                    \
  template <class A, class B,                                       \
            class Ret = decltype(tractor::fn(*(const A *)nullptr,   \
                                             *(const B *)nullptr))> \
  Ret operator op(const A &a, const B &b) {                         \
    return std::move(tractor::fn(a, b));                            \
  }                                                                 \
                                                                    \
  template <class A, class B>                                       \
  auto &operator op##=(tractor::Var<A> &a, const B & b) {           \
    a = std::move(tractor::fn(a, b));                               \
    return a;                                                       \
  }

VAR_OP(+, add)
VAR_OP(-, sub)
VAR_OP(*, mul)
VAR_OP(/, div)

template <class T>
Var<T> operator-(const Var<T> &v) {
  return minus(v);
}

typedef Var<float> tfloat;
typedef Var<double> tdouble;

template <class T>
std::ostream &operator<<(std::ostream &stream, const Var<T> &v) {
  return stream << v.value();
}

template <class T>
const T &value(const T &v) {
  return v;
}
template <class T>
const T &value(const Var<T> &v) {
  return v.value();
}

template <class T>
T &value(T &v) {
  return v;
}
template <class T>
T &value(Var<T> &v) {
  return v.value();
}

}  // namespace tractor
