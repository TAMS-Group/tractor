// 2020-2024 Philipp Ruppel

#pragma once

#include <cmath>

namespace tractor {

template <class Scalar, class Function>
static Scalar minimizeTernary(const Function &f, const Scalar &tolerance,
                              Scalar low = Scalar(0), Scalar high = Scalar(1)) {
  Scalar span = std::abs(high - low);
  while (tolerance < span) {
    Scalar a = (low + low + high) * Scalar(1.0 / 3.0);
    Scalar b = (low + high + high) * Scalar(1.0 / 3.0);
    auto fa = f(a);
    auto fb = f(b);
    if (fb < fa) {
      low = a;
    } else {
      high = b;
    }
    span *= Scalar(2.0 / 3.0);
  }
  Scalar a = low;
  Scalar b = (high + low) * Scalar(0.5);
  auto fa = f(a);
  auto fb = f(b);
  if (fb < fa) {
    return b;
  } else {
    return a;
  }
}

template <class Scalar, class Function>
static Scalar minimizeGoldenSection(const Function &f, const Scalar &tolerance,
                                    const Scalar &low = Scalar(0),
                                    const Scalar &high = Scalar(1)) {
  static constexpr Scalar step = (std::sqrt(5.0) - 1.0) * 0.5;
  Scalar x1 = low;
  Scalar x4 = high;
  Scalar x2 = x4 - (x4 - x1) * step;
  Scalar x3 = x1 + (x4 - x1) * step;
  auto f2 = f(x2);
  auto f3 = f(x3);
  while (std::abs(x4 - x1) > tolerance) {
    if (f3 < f2) {
      x1 = x2;
      x2 = x4 - (x4 - x1) * step;
      x3 = x1 + (x4 - x1) * step;
      f2 = f3;
      f3 = f(x3);
      // f2 = f(x2);
    } else {
      x4 = x3;
      x2 = x4 - (x4 - x1) * step;
      x3 = x1 + (x4 - x1) * step;
      f3 = f2;
      f2 = f(x2);
      // f3 = f(x3);
    }
  }
  Scalar x = (x2 + x3) * Scalar(0.5);
  // Scalar x = x2;
  // return x;
  auto fx = f(x);
  if (f(high) < fx) {
    return high;
  }
  if (fx < f(low)) {
    return x;
  }
  return low;
}

template <class Scalar, class Function>
static Scalar rootBisect(const Function &f, const Scalar &tolerance,
                         const Scalar &low = Scalar(0),
                         const Scalar &high = Scalar(1)) {
  Scalar x1 = low;
  Scalar x3 = high;

  while (std::abs(x3 - x1) > tolerance) {
    Scalar x2 = (x1 + x3) * Scalar(0.5);
    auto f2 = f(x2);

    if (f2 < Scalar(0)) {
      x1 = x2;
    } else {
      x3 = x2;
    }
  }

  return x1;
}

}  // namespace tractor
