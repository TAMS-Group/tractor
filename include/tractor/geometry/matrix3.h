// 2020-2024 Philipp Ruppel

#pragma once

#include <cstddef>

#include <tractor/geometry/vector3.h>

namespace tractor {

template <class Scalar>
class Matrix3 {
  Scalar _data[9];

 public:
  inline Matrix3() { setZero(); }
  template <class T, class R = decltype(Scalar(std::declval<T>()))>
  explicit Matrix3(const Matrix3<T> &other) {
    for (size_t i = 0; i < 3; i++) {
      for (size_t j = 0; j < 3; j++) {
        (*this)(i, j) = Scalar(other(i, j));
      }
    }
  }
  inline auto &operator()(size_t row, size_t col) const {
    return _data[row * 3 + col];
  }
  inline auto &operator()(size_t row, size_t col) {
    return _data[row * 3 + col];
  }
  inline void setZero() {
    for (size_t i = 0; i < 9; i++) {
      _data[i] = Scalar(0);
    }
  }
  static Matrix3 Zero() {
    Matrix3 ret;
    ret.setZero();
    return ret;
  }
  static Matrix3 Identity() {
    Matrix3 ret;
    ret.setZero();
    for (size_t i = 0; i < 3; i++) {
      ret(i, i) = Scalar(1);
    }
    return ret;
  }
  // inline const Scalar *data() const { return _data; }
  // inline Scalar *data() { return _data; }
  Matrix3 operator-() const {
    Matrix3 ret;
    for (size_t i = 0; i < 9; i++) {
      ret._data[i] = -_data[i];
    }
    return ret;
  }
  const Scalar *data() const { return _data; }
  Scalar *data() { return _data; }
};

template <class T>
void variable(Matrix3<Var<T>> &v) {
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      variable(v(i, j));
    }
  }
}
template <class T>
void parameter(Matrix3<Var<T>> &v) {
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      parameter(v(i, j));
    }
  }
}
template <class T>
void output(Matrix3<Var<T>> &v) {
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      output(v(i, j));
    }
  }
}
template <class T>
void goal(const Matrix3<Var<T>> &v) {
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      goal(v(i, j));
    }
  }
}

template <class T>
auto &operator<<(std::ostream &stream, const Matrix3<T> &v) {
  stream << "[";
  for (size_t row = 0; row < 3; row++) {
    stream << "[";
    for (size_t col = 0; col < 3; col++) {
      stream << v(row, col);
      if (col < 2) {
        stream << ",";
      }
    }
    stream << "]";
    if (row < 2) {
      stream << ",";
    }
  }
  stream << "]";
  return stream;
}

template <class T>
Vector3<T> operator*(const Matrix3<T> &a, const Vector3<T> &b) {
  Vector3<T> ret = Vector3<T>::Zero();
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      ret[row] += a(row, col) * b[col];
    }
  }
  return ret;
}

template <class T>
Matrix3<T> operator+(const Matrix3<T> &a, const Matrix3<T> &b) {
  Matrix3<T> ret;
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      ret(row, col) = a(row, col) + b(row, col);
    }
  }
  return ret;
}

template <class T>
Matrix3<T> operator-(const Matrix3<T> &a, const Matrix3<T> &b) {
  Matrix3<T> ret;
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      ret(row, col) = a(row, col) - b(row, col);
    }
  }
  return ret;
}

typedef Matrix3<double> Mat3d;
typedef Matrix3<float> Mat3f;

}  // namespace tractor
