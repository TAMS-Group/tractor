// 2020-2024 Philipp Ruppel

#pragma once

#include <cmath>
#include <iostream>

#include <tractor/core/batch.h>
#include <tractor/core/ops.h>

namespace tractor {

template <class T>
class Var;

template <class Scalar>
class Vector3 {
  Scalar _data[3];

 public:
  inline Vector3() {
    _data[0] = Scalar(0);
    _data[1] = Scalar(0);
    _data[2] = Scalar(0);
  }
  inline Vector3(const Scalar &x, const Scalar &y, const Scalar &z) {
    _data[0] = x;
    _data[1] = y;
    _data[2] = z;
  }
  template <class T, class R = decltype(Scalar(std::declval<T>()))>
  explicit Vector3(const Vector3<T> &other) {
    x() = Scalar(other.x());
    y() = Scalar(other.y());
    z() = Scalar(other.z());
  }
  inline auto &x() const { return _data[0]; }
  inline auto &y() const { return _data[1]; }
  inline auto &z() const { return _data[2]; }
  inline auto &x() { return _data[0]; }
  inline auto &y() { return _data[1]; }
  inline auto &z() { return _data[2]; }
  inline void setZero() {
    _data[0] = Scalar(0);
    _data[1] = Scalar(0);
    _data[2] = Scalar(0);
  }
  inline auto &operator+=(const Vector3<Scalar> &other) {
    _data[0] += other._data[0];
    _data[1] += other._data[1];
    _data[2] += other._data[2];
    return *this;
  }
  inline auto &operator-=(const Vector3<Scalar> &other) {
    _data[0] -= other._data[0];
    _data[1] -= other._data[1];
    _data[2] -= other._data[2];
    return *this;
  }
  template <class S>
  inline auto &operator*=(const S &scale) {
    _data[0] *= Scalar(scale);
    _data[1] *= Scalar(scale);
    _data[2] *= Scalar(scale);
    return *this;
  }
  template <class S>
  inline auto &operator/=(const S &divisor) {
    Scalar scale = Scalar(1) / Scalar(divisor);
    _data[0] *= scale;
    _data[1] *= scale;
    _data[2] *= scale;
    return *this;
  }
  static inline Vector3 Zero() {
    return Vector3(Scalar(0), Scalar(0), Scalar(0));
  }
  inline auto &operator[](size_t i) const { return _data[i]; }
  inline auto &operator[](size_t i) { return _data[i]; }
  inline const Scalar *data() const { return _data; }
  inline Scalar *data() { return _data; }
};

template <class T>
void variable(Vector3<Var<T>> &v) {
  variable(v.x());
  variable(v.y());
  variable(v.z());
}
template <class T>
void parameter(Vector3<Var<T>> &v) {
  parameter(v.x());
  parameter(v.y());
  parameter(v.z());
}
template <class T>
void output(Vector3<Var<T>> &v) {
  output(v.x());
  output(v.y());
  output(v.z());
}
template <class T>
void goal(const Vector3<Var<T>> &v) {
  goal(v.x());
  goal(v.y());
  goal(v.z());
}

template <class T, size_t S>
inline Vector3<T> indexBatch(const Vector3<Batch<T, S>> &v, size_t i) {
  return Vector3<T>(v.x()[i], v.y()[i], v.z()[i]);
}

template <class T>
auto &operator<<(std::ostream &stream, const Vector3<T> &v) {
  return stream << "[" << v.x() << "," << v.y() << "," << v.z() << "]";
}

template <class T>
inline Vector3<T> operator-(const Vector3<T> &v) {
  return Vector3<T>(-v.x(), -v.y(), -v.z());
}
template <class T>
inline Vector3<T> operator+(const Vector3<T> &v) {
  return v;
}

template <class T>
inline Vector3<T> operator+(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(a.x() + b.x(), a.y() + b.y(), a.z() + b.z());
}

template <class T>
inline Vector3<T> operator-(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(a.x() - b.x(), a.y() - b.y(), a.z() - b.z());
}

template <class T>
inline Vector3<T> operator*(const Vector3<T> &a, const T &b) {
  return Vector3<T>(a.x() * b, a.y() * b, a.z() * b);
}

template <class T>
inline Vector3<T> operator*(const T &a, const Vector3<T> &b) {
  return Vector3<T>(a * b.x(), a * b.y(), a * b.z());
}

template <class T>
T inline dot(const Vector3<T> &a, const Vector3<T> &b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}

template <class T>
inline Vector3<T> cross(const Vector3<T> &a, const Vector3<T> &b) {
  return Vector3<T>(a.y() * b.z() - a.z() * b.y(),  //
                    a.z() * b.x() - a.x() * b.z(),  //
                    a.x() * b.y() - a.y() * b.x()   //
  );
}

template <class T>
inline void vec3_unpack(const Vector3<T> &v, T &x, T &y, T &z) {
  x = v.x();
  y = v.y();
  z = v.z();
}

template <class T>
inline void vec3_pack(const T &x, const T &y, const T &z, Vector3<T> &vec) {
  vec = Vector3<T>(x, y, z);
}

template <class T>
inline auto squaredNorm(const Vector3<T> &v) {
  return T(dot(v, v));
}

template <class T>
inline auto norm(const Vector3<T> &v) {
  return T(sqrt(dot(v, v)));
}

template <class T>
inline auto normalized(const Vector3<T> &v) {
  return v * (T(1) / norm(v));
}

typedef Vector3<double> Vec3d;
typedef Vector3<float> Vec3f;

}  // namespace tractor
