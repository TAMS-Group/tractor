// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/batch.h>
#include <tractor/geometry/vector3.h>

namespace tractor {

template <class Scalar> class Quaternion {
  Scalar _x = Scalar(), _y = Scalar(), _z = Scalar(), _w = Scalar();

public:
  inline Quaternion() {}
  inline Quaternion(const Scalar &x, const Scalar &y, const Scalar &z,
                    const Scalar &w)
      : _x(x), _y(y), _z(z), _w(w) {}
  template <class T, class R = decltype(Scalar(std::declval<T>()))>
  explicit Quaternion(const Quaternion<T> &other) {
    x() = Scalar(other.x());
    y() = Scalar(other.y());
    z() = Scalar(other.z());
    w() = Scalar(other.w());
  }
  inline auto &x() const { return _x; }
  inline auto &y() const { return _y; }
  inline auto &z() const { return _z; }
  inline auto &w() const { return _w; }
  inline auto &x() { return _x; }
  inline auto &y() { return _y; }
  inline auto &z() { return _z; }
  inline auto &w() { return _w; }
  inline void setIdentity() {
    _x = Scalar(0);
    _y = Scalar(0);
    _z = Scalar(0);
    _w = Scalar(1);
  }
  inline void setZero() {
    _x = Scalar(0);
    _y = Scalar(0);
    _z = Scalar(0);
    _w = Scalar(0);
  }
  inline auto inverse() const {
    return Quaternion<Scalar>(-x(), -y(), -z(), w());
  }
  inline auto vec() const { return Vector3<Scalar>(_x, _y, _z); }
  static inline Quaternion Identity() {
    return Quaternion(Scalar(0), Scalar(0), Scalar(0), Scalar(1));
  }
  // std::string str() const {
  //   std::stringstream ss;
  //   ss << "Quaternion(" << _x << "," << _y << "," << _z << "," << _w << ")";
  //   return ss.str();
  // }
};

template <class T> void variable(Quaternion<Var<T>> &v) {
  variable(v.x());
  variable(v.y());
  variable(v.z());
  variable(v.w());
}
template <class T> void parameter(Quaternion<Var<T>> &v) {
  parameter(v.x());
  parameter(v.y());
  parameter(v.z());
  parameter(v.w());
}
template <class T> void output(Quaternion<Var<T>> &v) {
  output(v.x());
  output(v.y());
  output(v.z());
  output(v.w());
}
template <class T> void goal(const Quaternion<Var<T>> &v) {
  goal(v.x());
  goal(v.y());
  goal(v.z());
  goal(v.w());
}

template <class T, size_t S>
inline Quaternion<T> indexBatch(const Quaternion<Batch<T, S>> &v, size_t i) {
  return Quaternion<T>(v.x()[i], v.y()[i], v.z()[i], v.w()[i]);
}

template <class T>
auto &operator<<(std::ostream &stream, const Quaternion<T> &v) {
  return stream << "[" << v.x() << "," << v.y() << "," << v.z() << "," << v.w()
                << "]";
  // return stream << "Quaternion(" << v.x() << " " << v.y() << " " << v.z() <<
  // " "
  //               << v.w() << ")";
}

template <class T> T norm(const Quaternion<T> &q) {
  return sqrt(q.x() * q.x() + q.y() * q.y() + q.z() * q.z() + q.w() * q.w());
}

template <class T> Quaternion<T> normalized(const Quaternion<T> &q) {
  // T f = T(1) / norm(q);
  T n = norm(q);
  // T f = ((n > T(0)) ? (T(1) / n) : T(1));
  // T f = T(1) / std::max(T(1e-9), n);
  T f = T(1) / n;
  return Quaternion<T>(q.x() * f, q.y() * f, q.z() * f, q.w() * f);
}

template <class T>
inline Vector3<T> operator*(const Quaternion<T> &q, const Vector3<T> &v) {

  T v_x = v.x();
  T v_y = v.y();
  T v_z = v.z();

  T q_x = q.x();
  T q_y = q.y();
  T q_z = q.z();
  T q_w = q.w();

  T t_x = q_y * v_z - q_z * v_y;
  T t_y = q_z * v_x - q_x * v_z;
  T t_z = q_x * v_y - q_y * v_x;

  T r_x = q_w * t_x + q_y * t_z - q_z * t_y;
  T r_y = q_w * t_y + q_z * t_x - q_x * t_z;
  T r_z = q_w * t_z + q_x * t_y - q_y * t_x;

  r_x += r_x;
  r_y += r_y;
  r_z += r_z;

  r_x += v_x;
  r_y += v_y;
  r_z += v_z;

  return Vector3<T>(r_x, r_y, r_z);

  // auto r = q * Quaternion<T>(v.x(), v.y(), v.z(), 0) * q.inverse();
  // return Vector3<T>(r.x(), r.y(), r.z());
}

template <class T>
inline Quaternion<T> operator*(const Quaternion<T> &p, const Quaternion<T> &q) {

  T p_x = p.x();
  T p_y = p.y();
  T p_z = p.z();
  T p_w = p.w();

  T q_x = q.x();
  T q_y = q.y();
  T q_z = q.z();
  T q_w = q.w();

  T r_x = (p_w * q_x + p_x * q_w) + (p_y * q_z - p_z * q_y);
  T r_y = (p_w * q_y - p_x * q_z) + (p_y * q_w + p_z * q_x);
  T r_z = (p_w * q_z + p_x * q_y) - (p_y * q_x - p_z * q_w);
  T r_w = (p_w * q_w - p_x * q_x) - (p_y * q_y + p_z * q_z);

  return Quaternion<T>(r_x, r_y, r_z, r_w);
}

template <class T>
void quat_unpack(const Quaternion<T> &q, T &x, T &y, T &z, T &w) {
  x = q.x();
  y = q.y();
  z = q.z();
  w = q.w();
}

// template <class T>
// void quat_pack(const T &x, const T &y, const T &z, const T &w,
//                Quaternion<T> &vec) {
//   vec = Quaternion<T>(x, y, z, w);
// }

template <class T>
void quat_pack(const T &x, const T &y, const T &z, const T &w,
               Quaternion<T> &vec) {
  T f = T(1) / sqrt(x * x + y * y + z * z + w * w);
  vec = Quaternion<T>(x * f, y * f, z * f, w * f);
}

template <class T> Vector3<T> quat_residual(const Quaternion<T> &quat) {

  // Quaternion<T> quat_n = normalized(quat);
  //
  // T axis_temp = T(1) / sqrt(T(1) - quat_n.w() * quat_n.w());
  // T axis_x = quat_n.x() * axis_temp;
  // T axis_y = quat_n.y() * axis_temp;
  // T axis_z = quat_n.z() * axis_temp;
  //
  // T angle = T(2) * acos(quat_n.w());
  //
  // return Vector3<T>(axis_x * angle, axis_y * angle, axis_z * angle);

  T vec_f = T(2) * acos(quat.w()) / sqrt(T(1) - quat.w() * quat.w());

  T vec_x = quat.x() * vec_f;
  T vec_y = quat.y() * vec_f;
  T vec_z = quat.z() * vec_f;

  return Vector3<T>(vec_x, vec_y, vec_z);
}

// template <class T> Vector3<T> quat_residual(const Quaternion<T> &a) {
//   return a.vec() * T(a.w() < 0 ? -2 : 2);
// }
//
// template <class T, size_t S>
// Vector3<Batch<T, S>> quat_residual(const Quaternion<Batch<T, S>> &a) {
//   Batch<T, S> f;
//   for (size_t i = 0; i < S; i++) {
//     f[i] = T(a.w()[i] < 0 ? -2 : 2);
//   }
//   return a.vec() * f;
// }

template <class T>
Quaternion<T> angle_axis_quat(const T &angle, const Vector3<T> &axis) {
  Vector3<T> axis_n = normalized(axis);
  Quaternion<T> quat;
  T s = sin(angle * T(0.5));
  T c = cos(angle * T(0.5));
  quat.x() = axis_n.x() * s;
  quat.y() = axis_n.y() * s;
  quat.z() = axis_n.z() * s;
  quat.w() = c;
  // return normalized(quat);
  return quat;
}

/*

template <class T>
inline Vector3<T> quat_pack_forward(const Quaternion<T> &v,
                                    const Quaternion<T> &d) {

  // see quat pack op in geometry/ops.h

  T v_x = v.x();
  T v_y = v.y();
  T v_z = v.z();
  T v_w = v.w();

  T da = d.x();
  T db = d.y();
  T dc = d.z();
  T dd = d.w();

  T r_x = -dd * v_x + da * v_w - db * v_z + dc * v_y;
  T r_y = -dd * v_y + da * v_z + db * v_w - dc * v_x;
  T r_z = -dd * v_z - da * v_y + db * v_x + dc * v_w;

  Vector3<T> dx;
  dx.x() = r_x * T(2);
  dx.y() = r_y * T(2);
  dx.z() = r_z * T(2);
  return dx;
}

template <class T>
inline Quaternion<T> quat_pack_reverse(const Quaternion<T> &v,
                                       const Vector3<T> &dx) {

  // see quat pack op in geometry/ops.h

  T v_x = v.x();
  T v_y = v.y();
  T v_z = v.z();
  T v_w = v.w();

  T r_x = dx.x() * T(2);
  T r_y = dx.y() * T(2);
  T r_z = dx.z() * T(2);

  T da = +r_x * v_w + r_y * v_z - r_z * v_y;
  T db = -r_x * v_z + r_y * v_w + r_z * v_x;
  T dc = +r_x * v_y - r_y * v_x + r_z * v_w;
  T dd = -r_x * v_x - r_y * v_y - r_z * v_z;

  return Quaternion<T>(da, db, dc, dd);
}

*/

template <class T> T dot(const Quaternion<T> &a, const Quaternion<T> &b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

// template <class T> T norm(const Quaternion<T> &q) { return sqrt(dot(q, q)); }

template <class T> Quaternion<T> operator*(const Quaternion<T> &q, const T &f) {
  return Quaternion<T>(q.x() * f, q.y() * f, q.z() * f, q.w() * f);
}

template <class T> Quaternion<T> operator/(const Quaternion<T> &q, const T &f) {
  return q * (T(1) / f);
}

template <class T>
Quaternion<T> operator+(const Quaternion<T> &a, const Quaternion<T> &b) {
  return Quaternion<T>(a.x() + b.x(), a.y() + b.y(), a.z() + b.z(),
                       a.w() + b.w());
}

template <class T>
Quaternion<T> operator-(const Quaternion<T> &a, const Quaternion<T> &b) {
  return Quaternion<T>(a.x() - b.x(), a.y() - b.y(), a.z() - b.z(),
                       a.w() - b.w());
}

template <class T>
inline Vector3<T> quat_pack_forward(const Quaternion<T> &v, const T &v_norm_inv,
                                    const Quaternion<T> &d) {

  // see quat pack op in geometry/ops.h

  // auto v_n = normalized(v);
  auto v_n = v * v_norm_inv;

  // auto d_n = d / norm(v);

  auto d_p = (d - v_n * dot(v_n, d)) / norm(v);

  T v_x = v_n.x();
  T v_y = v_n.y();
  T v_z = v_n.z();
  T v_w = v_n.w();

  T da = d_p.x();
  T db = d_p.y();
  T dc = d_p.z();
  T dd = d_p.w();

  T r_x = -dd * v_x + da * v_w - db * v_z + dc * v_y;
  T r_y = -dd * v_y + da * v_z + db * v_w - dc * v_x;
  T r_z = -dd * v_z - da * v_y + db * v_x + dc * v_w;

  Vector3<T> dx;
  dx.x() = r_x * T(2);
  dx.y() = r_y * T(2);
  dx.z() = r_z * T(2);
  return dx;
}

template <class T>
inline Quaternion<T> quat_pack_reverse(const Quaternion<T> &v,
                                       const T &v_norm_inv,
                                       const Vector3<T> &dx) {

  // see quat pack op in geometry/ops.h

  T v_x = v.x() * v_norm_inv;
  T v_y = v.y() * v_norm_inv;
  T v_z = v.z() * v_norm_inv;
  T v_w = v.w() * v_norm_inv;

  T r_x = dx.x() * T(2) * v_norm_inv;
  T r_y = dx.y() * T(2) * v_norm_inv;
  T r_z = dx.z() * T(2) * v_norm_inv;

  T da = +r_x * v_w + r_y * v_z - r_z * v_y;
  T db = -r_x * v_z + r_y * v_w + r_z * v_x;
  T dc = +r_x * v_y - r_y * v_x + r_z * v_w;
  T dd = -r_x * v_x - r_y * v_y - r_z * v_z;

  return Quaternion<T>(da, db, dc, dd);
}

// template <class T>
// Quaternion<T> angle_axis_quat(const T &angle, const Vector3<T> &axis) {
//   Quaternion<T> quat;
//   T s = sin(angle * T(0.5));
//   T c = cos(angle * T(0.5));
//   quat.x() = axis.x() * s;
//   quat.y() = axis.y() * s;
//   quat.z() = axis.z() * s;
//   quat.w() = c;
//   return quat;
// }

typedef Quaternion<double> Quatd;
typedef Quaternion<float> Quatf;

typedef Quaternion<double> Quat3d;
typedef Quaternion<float> Quat3f;

} // namespace tractor
