// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/error.h>
#include <tractor/core/operator.h>
#include <tractor/geometry/quaternion.h>
#include <tractor/geometry/vector3.h>
#include <tractor/geometry/vector3_ops.h>

namespace tractor {

TRACTOR_OP_T(quat, zero, (Quaternion<T> & x), { x.setZero(); })
TRACTOR_D_T(prepare, quat, zero, (const Quaternion<T> &x), {})
TRACTOR_D_T(forward, quat, zero, (Quaternion<T> & dx), { dx.setZero(); })
TRACTOR_D_T(reverse, quat, zero, (const Quaternion<T> &dx), {})

TRACTOR_OP_T(quat, move, (const Quaternion<T> &v), { return Quaternion<T>(v); })
TRACTOR_D_T(prepare, quat, move,
            (const Quaternion<T> &a, const Quaternion<T> &x), {})
TRACTOR_D_T(forward, quat, move, (const Vector3<T> &da, Vector3<T> &dx),
            { dx = da; })
TRACTOR_D_T(reverse, quat, move, (Vector3<T> & da, const Vector3<T> &dx),
            { da = dx; })

TRACTOR_GRADIENT_TYPE_TEMPLATE(Quaternion<T>, Vector3<T>);

TRACTOR_OP_T(quat_vec3, mul, (const Quaternion<T> &a, const Vector3<T> &b),
             { return a * b; })
TRACTOR_D_T(prepare, quat_vec3, mul,
            (const Quaternion<T> &a, const Vector3<T> &b, const Vector3<T> &x,
             Quaternion<T> &va, Vector3<T> &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, quat_vec3, mul,
            (const Quaternion<T> &va, const Vector3<T> &vb,
             const Vector3<T> &da, const Vector3<T> &db, Vector3<T> &dx),
            { dx = va * db + cross(da, va * vb); })
TRACTOR_D_T(reverse, quat_vec3, mul,
            (const Quaternion<T> &va, const Vector3<T> &vb, Vector3<T> &da,
             Vector3<T> &db, const Vector3<T> &dx),
            {
              da = cross(va * vb, dx);
              db = va.inverse() * dx;
            })

TRACTOR_OP_T(quat, mul, (const Quaternion<T> &a, const Quaternion<T> &b),
             { return a * b; })
TRACTOR_D_T(prepare, quat, mul,
            (const Quaternion<T> &a, const Quaternion<T> &b,
             const Quaternion<T> &x, Quaternion<T> &va),
            { va = a; })
TRACTOR_D_T(forward, quat, mul,
            (const Quaternion<T> &va, const Vector3<T> &da,
             const Vector3<T> &db, Vector3<T> &dx),
            { dx = da + va * db; })
TRACTOR_D_T(reverse, quat, mul,
            (const Quaternion<T> &va, Vector3<T> &da, Vector3<T> &db,
             const Vector3<T> &dx),
            {
              da = dx;
              db = va.inverse() * dx;
            })

template <class T>
auto quat_inverse(const Quaternion<T> &a) {
  return a.inverse();
}
TRACTOR_OP(quat_inverse, (const Quaternion<T> &a), { return quat_inverse(a); })
TRACTOR_D(prepare, quat_inverse,
          (const Quaternion<T> &a, const Quaternion<T> &x, Quaternion<T> &va),
          { va = a; })
TRACTOR_D(forward, quat_inverse,
          (const Quaternion<T> &va, const Vector3<T> &a, Vector3<T> &x),
          { x = va.inverse() * -a; })
TRACTOR_D(reverse, quat_inverse,
          (const Quaternion<T> &va, Vector3<T> &a, const Vector3<T> &x),
          { a = va * -x; })

TRACTOR_OP(quat_unpack, (const Quaternion<T> &q, T &x, T &y, T &z, T &w),
           { quat_unpack(q, x, y, z, w); })
TRACTOR_D(prepare, quat_unpack,
          (const Quaternion<T> &a, const T &x, const T &y, const T &z,
           const T &w, Quaternion<T> &va),
          { va = a; })
TRACTOR_D(forward, quat_unpack,
          (const Quaternion<T> &va, const Vector3<T> &da, T &dx, T &dy, T &dz,
           T &dw),
          {
            T va_x = va.x();
            T va_y = va.y();
            T va_z = va.z();
            T va_w = va.w();

            T qda_x = da.x() * T(0.5);
            T qda_y = da.y() * T(0.5);
            T qda_z = da.z() * T(0.5);

            dx = +qda_x * va_w + qda_y * va_z - qda_z * va_y;
            dy = -qda_x * va_z + qda_y * va_w + qda_z * va_x;
            dz = +qda_x * va_y - qda_y * va_x + qda_z * va_w;
            dw = -qda_x * va_x - qda_y * va_y - qda_z * va_z;
          })
TRACTOR_D(reverse, quat_unpack,
          (const Quaternion<T> &va, Vector3<T> &da, const T &dx, const T &dy,
           const T &dz, const T &dw),
          {
            T va_x = va.x();
            T va_y = va.y();
            T va_z = va.z();
            T va_w = va.w();

            T qda_x = +dx * va_w - dy * va_z + dz * va_y - dw * va_x;
            T qda_y = +dx * va_z + dy * va_w - dz * va_x - dw * va_y;
            T qda_z = -dx * va_y + dy * va_x + dz * va_w - dw * va_z;

            da.x() = qda_x * T(0.5);
            da.y() = qda_y * T(0.5);
            da.z() = qda_z * T(0.5);
          })

TRACTOR_OP(quat_pack,
           (const T &a, const T &b, const T &c, const T &d, Quaternion<T> &x),
           { quat_pack(a, b, c, d, x); })
TRACTOR_D(prepare, quat_pack,
          (const T &a, const T &b, const T &c, const T &d,
           const Quaternion<T> &x, Quaternion<T> &v_quat, T &v_norm_inv),
          {
            v_quat = Quaternion<T>(a, b, c, d);
            v_norm_inv = T(1) / norm(v_quat);
          })
TRACTOR_D(forward, quat_pack,
          (const Quaternion<T> &v_quat, const T &v_norm_inv, const T &da,
           const T &db, const T &dc, const T &dd, Vector3<T> &dx),
          {
            dx = quat_pack_forward(v_quat, v_norm_inv,
                                   Quaternion<T>(da, db, dc, dd));
          })
TRACTOR_D(reverse, quat_pack,
          (const Quaternion<T> &v_quat, const T &v_norm_inv, T &da, T &db,
           T &dc, T &dd, const Vector3<T> &dx),
          {
            Quaternion d = quat_pack_reverse(v_quat, v_norm_inv, dx);
            da = d.x();
            db = d.y();
            dc = d.z();
            dd = d.w();
          })

template <class T>
T quat_residual_gradient(const T &x) {
  typedef typename BatchScalar<T>::Type S;
  T y;
  makeBatchLoop([](const S &x, S &y) {
    if (x < S(0)) {
      y = quat_residual_gradient(-x);
    } else {
      if (x < S(1)) {
        S r = S(1) - x * x;
        y = S(2) * x * acos(x) / (r * sqrt(r)) - S(2) / (S(1) - x * x);
      } else {
        y = S(-2) / S(3);
      }
    }
    if (!std::isfinite(y)) {
      TRACTOR_FATAL(x);
      TRACTOR_FATAL(y);
      throw std::runtime_error("quat residual gradient not finite");
    }
  }).run(x, y);
  return y;
}

template <class T>
auto quat_residual_factor(const T &x) ->
    typename std::enable_if<!IsVar<T>::value, T>::type {
  typedef typename BatchScalar<T>::Type S;
  T y;
  makeBatchLoop([](const S &x, S &y) {
    if (x < S(0)) {
      y = -quat_residual_factor(-x);
    } else {
      if (x < S(1)) {
        y = S(2) * acos(x) / sqrt(S(1) - x * x);
      } else {
        y = S(2);
      }
    }
    if (!std::isfinite(y)) {
      TRACTOR_FATAL(x);
      TRACTOR_FATAL(y);
      throw std::runtime_error("quat residual gradient not finite");
    }
  }).run(x, y);
  return y;
}

TRACTOR_OP(quat_residual_factor, (const T &x),
           { return quat_residual_factor(x); })
TRACTOR_D(prepare, quat_residual_factor, (const T &a, const T &x, T &p),
          { p = quat_residual_gradient(a); })
TRACTOR_D(forward, quat_residual_factor, (const T &p, const T &da, T &dx),
          { dx = da * p; })
TRACTOR_D(reverse, quat_residual_factor, (const T &p, T &da, const T &dx),
          { da = dx * p; })

template <class T>
struct QuatResidualLinearization {
  T vec_f;
  T d_vec_f;
  Quaternion<T> va;
};

template <class T>
Vector3<T> quat_residual(const Quaternion<T> &quat) {
  T vec_f = quat_residual_factor(quat.w());
  T vec_x = quat.x() * vec_f;
  T vec_y = quat.y() * vec_f;
  T vec_z = quat.z() * vec_f;
  return Vector3<T>(vec_x, vec_y, vec_z);
}

TRACTOR_OP(quat_residual, (const Quaternion<T> &a),
           { return quat_residual(a); })
TRACTOR_D(prepare, quat_residual,
          (const Quaternion<T> &va, const Vector3<T> &vx,
           QuatResidualLinearization<T> &v),
          {
            v.vec_f = quat_residual_factor(va.w());
            v.d_vec_f = quat_residual_gradient(va.w());
            v.va = va;
          })
TRACTOR_D(forward, quat_residual,
          (const QuatResidualLinearization<T> &v, const Vector3<T> &da,
           Vector3<T> &dx),
          {
            auto &vec_f = v.vec_f;
            auto &d_vec_f = v.d_vec_f;
            auto &va = v.va;

            Quaternion<T> dqda = Quaternion<T>(da.x() * T(0.5), da.y() * T(0.5),
                                               da.z() * T(0.5), T(0));

            auto dqa = dqda * va;

            T d_vec_f_w = d_vec_f * dqa.w();

            T d_vec_x = dqa.x() * vec_f + va.x() * d_vec_f_w;
            T d_vec_y = dqa.y() * vec_f + va.y() * d_vec_f_w;
            T d_vec_z = dqa.z() * vec_f + va.z() * d_vec_f_w;

            dx = Vector3<T>(d_vec_x, d_vec_y, d_vec_z);
          })
TRACTOR_D(reverse, quat_residual,
          (const QuatResidualLinearization<T> &v,  //
           Vector3<T> &da, const Vector3<T> &dx),
          {
            auto &vec_f = v.vec_f;
            auto &d_vec_f = v.d_vec_f;
            auto &va = v.va;

            T d_vec_x = dx.x();
            T d_vec_y = dx.y();
            T d_vec_z = dx.z();

            T d_vec_f_w =
                va.x() * d_vec_x + va.y() * d_vec_y + va.z() * d_vec_z;

            Quaternion<T> dqa;
            dqa.x() = d_vec_x * vec_f;
            dqa.y() = d_vec_y * vec_f;
            dqa.z() = d_vec_z * vec_f;
            dqa.w() = d_vec_f_w * d_vec_f;

            Quaternion<T> dqda = dqa * va.inverse();

            da.x() = dqda.x() * T(0.5);
            da.y() = dqda.y() * T(0.5);
            da.z() = dqda.z() * T(0.5);
          })

template <class T>
struct AngleAxisQuatLinerization {
  Vector3<T> axis_normalized;
  T sin_angle_by_axis_length;
  T cos_angle_minus_one_by_axis_length;
};
TRACTOR_OP(angle_axis_quat, (const T &angle, const Vector3<T> &axis),
           { return angle_axis_quat(angle, axis); })
TRACTOR_D(prepare, angle_axis_quat,
          (const T &angle, const Vector3<T> &axis, const Quaternion<T> &rot,
           AngleAxisQuatLinerization<T> &v),
          {
            v.axis_normalized = normalized(axis);
            v.sin_angle_by_axis_length = T(sin(angle)) / norm(axis);
            v.cos_angle_minus_one_by_axis_length =
                (T(cos(angle)) - T(1)) / norm(axis);
          })
TRACTOR_D(forward, angle_axis_quat,
          (const AngleAxisQuatLinerization<T> &v,  //
           const T &d_angle, const Vector3<T> &d_axis, Vector3<T> &d_rot),
          {
            Vector3<T> d_axis_p =
                (d_axis - v.axis_normalized * dot(v.axis_normalized, d_axis));
            d_rot = v.axis_normalized * d_angle              //
                    + d_axis_p * v.sin_angle_by_axis_length  //
                    + cross(d_axis_p, v.axis_normalized) *
                          v.cos_angle_minus_one_by_axis_length;
          })
TRACTOR_D(reverse, angle_axis_quat,
          (const AngleAxisQuatLinerization<T> &v,  //
           T &d_angle, Vector3<T> &d_axis, const Vector3<T> &d_rot),
          {
            Vector3<T> d_rot_p =
                (d_rot - v.axis_normalized * dot(v.axis_normalized, d_rot));

            d_angle = dot(v.axis_normalized, d_rot);

            d_axis = d_rot_p * v.sin_angle_by_axis_length +
                     cross(v.axis_normalized, d_rot_p) *
                         v.cos_angle_minus_one_by_axis_length;
          })

template <class T>
auto sinc(const T &x) -> typename std::enable_if<!IsVar<T>::value, T>::type {
  typedef typename BatchScalar<T>::Type S;
  T y;
  makeBatchLoop([](const S &x, S &y) {
    if (x != S(0)) {
      y = S(sin(x)) / x;
    } else {
      y = S(1);
    }
  }).run(x, y);
  return y;
}

template <class T>
auto sinc_gradient(const T &x) ->
    typename std::enable_if<!IsVar<T>::value, T>::type {
  typedef typename BatchScalar<T>::Type S;
  T y;
  makeBatchLoop([](const S &x, S &y) {
    if (x != S(0)) {
      y = (x * cos(x) - sin(x)) / (x * x);
    } else {
      y = S(0);
    }
  }).run(x, y);
  return y;
}

template <class T>
auto quat_vec_add_gradient(const T &x) ->
    typename std::enable_if<!IsVar<T>::value, T>::type {
  typedef typename BatchScalar<T>::Type S;
  T y;
  makeBatchLoop([](const S &x, S &y) {
    if (x != S(0)) {
      y = sinc_gradient(x * S(0.5)) / x * S(0.25);
    } else {
      y = S(1) / S(3) * S(0.25);
    }
  }).run(x, y);
  return y;
}

TRACTOR_OP(sinc, (const T &a), { return sinc(a); })
TRACTOR_D(prepare, sinc, (const T &a, const T &x, T &p),
          { p = sinc_gradient(a); })
TRACTOR_D(forward, sinc, (const T &p, const T &da, T &dx), { dx = da * p; })
TRACTOR_D(reverse, sinc, (const T &p, T &da, const T &dx), { da = dx * p; })

template <class T>
Quaternion<T> operator+(const Quaternion<T> &a, const Vector3<T> &b) {
  T angle = norm(b);
  T f = sinc(angle * T(0.5)) * T(0.5);
  T c = cos(angle * T(0.5));
  Quaternion<T> quat;
  quat.x() = b.x() * f;
  quat.y() = b.y() * f;
  quat.z() = b.z() * f;
  quat.w() = c;
  return quat * a;
}
template <class T>
Quaternion<T> &operator+=(Quaternion<T> &a, const Vector3<T> &b) {
  a = a + b;
  return a;
}
TRACTOR_OP_T(quat_vec3, add, (const Quaternion<T> &a, const Vector3<T> &b),
             { return a + b; })
template <class T>
struct QuatVec3AddLinearization {
  Vector3<T> b;
  Quaternion<T> quat;
  T sgradn;
  T f;
};
TRACTOR_D_T(prepare, quat_vec3, add,
            (const Quaternion<T> &a, const Vector3<T> &b,
             const Quaternion<T> &x, QuatVec3AddLinearization<T> &v),
            {
              T angle = norm(b);
              T f = sinc(angle * T(0.5)) * T(0.5);
              T c = cos(angle * T(0.5));
              Quaternion<T> quat;
              quat.x() = b.x() * f;
              quat.y() = b.y() * f;
              quat.z() = b.z() * f;
              quat.w() = c;
              v.b = b;
              v.quat = quat;
              v.sgradn = quat_vec_add_gradient(angle);
              v.f = f;
            })
TRACTOR_D_T(forward, quat_vec3, add,
            (const QuatVec3AddLinearization<T> &v, const Vector3<T> &da,
             const Vector3<T> &db, Vector3<T> &dx),
            {
              T d_angle = dot(v.b, db);

              T d_f = d_angle * v.sgradn;
              T d_c = d_angle * v.f * T(-0.5);

              Quaternion<T> d_quat;
              d_quat.x() = v.b.x() * d_f + db.x() * v.f;
              d_quat.y() = v.b.y() * d_f + db.y() * v.f;
              d_quat.z() = v.b.z() * d_f + db.z() * v.f;
              d_quat.w() = d_c;

              Vector3 d_vec = quat_pack_forward(v.quat, T(1), d_quat);

              dx = d_vec + v.quat * da;
            })
TRACTOR_D_T(reverse, quat_vec3, add,
            (const QuatVec3AddLinearization<T> &v, Vector3<T> &da,
             Vector3<T> &db, const Vector3<T> &dx),
            {
              da = v.quat.inverse() * dx;

              Quaternion<T> d_quat = quat_pack_reverse(v.quat, T(1), dx);

              db.x() = d_quat.x() * v.f;
              db.y() = d_quat.y() * v.f;
              db.z() = d_quat.z() * v.f;

              T d_f = v.b.x() * d_quat.x() + v.b.y() * d_quat.y() +
                      v.b.z() * d_quat.z();

              T d_c = d_quat.w();

              T d_angle = d_f * v.sgradn + d_c * v.f * T(-0.5);

              db += v.b * d_angle;
            })

}  // namespace tractor
