// (c) 2020-2022 Philipp Ruppel

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

//         da.x   da.y   da.z
// -----+-----------------
// dx.x |   0     vb.z  -vb.y
// dx.y | -vb.z    0     vb.x
// dx.z |  vb.y  -vb.x    0
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
            {
              // dx = Quaternion<T>(da.x(), da.y(), da.z(), 0.0) * vb + va * db;
              // dx = va * db + cross(da, vb);
              dx = va * db + cross(da, va * vb);
              /*
              return Vector3<T>(a.y() * b.z() - a.z() * b.y(), //
                              a.z() * b.x() - a.x() * b.z(), //
                              a.x() * b.y() - a.y() * b.x()  //
                              */
            })
TRACTOR_D_T(reverse, quat_vec3, mul,
            (const Quaternion<T> &va, const Vector3<T> &vb, Vector3<T> &da,
             Vector3<T> &db, const Vector3<T> &dx),
            {
              da = cross(va * vb, dx);
              db = va.inverse() * dx;
            })

/*
TRACTOR_OP_T(quat, mul, (const Quaternion<T> &a, const Quaternion<T> &b),
             { return a * b; })
TRACTOR_D_T(prepare, quat, mul,
            (const Quaternion<T> &a, const Quaternion<T> &b, const Quaternion<T>
&x, Quaternion<T> &va, Quaternion<T> &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, quat, mul,
            (const Quaternion<T> &va, const Quaternion<T> &vb, const Vector3<T>
&da, const Vector3<T> &db, Vector3<T> &dx),
            {
              // dx = (va * Quaternion<T>(db.x(), db.y(), db.z(), 1.0)).vec() +
              //       (Quaternion<T>(da.x(), da.y(), da.z(), 1.0) * vb).vec();
              dx = va * db + da;
            })
TRACTOR_D_T(reverse, quat, mul,
            (const Quaternion<T> &va, const Quaternion<T> &vb, Vector3<T> &da,
Vector3<T> &db, const Vector3<T> &dx),
            {
              // da = (Quaternion<T>(dx.x(), dx.y(), dx.z(), 1.0) *
              // vb.inverse()).vec(); db = (va.inverse() * Quaternion<T>(dx.x(),
              // dx.y(), dx.z(), 1.0)).vec();
              da = dx;
              db = va.inverse() * dx;
            })
*/

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

template <class T> auto quat_inverse(const Quaternion<T> &a) {
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

// -------------------------------------------------------------------------

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
            // Quaternion<T> qda(da.x() * T(0.5), da.y() * T(0.5), da.z() *
            // T(0.5),
            //                   T(1.0));
            // auto r = qda * va;
            // dx = r.x() - va.x();
            // dy = r.y() - va.y();
            // dz = r.z() - va.z();
            // dw = r.w() - va.w();

            T va_x = va.x();
            T va_y = va.y();
            T va_z = va.z();
            T va_w = va.w();

            T qda_x = da.x() * T(0.5);
            T qda_y = da.y() * T(0.5);
            T qda_z = da.z() * T(0.5);
            // T qda_w = T(1.0);

            // T r_x = (qda_w * va_x + qda_x * va_w) + (qda_y * va_z - qda_z *
            // va_y); T r_y = (qda_w * va_y - qda_x * va_z) + (qda_y * va_w +
            // qda_z * va_x); T r_z = (qda_w * va_z + qda_x * va_y) - (qda_y *
            // va_x - qda_z * va_w); T r_w = (qda_w * va_w - qda_x * va_x) -
            // (qda_y * va_y + qda_z * va_z);

            // T r_x = qda_w * va_x + qda_x * va_w + qda_y * va_z - qda_z *
            // va_y; T r_y = qda_w * va_y - qda_x * va_z + qda_y * va_w + qda_z
            // * va_x; T r_z = qda_w * va_z + qda_x * va_y - qda_y * va_x +
            // qda_z * va_w; T r_w = qda_w * va_w - qda_x * va_x - qda_y * va_y
            // - qda_z * va_z;
            //
            // dx = r_x - va.x();
            // dy = r_y - va.y();
            // dz = r_z - va.z();
            // dw = r_w - va.w();

            dx = +qda_x * va_w + qda_y * va_z - qda_z * va_y;
            dy = -qda_x * va_z + qda_y * va_w + qda_z * va_x;
            dz = +qda_x * va_y - qda_y * va_x + qda_z * va_w;
            dw = -qda_x * va_x - qda_y * va_y - qda_z * va_z;

            // dxyzw = qda * va - va
          })
TRACTOR_D(reverse, quat_unpack,
          (const Quaternion<T> &va, Vector3<T> &da, const T &dx, const T &dy,
           const T &dz, const T &dw),
          {
            // Quaternion<T> qda = va * Quaternion<T>(dx, dy, dz, dw).inverse();

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

            // Quaternion<T> r;
            // r.x() = dx + va.x();
            // r.y() = dy + va.y();
            // r.z() = dz + va.z();
            // r.w() = dw + va.w();
            //
            // Quaternion<T> qda = va.inverse() * r;
            //
            // da.x() = qda.x() * T(0.5);
            // da.y() = qda.y() * T(0.5);
            // da.z() = qda.z() * T(0.5);

            // Quaternion<T> r;
            // r.x() = dx - va.x();
            // r.y() = dy - va.y();
            // r.z() = dz - va.z();
            // r.w() = dw - va.w();
            //
            // Quaternion<T> qda = r * va;
            //
            // da.x() = qda.x() * T(0.5);
            // da.y() = qda.y() * T(0.5);
            // da.z() = qda.z() * T(0.5);

            // Quaternion<T> r;
            // r.x() = dx + va.x();
            // r.y() = dy + va.y();
            // r.z() = dz + va.z();
            // r.w() = dw + va.w();
            //
            // // r = qda * va;
            // // qda^-1 * r = va
            // // qda^-1 = va * r^-1
            // // qda = (va * r^-1)^-1
            // // qda = r * va^-1
            //
            // // Quaternion<T> qda = va.inverse() * r;
            // // Quaternion<T> qda = r * va.inverse();
            // Quaternion<T> qda = r;
            //
            // da.x() = qda.x() * T(2);
            // da.y() = qda.y() * T(2);
            // da.z() = qda.z() * T(2);
          })

// -------------------------------------------------------------------------

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
            // Quaternion<T> r =
            //     v.inverse() *
            //     Quaternion<T>(v.x() + da, v.y() + db, v.z() + dc, v.w() +
            //     dd);

            // Quaternion<T> r =
            //     Quaternion<T>(v.x() + da, v.y() + db, v.z() + dc, v.w() + dd)
            //     * v.inverse();
            // dx.x() = r.x() * T(2);
            // dx.y() = r.y() * T(2);
            // dx.z() = r.z() * T(2);

            // auto vv = v * v.inverse();
            // auto dv = Quaternion<T>(da, db, dc, dd) * v.inverse();
            //
            // Quaternion<T> r;
            // r.x() = vv.x() + dv.x();
            // r.y() = vv.y() + dv.y();
            // r.z() = vv.z() + dv.z();
            // r.w() = vv.w() + dv.w();
            //
            // dx.x() = r.x() * T(2);
            // dx.y() = r.y() * T(2);
            // dx.z() = r.z() * T(2);

            // T v_x = v.x();
            // T v_y = v.y();
            // T v_z = v.z();
            // T v_w = v.w();

            // T p_x = v_x + da;
            // T p_y = v_y + db;
            // T p_z = v_z + dc;
            // T p_w = v_w + dd;
            //
            // T q_x = -v_x;
            // T q_y = -v_y;
            // T q_z = -v_z;
            // T q_w = v_w;
            //
            // T r_x = p_w * q_x + p_x * q_w + p_y * q_z - p_z * q_y;
            // T r_y = p_w * q_y - p_x * q_z + p_y * q_w + p_z * q_x;
            // T r_z = p_w * q_z + p_x * q_y - p_y * q_x + p_z * q_w;
            // T r_w = p_w * q_w - p_x * q_x - p_y * q_y - p_z * q_z;
            //
            // dx.x() = r_x * T(2);
            // dx.y() = r_y * T(2);
            // dx.z() = r_z * T(2);

            // clang-format off

            // // ---
            //
            // T r_x = (v_w + dd) * -v_x + (v_x + da) * v_w + (v_y + db) * -v_z - (v_z + dc) * -v_y;
            // T r_y = (v_w + dd) * -v_y - (v_x + da) * -v_z + (v_y + db) * v_w + (v_z + dc) * -v_x;
            // T r_z = (v_w + dd) * -v_z + (v_x + da) * -v_y - (v_y + db) * -v_x + (v_z + dc) * v_w;
            // T r_w = (v_w + dd) * v_w - (v_x + da) * -v_x - (v_y + db) * -v_y - (v_z + dc) * -v_z;

            // ---

            // T r_x = (v_w * -v_x + dd * -v_x) + (v_x * v_w + da * v_w) + (v_y * -v_z + db * -v_z) - (v_z * -v_y + dc * -v_y);
            // T r_y = (v_w * -v_y + dd * -v_y) - (v_x * -v_z + da * -v_z) + (v_y * v_w + db * v_w) + (v_z * -v_x + dc * -v_x);
            // T r_z = (v_w * -v_z + dd * -v_z) + (v_x * -v_y + da * -v_y) - (v_y * -v_x + db * -v_x) + (v_z * v_w + dc * v_w);
            // T r_w = (v_w * v_w + dd * v_w) - (v_x * -v_x + da * -v_x) - (v_y * -v_y + db * -v_y) - (v_z * -v_z + dc * -v_z);

            // ---

            // T r_x = v_w * -v_x + dd * -v_x + v_x * v_w + da * v_w + v_y * -v_z + db * -v_z - v_z * -v_y - dc * -v_y;
            // T r_y = v_w * -v_y + dd * -v_y - v_x * -v_z - da * -v_z + v_y * v_w + db * v_w + v_z * -v_x + dc * -v_x;
            // T r_z = v_w * -v_z + dd * -v_z + v_x * -v_y + da * -v_y - v_y * -v_x - db * -v_x + v_z * v_w + dc * v_w;
            // T r_w = v_w * v_w + dd * v_w - v_x * -v_x - da * -v_x - v_y * -v_y - db * -v_y - v_z * -v_z - dc * -v_z;

            // ---

            // T r_x = v_w * -v_x + dd * -v_x + v_x * +v_w + da * +v_w + v_y * -v_z + db * -v_z - v_z * -v_y - dc * -v_y;
            // T r_y = v_w * -v_y + dd * -v_y - v_x * -v_z - da * -v_z + v_y * +v_w + db * +v_w + v_z * -v_x + dc * -v_x;
            // T r_z = v_w * -v_z + dd * -v_z + v_x * -v_y + da * -v_y - v_y * -v_x - db * -v_x + v_z * +v_w + dc * +v_w;
            // T r_w = v_w * +v_w + dd * +v_w - v_x * -v_x - da * -v_x - v_y * -v_y - db * -v_y - v_z * -v_z - dc * -v_z;

            // T r_x = + dd * -v_x  + da * +v_w + v_y * -v_z + db * -v_z - v_z * -v_y - dc * -v_y;
            // T r_y = + dd * -v_y - v_x * -v_z - da * -v_z + db * +v_w + v_z * -v_x + dc * -v_x;
            // T r_z = + dd * -v_z + v_x * -v_y + da * -v_y - v_y * -v_x - db * -v_x + dc * +v_w;
            // T r_w = v_w * +v_w + dd * +v_w - v_x * -v_x - da * -v_x - v_y * -v_y - db * -v_y - v_z * -v_z - dc * -v_z;

            // T r_x = +dd * -v_x + da * +v_w + db * -v_z - dc * -v_y;
            // T r_y = +dd * -v_y - da * -v_z + db * +v_w + dc * -v_x;
            // T r_z = +dd * -v_z + da * -v_y - db * -v_x + dc * +v_w;

            // clang-format on

            // T v_x = v.x();
            // T v_y = v.y();
            // T v_z = v.z();
            // T v_w = v.w();
            //
            // T r_x = -dd * v_x + da * v_w - db * v_z + dc * v_y;
            // T r_y = -dd * v_y + da * v_z + db * v_w - dc * v_x;
            // T r_z = -dd * v_z - da * v_y + db * v_x + dc * v_w;
            //
            // dx.x() = r_x * T(2);
            // dx.y() = r_y * T(2);
            // dx.z() = r_z * T(2);

            dx = quat_pack_forward(v_quat, v_norm_inv,
                                   Quaternion<T>(da, db, dc, dd));
          })
TRACTOR_D(reverse, quat_pack,
          (const Quaternion<T> &v_quat, const T &v_norm_inv, T &da, T &db,
           T &dc, T &dd, const Vector3<T> &dx),
          {
            // T v_x = v.x();
            // T v_y = v.y();
            // T v_z = v.z();
            // T v_w = v.w();
            //
            // T r_x = dx.x() * T(2);
            // T r_y = dx.y() * T(2);
            // T r_z = dx.z() * T(2);
            //
            // da = +r_x * v_w + r_y * v_z - r_z * v_y;
            // db = -r_x * v_z + r_y * v_w + r_z * v_x;
            // dc = +r_x * v_y - r_y * v_x + r_z * v_w;
            // dd = -r_x * v_x - r_y * v_y - r_z * v_z;

            Quaternion d = quat_pack_reverse(v_quat, v_norm_inv, dx);
            da = d.x();
            db = d.y();
            dc = d.z();
            dd = d.w();
          })

// -------------------------------------------------------------------------

template <class T> T quat_residual_gradient(const T &x) {

  // typedef typename BatchScalar<T>::Type S;
  // T y;
  // makeBatchLoop([](const S &x, S &y) {
  //   if (x >= S(1)) {
  //     y = S(-2) / S(3);
  //   } else {
  //     S r = S(1) - x * x;
  //     y = S(2) * x * acos(x) / (r * sqrt(r)) - S(2) / (S(1) - x * x);
  //   }
  // }).run(x, y);
  // return y;

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
  }).run(x, y);
  return y;

  // T r = T(1) - x * x;
  // T y = T(2) * x * acos(x) / (r * sqrt(r)) - T(2) / (T(1) - x * x);
  // return y;
}

template <class T>
auto quat_residual_factor(const T &x) ->
    typename std::enable_if<!IsVar<T>::value, T>::type {

  // typedef typename BatchScalar<T>::Type S;
  // T y;
  // makeBatchLoop([](const S &x, S &y) {
  //   if (x >= S(1)) {
  //     y = S(2);
  //   } else {
  //     y = S(2) * acos(x) / sqrt(S(1) - x * x);
  //   }
  // }).run(x, y);
  // return y;

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
  }).run(x, y);
  return y;

  // return x;

  // return T(2) * acos(a) / sqrt(T(1) - a * a);
}

TRACTOR_OP(quat_residual_factor, (const T &x), {
  return quat_residual_factor(x);

  // typedef typename BatchScalar<T>::Type S;
  // T y;
  // makeBatchLoop([](const S &x, S &y) {
  //   if (x >= S(1)) {
  //     y = S(-2) / S(3);
  //   } else {
  //     y = S(2) * acos(x) / sqrt(S(1) - x * x);
  //   }
  // }).run(x, y);
  // return y;

  // return x;
})
TRACTOR_D(prepare, quat_residual_factor, (const T &a, const T &x, T &p),
          { p = quat_residual_gradient(a); })
TRACTOR_D(forward, quat_residual_factor, (const T &p, const T &da, T &dx),
          { dx = da * p; })
TRACTOR_D(reverse, quat_residual_factor, (const T &p, T &da, const T &dx),
          { da = dx * p; })

// -------------------------------------------------------------------------

// template <class T>
// Quaternion<T>

// -------------------------------------------------------------------------

// TRACTOR_OP(quat_residual, (const Quaternion<T> &a), { return a.vec() *
// T(2);
// })

// TRACTOR_OP(quat_residual, (const Quaternion<T> &a),
//            { return quat_residual(a); })
// TRACTOR_D(prepare, quat_residual, (const Quaternion<T> &a, const
// Vector3<T> &x),
//           {})
// TRACTOR_D(forward, quat_residual, (const Vector3<T> &a, Vector3<T> &x),
//           { x = a; })
// TRACTOR_D(reverse, quat_residual, (Vector3<T> & a, const Vector3<T> &x),
//           { a = x; })

template <class T> struct QuatResidualLinearization {
  T vec_f;
  T d_vec_f;
  Quaternion<T> va;
};

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

  // T vec_f = T(2) * acos(quat.w()) / sqrt(T(1) - quat.w() * quat.w());
  T vec_f = quat_residual_factor(quat.w());

  // Quaternion<T> quat_n = normalized(quat);
  //
  // T w = quat_n.w() * T(0.999999);
  // T vec_f = T(2) * acos(w) / sqrt(T(1) - w * w);

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
            // v.vec_f = T(2) * acos(va.w()) / sqrt(T(1) - va.w() * va.w());
            //
            // T d_va_w_r = T(1) - va.w() * va.w();
            // v.d_vec_f =
            //     T(2) * va.w() * acos(va.w()) / (d_va_w_r * sqrt(d_va_w_r)) -
            //     T(2) / (T(1) - va.w() * va.w());

            v.vec_f = quat_residual_factor(va.w());
            v.d_vec_f = quat_residual_gradient(va.w());

            v.va = va;
          })
TRACTOR_D(forward, quat_residual,
          (
              // const Quaternion<T> &va, const Vector3<T> &vx,
              const QuatResidualLinearization<T> &v, //
              const Vector3<T> &da, Vector3<T> &dx),
          {
            // T d_va_w_r = T(1) - va.w() * va.w();
            // T d_va_w =
            //     T(2) * va.w() * acos(va.w()) / (d_va_w_r * sqrt(d_va_w_r)) -
            //     T(2) / (T(1) - va.w() * va.w());
            //
            // T vec_f = T(2) * acos(quat.w()) / sqrt(T(1) - quat.w() *
            // quat.w());
            //
            // T vec_x = quat.x() * vec_f;
            // T vec_y = quat.y() * vec_f;
            // T vec_z = quat.z() * vec_f;
            //
            // return Vector3<T>(vec_x, vec_y, vec_z);

            // x = va * a;

            // auto vqa = va;

            // T vec_f = T(2) * acos(va.w()) / sqrt(T(1) - va.w() * va.w());
            // T d_va_w_r = T(1) - va.w() * va.w();
            // auto d_vec_f =
            //     T(2) * va.w() * acos(va.w()) / (d_va_w_r * sqrt(d_va_w_r)) -
            //     T(2) / (T(1) - va.w() * va.w());

            //

            auto &vec_f = v.vec_f;
            auto &d_vec_f = v.d_vec_f;
            auto &va = v.va;

            //

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
          (
              // const Quaternion<T> &va, const Vector3<T> &vx,
              const QuatResidualLinearization<T> &v, //
              Vector3<T> &da, const Vector3<T> &dx),
          {
            // da = dx;

            // T vec_f = T(2) * acos(va.w()) / sqrt(T(1) - va.w() * va.w());
            // T d_va_w_r = T(1) - va.w() * va.w();
            // auto d_vec_f =
            //     T(2) * va.w() * acos(va.w()) / (d_va_w_r * sqrt(d_va_w_r)) -
            //     T(2) / (T(1) - va.w() * va.w());

            //

            auto &vec_f = v.vec_f;
            auto &d_vec_f = v.d_vec_f;
            auto &va = v.va;

            //

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

// -------------------------------------------------------------------------

template <class T> struct AngleAxisQuatLinerization {
  Vector3<T> axis_normalized;
  T sin_angle_by_axis_length;
  T cos_angle_minus_one_by_axis_length;
};
TRACTOR_OP(angle_axis_quat, (const T &angle, const Vector3<T> &axis),
           { return angle_axis_quat(angle, axis); })
TRACTOR_D(prepare, angle_axis_quat,
          (const T &angle, const Vector3<T> &axis, const Quaternion<T> &rot,
           // T &v_angle, Vector3<T> &v_axis
           AngleAxisQuatLinerization<T> &v),
          {
            // v_angle = angle;
            // v_axis = axis;
            v.axis_normalized = normalized(axis);
            v.sin_angle_by_axis_length = T(sin(angle)) / norm(axis);
            v.cos_angle_minus_one_by_axis_length =
                (T(cos(angle)) - T(1)) / norm(axis);
          })
TRACTOR_D(forward, angle_axis_quat,
          (
              // const T &v_angle, const Vector3<T> &v_axis,
              const AngleAxisQuatLinerization<T> &v, //
              const T &d_angle, const Vector3<T> &d_axis, Vector3<T> &d_rot),
          {
            Vector3<T> d_axis_p =
                (d_axis - v.axis_normalized * dot(v.axis_normalized, d_axis));
            d_rot = v.axis_normalized * d_angle             //
                    + d_axis_p * v.sin_angle_by_axis_length //
                    + cross(d_axis_p, v.axis_normalized) *
                          v.cos_angle_minus_one_by_axis_length;

            // T v_axis_f = T(1) / norm(v_axis);
            // Vector3<T> v_axis_n = v_axis * v_axis_f;
            // Vector3<T> d_axis_n =
            //     (d_axis - v_axis_n * dot(v_axis_n, d_axis)) * v_axis_f;
            //
            // d_rot = v_axis_n * d_angle           //
            //         + d_axis_n * T(sin(v_angle)) //
            //         + cross(d_axis_n, v_axis_n) * (T(cos(v_angle)) - T(1));

            // Vector3<T> v_axis_n = v_axis;
            // Vector3<T> d_axis_n = d_axis - v_axis_n * dot(v_axis_n, d_axis);
            //
            // // clang-format off
            // d_rot = v_axis * d_angle
            //       + d_axis_n * T(sin(v_angle))
            //       + cross(d_axis, v_axis) * (T(cos(v_angle)) - T(1));
            // // clang-format on

            // Vector3<T> v_axis_n = v_axis;
            // Vector3<T> d_axis_n = d_axis - v_axis_n * dot(v_axis_n, d_axis);

            // // clang-format off
            // d_rot = v_axis_n * d_angle
            //       + d_axis_n * T(sin(v_angle))
            //       + cross(d_axis_n, v_axis_n) * (T(cos(v_angle)) - T(1));
            // // clang-format on

            // T v_s = sin(v_angle * T(0.5));
            // T v_c = cos(v_angle * T(0.5));

            // // clang-format off
            // d_rot = v_axis * d_angle
            //       + d_axis_n * v_c * v_s * T(2)
            //       - cross(d_axis_n, v_axis_n) * v_s * v_s * T(2);
            // // clang-format on

            // Vector3<T> rv = d_axis_n * v_c - cross(d_axis_n, v_axis_n) * v_s;
            // d_rot = v_axis * d_angle + rv * v_s * T(2);

            // T vqx = v_axis_n.x() * v_s;
            // T vqy = v_axis_n.y() * v_s;
            // T vqz = v_axis_n.z() * v_s;
            // T vqw = v_c;
            //
            // T dqx = d_axis_n.x();
            // T dqy = d_axis_n.y();
            // T dqz = d_axis_n.z();
            //
            // T r_x = +dqx * vqw - (dqy * vqz - dqz * vqy);
            // T r_y = +dqy * vqw - (dqz * vqx - dqx * vqz);
            // T r_z = +dqz * vqw - (dqx * vqy - dqy * vqx);
            //
            // d_rot = v_axis * d_angle + Vector3<T>(r_x, r_y, r_z) * v_s *
            // T(2);

            // Quaternion<T> v_quat;
            // v_quat.x() = v_axis_n.x() * v_s;
            // v_quat.y() = v_axis_n.y() * v_s;
            // v_quat.z() = v_axis_n.z() * v_s;
            // v_quat.w() = v_c;
            //
            // Quaternion<T> d_quat;
            // d_quat.x() = d_axis_n.x();
            // d_quat.y() = d_axis_n.y();
            // d_quat.z() = d_axis_n.z();
            // d_quat.w() = T(0);
            //
            // d_rot = v_axis * d_angle + quat_pack_forward(v_quat, d_quat) *
            // v_s;

            /*
            // clang-format off

            T vs = sin(v_angle * T(0.5));
            T vc = cos(v_angle * T(0.5));

            T ds = d_angle * +vc * T(0.5);
            T dc = d_angle * -vs * T(0.5);

            T r_x =
                    -v_axis_n.x() * dc * vs
                    +v_axis_n.x() * ds * vc
                    -v_axis_n.z() * v_axis_n.y() * ds * vs
                    +v_axis_n.y() * v_axis_n.z() * ds * vs

                    +d_axis_n.x() * vs * vc
                    -d_axis_n.y() * v_axis_n.z() * vs * vs
                    +d_axis_n.z() * v_axis_n.y() * vs * vs
                    ;

            T r_y =
                    -v_axis_n.y() * dc * vs
                    +v_axis_n.y() * ds * vc
                    +v_axis_n.x() * ds * v_axis_n.z() * vs
                    -v_axis_n.z() * ds * v_axis_n.x() * vs

                    +d_axis_n.y() * vs * vc
                    +d_axis_n.x() * vs * v_axis_n.z() * vs
                    -d_axis_n.z() * vs * v_axis_n.x() * vs
                    ;

            T r_z =
                    -v_axis_n.z() * dc * vs
                    +v_axis_n.z() * ds * vc
                    -v_axis_n.x() * ds * v_axis_n.y() * vs
                    +v_axis_n.y() * ds * v_axis_n.x() * vs

                    +d_axis_n.z() * vs * vc
                    -d_axis_n.x() * vs * v_axis_n.y() * vs
                    +d_axis_n.y() * vs * v_axis_n.x() * vs
                    ;

            d_rot.x() = r_x * T(2);
            d_rot.y() = r_y * T(2);
            d_rot.z() = r_z * T(2);

            // clang-format on
            */

            /*
            // clang-format off

            T vs = sin(v_angle * T(0.5));
            T vc = cos(v_angle * T(0.5));

            T ds = d_angle * +vc * T(0.5);
            T dc = d_angle * -vs * T(0.5);

            T r_x =
                    -v_axis_n.x() * dc * vs
                    +v_axis_n.x() * ds * vc
                    +d_axis_n.x() * vs * vc

                    -v_axis_n.y() * ds * v_axis_n.z() * vs
                    -d_axis_n.y() * vs * v_axis_n.z() * vs
                    +v_axis_n.z() * ds * v_axis_n.y() * vs
                    +d_axis_n.z() * vs * v_axis_n.y() * vs
                    ;

            T r_y =
                    -v_axis_n.y() * dc * vs
                    +v_axis_n.y() * ds * vc
                    +d_axis_n.y() * vs * vc

                    +v_axis_n.x() * ds * v_axis_n.z() * vs
                    +d_axis_n.x() * vs * v_axis_n.z() * vs
                    -v_axis_n.z() * ds * v_axis_n.x() * vs
                    -d_axis_n.z() * vs * v_axis_n.x() * vs
                    ;

            T r_z =
                    -v_axis_n.z() * dc * vs
                    +v_axis_n.z() * ds * vc
                    +d_axis_n.z() * vs * vc

                    -v_axis_n.x() * ds * v_axis_n.y() * vs
                    -d_axis_n.x() * vs * v_axis_n.y() * vs
                    +v_axis_n.y() * ds * v_axis_n.x() * vs
                    +d_axis_n.y() * vs * v_axis_n.x() * vs
                    ;

            d_rot.x() = r_x * T(2);
            d_rot.y() = r_y * T(2);
            d_rot.z() = r_z * T(2);

            // clang-format on
            */

            /*
            Vector3<T> v_axis_n = normalized(v_axis);
            Vector3<T> d_axis_n = d_axis - v_axis_n * dot(v_axis_n, d_axis);

            T vs = sin(v_angle * T(0.5));
            T vc = cos(v_angle * T(0.5));

            T ds = d_angle * +vc * T(0.5);
            T dc = d_angle * -vs * T(0.5);

            T r_x = -dc * v_axis_n.x() * vs + v_axis_n.x() * ds * vc +
                    d_axis_n.x() * vs * vc -
                    v_axis_n.y() * ds * v_axis_n.z() * vs -
                    d_axis_n.y() * vs * v_axis_n.z() * vs +
                    v_axis_n.z() * ds * v_axis_n.y() * vs +
                    d_axis_n.z() * vs * v_axis_n.y() * vs;

            T r_y = -dc * v_axis_n.y() * vs +
                    v_axis_n.x() * ds * v_axis_n.z() * vs +
                    d_axis_n.x() * vs * v_axis_n.z() * vs +
                    v_axis_n.y() * ds * vc + d_axis_n.y() * vs * vc -
                    v_axis_n.z() * ds * v_axis_n.x() * vs -
                    d_axis_n.z() * vs * v_axis_n.x() * vs;

            T r_z = -dc * v_axis_n.z() * vs -
                    v_axis_n.x() * ds * v_axis_n.y() * vs -
                    d_axis_n.x() * vs * v_axis_n.y() * vs +
                    v_axis_n.y() * ds * v_axis_n.x() * vs +
                    d_axis_n.y() * vs * v_axis_n.x() * vs +
                    v_axis_n.z() * ds * vc + d_axis_n.z() * vs * vc;

            d_rot.x() = r_x * T(2);
            d_rot.y() = r_y * T(2);
            d_rot.z() = r_z * T(2);
            */

            /*
            Vector3<T> v_axis_n = normalized(v_axis);
            Vector3<T> d_axis_n = d_axis - v_axis_n * dot(v_axis_n, d_axis);

            T vs = sin(v_angle * T(0.5));
            T vc = cos(v_angle * T(0.5));

            T ds = d_angle * +vc * T(0.5);
            T dc = d_angle * -vs * T(0.5);

            T qvx = v_axis_n.x() * vs;
            T qvy = v_axis_n.y() * vs;
            T qvz = v_axis_n.z() * vs;
            T qvw = vc;

            T qdx = v_axis_n.x() * ds + d_axis_n.x() * vs;
            T qdy = v_axis_n.y() * ds + d_axis_n.y() * vs;
            T qdz = v_axis_n.z() * ds + d_axis_n.z() * vs;
            T qdw = dc;

            T r_x =
                -dc * (v_axis_n.x() * vs) +
                (v_axis_n.x() * ds + d_axis_n.x() * vs) * vc -
                (v_axis_n.y() * ds + d_axis_n.y() * vs) * (v_axis_n.z() * vs) +
                (v_axis_n.z() * ds + d_axis_n.z() * vs) * (v_axis_n.y() * vs);

            T r_y =
                -dc * (v_axis_n.y() * vs) +
                (v_axis_n.x() * ds + d_axis_n.x() * vs) * (v_axis_n.z() * vs) +
                (v_axis_n.y() * ds + d_axis_n.y() * vs) * qvw -
                (v_axis_n.z() * ds + d_axis_n.z() * vs) * (v_axis_n.x() * vs);

            T r_z =
                -dc * (v_axis_n.z() * vs) -
                (v_axis_n.x() * ds + d_axis_n.x() * vs) * (v_axis_n.y() * vs) +
                (v_axis_n.y() * ds + d_axis_n.y() * vs) * (v_axis_n.x() * vs) +
                (v_axis_n.z() * ds + d_axis_n.z() * vs) * vc;

            d_rot.x() = r_x * T(2);
            d_rot.y() = r_y * T(2);
            d_rot.z() = r_z * T(2);
            */

            /*
            Vector3<T> v_axis_n = normalized(v_axis);
            Vector3<T> d_axis_n = d_axis - v_axis_n * dot(v_axis_n, d_axis);

            T vs = sin(v_angle * T(0.5));
            T vc = cos(v_angle * T(0.5));

            T ds = d_angle * +vc * T(0.5);
            T dc = d_angle * -vs * T(0.5);

            T qvx = v_axis_n.x() * vs;
            T qvy = v_axis_n.y() * vs;
            T qvz = v_axis_n.z() * vs;
            T qvw = vc;

            T qdx = v_axis_n.x() * ds + d_axis_n.x() * vs;
            T qdy = v_axis_n.y() * ds + d_axis_n.y() * vs;
            T qdz = v_axis_n.z() * ds + d_axis_n.z() * vs;
            T qdw = dc;

            T r_x = -qdw * qvx + qdx * qvw - qdy * qvz + qdz * qvy;
            T r_y = -qdw * qvy + qdx * qvz + qdy * qvw - qdz * qvx;
            T r_z = -qdw * qvz - qdx * qvy + qdy * qvx + qdz * qvw;

            d_rot.x() = r_x * T(2);
            d_rot.y() = r_y * T(2);
            d_rot.z() = r_z * T(2);
            */

            /*
            Vector3<T> v_axis_n = normalized(v_axis);
            Vector3<T> d_axis_n = d_axis - v_axis_n * dot(v_axis_n, d_axis);

            T v_s = sin(v_angle * T(0.5));
            T v_c = cos(v_angle * T(0.5));

            T d_s = d_angle * +v_c * T(0.5);
            T d_c = d_angle * -v_s * T(0.5);

            Quaternion<T> v_quat;
            v_quat.x() = v_axis_n.x() * v_s;
            v_quat.y() = v_axis_n.y() * v_s;
            v_quat.z() = v_axis_n.z() * v_s;
            v_quat.w() = v_c;

            Quaternion<T> d_quat;
            d_quat.x() = v_axis_n.x() * d_s + d_axis_n.x() * v_s;
            d_quat.y() = v_axis_n.y() * d_s + d_axis_n.y() * v_s;
            d_quat.z() = v_axis_n.z() * d_s + d_axis_n.z() * v_s;
            d_quat.w() = d_c;

            d_rot = quat_pack_forward(v_quat, d_quat);
            */

            /*
            // d_rot = v_axis * d_angle + d_axis * v_angle;

            // d_rot = v_axis * d_angle +
            //         (d_axis - v_axis * dot(d_axis, v_axis)) * v_angle;

            // d_rot = v_axis * d_angle +
            //         angle_axis_quat(v_angle, v_axis).inverse() *
            //             (d_axis - v_axis * dot(d_axis, v_axis)) * v_angle;

            d_rot = v_axis * d_angle;
            T s = sin(v_angle * T(0.5)) * T(2.0);

            // T c = cos(v_angle * T(0.5));
            //  quat.x() = axis_n.x() * s;
            //  quat.y() = axis_n.y() * s;
            //  quat.z() = axis_n.z() * s;

            // d_rot += angle_axis_quat(v_angle, v_axis).inverse() * d_axis * s;
            // d_rot += d_axis * s;
            d_rot += (d_axis - v_axis * dot(d_axis, v_axis)) * s;
            */
          })
TRACTOR_D(reverse, angle_axis_quat,
          (
              // const T &v_angle, const Vector3<T> &v_axis,
              const AngleAxisQuatLinerization<T> &v, //
              T &d_angle, Vector3<T> &d_axis, const Vector3<T> &d_rot),
          {
            // d_angle = d_rot.x() * v_axis.x() + d_rot.y() * v_axis.y() +
            //          d_rot.z() * v_axis.z();

            // d_angle = dot(d_rot, v_axis);
            // d_axis = d_rot * v_angle;

            Vector3<T> d_rot_p =
                (d_rot - v.axis_normalized * dot(v.axis_normalized, d_rot));

            d_angle = dot(v.axis_normalized, d_rot);

            d_axis = d_rot_p * v.sin_angle_by_axis_length +
                     cross(v.axis_normalized, d_rot_p) *
                         v.cos_angle_minus_one_by_axis_length;

            // d_rot = v.axis_normalized * d_angle             //
            //         + d_axis_p * v.sin_angle_by_axis_length //
            //         + cross(d_axis_p, v.axis_normalized) *
            //               v.cos_angle_minus_one_by_axis_length;
          })

// -------------------------------------------------------------------------
//
// template <class T>
// Quaternion<T> operator+(const Quaternion<T> &a, const Vector3<T> &b) {
//
//   // return normalized(normalized(Quaternion<T>(b.x() * T(0.5), b.y() *
//   T(0.5),
//   //                                            b.z() * T(0.5), T(1.0))) *
//   //                   a);
//
//   // return normalized(angle_axis_quat(norm(b), normalized(b)) * a);
//
//   return normalized(normalized(Quaternion<T>(b.x() * T(0.5), b.y() * T(0.5),
//                                              b.z() * T(0.5), T(1))) *
//                     a);
//
//   // Quaternion<T> qb = normalized(
//   //     Quaternion<T>(b.x() * T(0.5), b.y() * T(0.5), b.z() * T(0.5),
//   T(1)));
//   // return normalized(qb * a);
//
//   // Quaternion<T> qb =
//   //     Quaternion<T>(b.x() * T(0.5), b.y() * T(0.5), b.z() * T(0.5), T(1));
//   // return qb * a;
//
//   // T angle = norm(b);
//   // Vector3<T> axis_n = b / angle;
//   //
//   // T s = sin(angle * T(0.5));
//   // T c = cos(angle * T(0.5));
//   //
//   // Quaternion<T> quat;
//   //
//   // quat.x() = axis_n.x() * s;
//   // quat.y() = axis_n.y() * s;
//   // quat.z() = axis_n.z() * s;
//   // quat.w() = c;
//   //
//   // return quat * a;
//
//   // Quaternion<T> b_quat(a * b.x() * T(0.5), //
//   //                      a * b.y() * T(0.5), //
//   //                      a * b.z() * T(0.5), //
//   //                      T(1));
//
//   // T norm = b.x() * b.x() + b.y() * b.y() + b.z() * b.z();
//   // T angle = norm;
//   // Vector3<T> axis = v / norm;
//   // Quaternion<T> quat;
//   // T s = sin(angle * T(0.5));
//   // T c = cos(angle * T(0.5));
//   // quat.x() = axis.x() * s;
//   // quat.y() = axis.y() * s;
//   // quat.z() = axis.z() * s;
//   // quat.w() = c;
//   // return quat;
// }
// template <class T>
// Quaternion<T> &operator+=(Quaternion<T> &a, const Vector3<T> &b) {
//   a = a + b;
//   return a;
// }
// TRACTOR_OP_T(quat_vec3, add, (const Quaternion<T> &a, const Vector3<T> &b),
//              { return a + b; })
// // TRACTOR_D_T(prepare, quat_vec3, add,
// //             (const Quaternion<T> &a, const Vector3<T> &b,
// //              const Quaternion<T> &x, Quaternion<T> &va, Vector3<T> &vb),
// //             {
// //               va = a;
// //               vb = b;
// //               // T angle = norm(b);
// //               // Vector3<T> axis = normalized(b);
// //               // vb.axis_normalized = normalized(axis);
// //               // vb.sin_angle_by_axis_length = T(sin(angle)) / norm(axis);
// //               // vb.cos_angle_minus_one_by_axis_length =
// //               //     (T(cos(angle)) - T(1)) / norm(axis);
// //             })
// template <class T> struct QuatVec3AddLinearization {
//   Quaternion<T> bqn;
//   T bqfh;
// };
// TRACTOR_D_T(prepare, quat_vec3, add,
//             (const Quaternion<T> &a, const Vector3<T> &b,
//              const Quaternion<T> &x, QuatVec3AddLinearization<T> &v),
//             {
//               Quaternion<T> bq = Quaternion<T>(b.x() * T(0.5), //
//                                                b.y() * T(0.5), //
//                                                b.z() * T(0.5), //
//                                                T(1)            //
//               );
//               T bqf = T(1) / norm(bq);
//               v.bqn = normalized(bq);
//               v.bqfh = bqf * T(0.5);
//             })
// TRACTOR_D_T(forward, quat_vec3, add,
//             (
//                 // const Quaternion<T> &va, const Vector3<T> &vb,
//                 // const Quaternion<T> &vx,
//                 const QuatVec3AddLinearization<T> &v, const Vector3<T> &da,
//                 const Vector3<T> &db, Vector3<T> &dx),
//             {
//               Vector3 dqb = quat_pack_forward(v.bqn, T(1),
//                                               Quaternion<T>(       //
//                                                   db.x() * v.bqfh, //
//                                                   db.y() * v.bqfh, //
//                                                   db.z() * v.bqfh, //
//                                                   T(0)             //
//                                                   ));
//               dx = dqb + v.bqn * da;
//
//               // Quaternion<T> vbq = Quaternion<T>(vb.x() * T(0.5), //
//               //                                   vb.y() * T(0.5), //
//               //                                   vb.z() * T(0.5), //
//               //                                   T(1)             //
//               // );
//               // T vbqf = T(1) / norm(vbq);
//               // Quaternion<T> vbqn = normalized(vbq);
//               // Vector3 dqb = quat_pack_forward(vbqn,
//               //                                 Quaternion<T>( //
//               //                                     db.x() * T(0.5) * vbqf,
//               //
//               //                                     db.y() * T(0.5) * vbqf,
//               //
//               //                                     db.z() * T(0.5) * vbqf,
//               //
//               //                                     T(0) //
//               //                                     ));
//               // dx = dqb + vbqn * da;
//
//               // T f = T(1) / norm(Quaternion<T>(vb.x() * T(0.5), //
//               //                                 vb.y() * T(0.5), //
//               //                                 vb.z() * T(0.5), //
//               //                                 T(1)             //
//               //                                 ));
//               // Quaternion<T> vbq = normalized(Quaternion<T>(vb.x() *
//               T(0.5),
//               // //
//               //                                              vb.y() *
//               T(0.5),
//               //                                              // vb.z() *
//               //                                              T(0.5), // T(1)
//               //
//               //                                              ));
//               // Vector3 dqb = quat_pack_forward(
//               //     vbq, Quaternion<T>(db.x() * T(0.5) * f, db.y() * T(0.5)
//               *
//               //     f,
//               //                        db.z() * T(0.5) * f, T(0)));
//               // dx = dqb + vbq * da;
//
//               // Vector3<T> d_axis_p =
//               //     (d_axis - v.axis_normalized * dot(v.axis_normalized,
//               //     d_axis));
//               //
//               // d_rot = v.axis_normalized * d_angle             //
//               //         + d_axis_p * v.sin_angle_by_axis_length //
//               //         + cross(d_axis_p, v.axis_normalized) *
//               //               v.cos_angle_minus_one_by_axis_length;
//
//               // x = a + b;
//               // Quaternion<T> qvb = normalized(Quaternion<T>(
//               //     vb.x() * T(0.5), vb.y() * T(0.5), vb.z() * T(0.5),
//               T(1)));
//               //
//               // // dx = qvb * da + db * va;
//               //
//               // // dx = qvb * da + va.inverse() * db;
//               // // dx = qvb * da + db;
//               //
//               // dx = qvb * da + db * va;
//
//               // static auto q = [](const Vector3<T> &v) {
//               //   return Quaternion<T>(v.x() * T(0.5), v.y() * T(0.5),
//               //                        v.z() * T(0.5), T(1.0));
//               // };
//               //
//               // Quaternion<T> qdx = q(vb + db) * q(da) * va * vx.inverse();
//               //
//               // dx.x() = qdx.x() * T(2.0);
//               // dx.y() = qdx.y() * T(2.0);
//               // dx.z() = qdx.z() * T(2.0);
//             })
// TRACTOR_D_T(reverse, quat_vec3, add,
//             (const QuatVec3AddLinearization<T> &v, Vector3<T> &da,
//              Vector3<T> &db, const Vector3<T> &dx),
//             {
//               // da.setZero();
//               // db.setZero();
//
//               da = v.bqn.inverse() * dx;
//               Quaternion<T> qdb = quat_pack_reverse(v.bqn, T(1), dx *
//               v.bqfh); db.x() = qdb.x(); db.y() = qdb.y(); db.z() = qdb.z();
//
//               // Vector3<T> xdb = dx;
//               // Vector3<T> xda = v.bqn.inverse() * dx;
//               //
//               // Quaternion<T> qdb = quat_pack_reverse(v.bqn,
//               //                                       Vector3<T>( //
//               //                                           xdb.x() * v.bqfh,
//               //
//               //                                           xdb.y() * v.bqfh,
//               //
//               //                                           xdb.z() * v.bqfh
//               //
//               //                                           ));
//               //
//               // db.x() = qdb.x();
//               // db.y() = qdb.y();
//               // db.z() = qdb.z();
//               //
//               // da = xda;
//             })

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
template <class T> struct QuatVec3AddLinearization {
  // Quaternion<T> a;
  Vector3<T> b;
  // Vector3<T> b_n;
  Quaternion<T> quat;
  T sgradn;
  // T msinan;
  T f;
  // T c;
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
              // v.a = a;
              v.b = b;
              // v.b_n = normalized(b);
              v.quat = quat;
              // v.sgrad = sinc_gradient(angle * T(0.5));
              // v.msina = -sin(angle * T(0.5));
              // if (angle == T(0)) {
              //   v.sgradn = T(1) / T(3) * T(0.25);
              // } else {
              //   v.sgradn = sinc_gradient(angle * T(0.5)) / angle * T(0.25);
              // }
              v.sgradn = quat_vec_add_gradient(angle);
              // v.msinan = -sin(angle * T(0.5)) / angle;
              // v.msinan = sinc(angle * T(0.5)) * T(0.5) * T(-0.5);
              v.f = f;
              // v.c = c;
              // TRACTOR_ASSERT(std::isfinite(angle));
              // TRACTOR_ASSERT(std::isfinite(f));
              // TRACTOR_ASSERT(std::isfinite(c));
            })
TRACTOR_D_T(forward, quat_vec3, add,
            (const QuatVec3AddLinearization<T> &v, const Vector3<T> &da,
             const Vector3<T> &db, Vector3<T> &dx),
            {
              T d_angle = dot(v.b, db);

              T d_f = d_angle * v.sgradn;
              // T d_c = d_angle * v.msinan;
              T d_c = d_angle * v.f * T(-0.5);

              // Quaternion<T> v_quat;
              // v_quat.x() = v.b.x() * v.f;
              // v_quat.y() = v.b.y() * v.f;
              // v_quat.z() = v.b.z() * v.f;
              // v_quat.w() = v.c;

              Quaternion<T> d_quat;
              d_quat.x() = v.b.x() * d_f + db.x() * v.f;
              d_quat.y() = v.b.y() * d_f + db.y() * v.f;
              d_quat.z() = v.b.z() * d_f + db.z() * v.f;
              d_quat.w() = d_c;

              Vector3 d_vec = quat_pack_forward(v.quat, T(1), d_quat);

              dx = d_vec + v.quat * da;

              // T v_angle = norm(v.b);
              // T d_angle = dot(normalized(v.b), db);
              //
              // T v_angle_half = v_angle * T(0.5);
              // T d_angle_half = d_angle * T(0.5);
              //
              // T v_f_2 = sinc(v_angle_half);
              // T d_f_2 = d_angle_half * sinc_gradient(v_angle_half);
              //
              // T v_f = v_f_2 * T(0.5);
              // T d_f = d_f_2 * T(0.5);
              //
              // T v_c = cos(v_angle_half);
              // T d_c = d_angle_half * -sin(v_angle_half);
              //
              // Quaternion<T> v_quat;
              // v_quat.x() = v.b.x() * v_f;
              // v_quat.y() = v.b.y() * v_f;
              // v_quat.z() = v.b.z() * v_f;
              // v_quat.w() = v_c;
              //
              // Quaternion<T> d_quat;
              // d_quat.x() = v.b.x() * d_f + db.x() * v_f;
              // d_quat.y() = v.b.y() * d_f + db.y() * v_f;
              // d_quat.z() = v.b.z() * d_f + db.z() * v_f;
              // d_quat.w() = d_c;
              //
              // Vector3 d_vec = quat_pack_forward(v_quat, T(1), d_quat);
              //
              // dx = d_vec + v_quat * da;
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

              // T d_angle = d_f * v.sgradn + d_c * v.msinan;
              T d_angle = d_f * v.sgradn + d_c * v.f * T(-0.5);

              db += v.b * d_angle;
            })

// TRACTOR_OP(vec_to_quat, (const Vector3<T> &a), {
//   T angle = norm(a);
//   Vector3<T> axis = normalized(v);
//   Quaternion<T> quat;
//   T s = sin(angle * T(0.5));
//   T c = cos(angle * T(0.5));
//   quat.x() = axis.x() * s;
//   quat.y() = axis.y() * s;
//   quat.z() = axis.z() * s;
//   quat.w() = c;
//   return quat;
// })
// TRACTOR_D(prepare, vec_to_quat, (const Vector3<T> &a, const Quaternion<T>
// &x),
//           {})
// TRACTOR_D(forward, vec_to_quat, (const Vector3<T> &a, Vector3<T> &x),
//           { x = a + b; })
// TRACTOR_D(reverse, vec_to_quat, (Vector3<T> & a, const Vector3<T> &x), {
//   a = x;
//   b = x;
// })

// template <class T>
// auto quaternion_trust_region_constraint(const Quaternion<T> &a, const T &tr)
// {
//   return T(0);
// }
// TRACTOR_OP(quaternion_trust_region_constraint,
//            (const Quaternion<T> &a, const T &tr), { return T(0); })
// TRACTOR_D(prepare, quaternion_trust_region_constraint,
//           (const Quaternion<T> &a, const T &tr, const T &x), {})
// TRACTOR_D(forward, quaternion_trust_region_constraint,
//           (const Vector3<T> &a, const T &tr, T &x), { x = T(0); })
// TRACTOR_D(reverse, quaternion_trust_region_constraint,
//           (Vector3<T> & a, T &tr, const T &x), {
//             a.setZero();
//             tr = T(0);
//           })
// TRACTOR_D(project, quaternion_trust_region_constraint,
//           (const Quaternion<T> &a, const T &tr, const Vector3<T> &da,
//            const T &padding, Vector3<T> &dx),
//           { dx = da; })
// TRACTOR_D(barrier_init, quaternion_trust_region_constraint,
//           (const Quaternion<T> &a, const T &tr, const Vector3<T> &da,
//            Vector3<T> &dx, Vector3<T> &ddx),
//           {
//             for (size_t i = 0; i < 6; i++) {
//               T p = da[i];
//               T lo2 = -tr;
//               T hi2 = +tr;
//               T u = T(-1) / std::max(T(0), p - lo2);
//               T v = T(+1) / std::max(T(0), hi2 - p);
//               dx[i] = u + v;
//               ddx[i] = (u * u) + (v * v);
//             }
//           })
// TRACTOR_D(barrier_step, quaternion_trust_region_constraint,
//           (const Vector3<T> &dda, const Vector3<T> &da, Vector3<T> &dx), {
//             for (size_t i = 0; i < 6; i++) {
//               dx[i] = dda[i] * da[i];
//             }
//           })
// TRACTOR_D(barrier_diagonal, quaternion_trust_region_constraint,
//           (const Vector3<T> &dda, Vector3<T> &ddx), { ddx = dda; })

} // namespace tractor
