// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>
#include <tractor/geometry/matrix3.h>
#include <tractor/geometry/pose.h>
#include <tractor/geometry/quaternion.h>
#include <tractor/geometry/twist.h>
#include <tractor/geometry/vector3.h>

namespace tractor {

TRACTOR_VAR_OP(dot)
TRACTOR_VAR_OP(cross)

// ---------------------------------------------------------

TRACTOR_OP_T(mat3, move, (const Matrix3<T> &v), { return Matrix3<T>(v); })
TRACTOR_D_T(prepare, mat3, move, (const Matrix3<T> &a, const Matrix3<T> &x), {})
TRACTOR_D_T(forward, mat3, move, (const Matrix3<T> &da, Matrix3<T> &dx),
            { dx = da; })
TRACTOR_D_T(reverse, mat3, move, (Matrix3<T> & da, const Matrix3<T> &dx),
            { da = dx; })

TRACTOR_OP_T(mat3_vec3, mul, (const Matrix3<T> &a, const Vector3<T> &b), {
  // return a * b;
  Vector3<T> x = Vector3<T>::Zero();
  for (size_t row = 0; row < 3; row++) {
    for (size_t col = 0; col < 3; col++) {
      x[row] += a(row, col) * b[col];
    }
  }
  return x;
})
TRACTOR_D_T(forward, mat3_vec3, mul,
            (const Matrix3<T> &pa, const Vector3<T> &pb, const Vector3<T> &px,
             const Matrix3<T> &da, const Vector3<T> &db, Vector3<T> &dx),
            {
              // dx = pa * db + da * pb;
              dx.setZero();
              for (size_t row = 0; row < 3; row++) {
                for (size_t col = 0; col < 3; col++) {
                  dx[row] += da(row, col) * pb[col];
                  dx[row] += pa(row, col) * db[col];
                }
              }
            })
TRACTOR_D_T(reverse, mat3_vec3, mul,
            (const Matrix3<T> &pa, const Vector3<T> &pb, const Vector3<T> &px,
             Matrix3<T> &da, Vector3<T> &db, const Vector3<T> &dx),
            {
              da.setZero();
              db.setZero();
              for (size_t row = 0; row < 3; row++) {
                for (size_t col = 0; col < 3; col++) {
                  da(row, col) += dx[row] * pb[col];
                  db[col] += dx[row] * pa(row, col);
                }
              }
            })

TRACTOR_OP_T(mat3, add, (const Matrix3<T> &a, const Matrix3<T> &b),
             { return a + b; })
TRACTOR_D_T(prepare, mat3, add,
            (const Matrix3<T> &a, const Matrix3<T> &b, const Matrix3<T> &x), {})
TRACTOR_D_T(forward, mat3, add,
            (const Matrix3<T> &da, const Matrix3<T> &db, Matrix3<T> &dx),
            { dx = da + db; })
TRACTOR_D_T(reverse, mat3, add,
            (Matrix3<T> & da, Matrix3<T> &db, const Matrix3<T> &dx), {
              da = dx;
              db = dx;
            })

TRACTOR_OP_T(mat3, zero, (Matrix3<T> & x), { x.setZero(); })
TRACTOR_D_T(prepare, mat3, zero, (const Matrix3<T> &x), {})
TRACTOR_D_T(forward, mat3, zero, (Matrix3<T> & dx), { dx.setZero(); })
TRACTOR_D_T(reverse, mat3, zero, (const Matrix3<T> &dx), {})

TRACTOR_OP_T(mat3, minus, (const Matrix3<T> &a), { return -a; })
TRACTOR_D_T(prepare, mat3, minus, (const Matrix3<T> &a, const Matrix3<T> &x),
            {})
TRACTOR_D_T(forward, mat3, minus, (const Matrix3<T> &da, Matrix3<T> &dx),
            { dx = -da; })
TRACTOR_D_T(reverse, mat3, minus, (Matrix3<T> & da, const Matrix3<T> &dx),
            { da = -dx; })

// ---------------------------------------------------------

TRACTOR_OP_T(vec3, minus, (const Vector3<T> &a), { return -a; })
TRACTOR_D_T(prepare, vec3, minus, (const Vector3<T> &a, const Vector3<T> &x),
            {})
TRACTOR_D_T(forward, vec3, minus, (const Vector3<T> &da, Vector3<T> &dx),
            { dx = -da; })
TRACTOR_D_T(reverse, vec3, minus, (Vector3<T> & da, const Vector3<T> &dx),
            { da = -dx; })

TRACTOR_OP_T(vec3, zero, (Vector3<T> & x), { x.setZero(); })
TRACTOR_D_T(prepare, vec3, zero, (const Vector3<T> &x), {})
TRACTOR_D_T(forward, vec3, zero, (Vector3<T> & dx), { dx.setZero(); })
TRACTOR_D_T(reverse, vec3, zero, (const Vector3<T> &dx), {})

TRACTOR_OP_T(vec3, move, (const Vector3<T> &v), { return Vector3<T>(v); })
TRACTOR_D_T(prepare, vec3, move, (const Vector3<T> &a, const Vector3<T> &x), {})
TRACTOR_D_T(forward, vec3, move, (const Vector3<T> &da, Vector3<T> &dx),
            { dx = da; })
TRACTOR_D_T(reverse, vec3, move, (Vector3<T> & da, const Vector3<T> &dx),
            { da = dx; })

TRACTOR_OP_T(vec3, add, (const Vector3<T> &a, const Vector3<T> &b),
             { return a + b; })
TRACTOR_D_T(prepare, vec3, add,
            (const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &x), {})
TRACTOR_D_T(forward, vec3, add,
            (const Vector3<T> &da, const Vector3<T> &db, Vector3<T> &dx),
            { dx = da + db; })
TRACTOR_D_T(reverse, vec3, add,
            (Vector3<T> & da, Vector3<T> &db, const Vector3<T> &dx), {
              da = dx;
              db = dx;
            })

TRACTOR_OP_T(vec3, sub, (const Vector3<T> &a, const Vector3<T> &b),
             { return a - b; })
TRACTOR_D_T(prepare, vec3, sub,
            (const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &x), {})
TRACTOR_D_T(forward, vec3, sub,
            (const Vector3<T> &da, const Vector3<T> &db, Vector3<T> &dx),
            { dx = da - db; })
TRACTOR_D_T(reverse, vec3, sub,
            (Vector3<T> & da, Vector3<T> &db, const Vector3<T> &dx), {
              da = dx;
              db = -dx;
            })

TRACTOR_OP_T(vec3_s, mul, (const Vector3<T> &a, const T &b), { return a * b; })
TRACTOR_D_T(prepare, vec3_s, mul,
            (const Vector3<T> &a, const T &b, const Vector3<T> &x,
             Vector3<T> &va, T &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, vec3_s, mul,
            (const Vector3<T> &va, const T &vb, const Vector3<T> &da,
             const T &db, Vector3<T> &dx),
            {
              dx = da * vb + va * db;
              /*dx.x() = da.x() * vb + db * va.x();
              dx.x() = da.y() * vb + db * va.y();
              dx.x() = da.z() * vb + db * va.z();*/
            })
TRACTOR_D_T(reverse, vec3_s, mul,
            (const Vector3<T> &va, const T &vb, Vector3<T> &da, T &db,
             const Vector3<T> &dx),
            {
              da = dx * vb;
              db = dx.x() * va.x() + dx.y() * va.y() + dx.z() * va.z();
            })

TRACTOR_OP_T(s_vec3, mul, (const T &a, const Vector3<T> &b), { return a * b; })
TRACTOR_D_T(prepare, s_vec3, mul,
            (const T &a, const Vector3<T> &b, const Vector3<T> &x, T &va,
             Vector3<T> &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, s_vec3, mul,
            (const T &va, const Vector3<T> &vb, const T &da,
             const Vector3<T> &db, Vector3<T> &dx),
            { dx = da * vb + va * db; })
TRACTOR_D_T(reverse, s_vec3, mul,
            (const T &va, const Vector3<T> &vb, T &da, Vector3<T> &db,
             const Vector3<T> &dx),
            {
              da = dx.x() * vb.x() + dx.y() * vb.y() + dx.z() * vb.z();
              db = dx * va;
            })

//    | da.x da.y da.z db.x db.y db.z
// ---+------------------------------
// dx | vb.x vb.y vb.z va.x va.y va.z
TRACTOR_OP_T(vec3, dot, (const Vector3<T> &a, const Vector3<T> &b),
             { return dot(a, b); })
TRACTOR_D_T(prepare, vec3, dot,
            (const Vector3<T> &a, const Vector3<T> &b, const T &x,
             Vector3<T> &va, Vector3<T> &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, vec3, dot,
            (const Vector3<T> &va, const Vector3<T> &vb, const Vector3<T> &da,
             const Vector3<T> &db, T &dx),
            {
              // dx = (va.x() * db.x() + va.y() * db.y() + va.z() * db.z()) +
              //       (da.x() * vb.x() + da.y() * vb.y() + da.z() * vb.z());
              dx = dot(va, db) + dot(da, vb);
            })
TRACTOR_D_T(reverse, vec3, dot,
            (const Vector3<T> &va, const Vector3<T> &vb, Vector3<T> &da,
             Vector3<T> &db, const T &dx),
            {
              // da.x() = dx * vb.x();
              // da.y() = dx * vb.y();
              // da.z() = dx * vb.z();
              // db.x() = dx * va.x();
              // db.y() = dx * va.y();
              // db.z() = dx * va.z();
              da = vb * dx;
              db = va * dx;
            })

//      |  da.x  da.y   da.z  |  db.x  db.y   db.z
// -----+---------------------+------------------
// dx.x |   0   +vb.z  -vb.y  |   0   -va.z  +va.y
// dx.y | -vb.z   0     vb.x  |  va.z   0    -va.x
// dx.z |  vb.y -vb.x    0    | -va.y  va.x   0
TRACTOR_OP_T(vec3, cross, (const Vector3<T> &a, const Vector3<T> &b),
             { return cross(a, b); })
TRACTOR_D_T(prepare, vec3, cross,
            (const Vector3<T> &a, const Vector3<T> &b, const Vector3<T> &x,
             Vector3<T> &va, Vector3<T> &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, vec3, cross,
            (const Vector3<T> &va, const Vector3<T> &vb, const Vector3<T> &da,
             const Vector3<T> &db, Vector3<T> &dx),
            {
              dx.x() = (da.y() * vb.z() - da.z() * vb.y()) +
                       (va.y() * db.z() - va.z() * db.y());
              dx.y() = (da.z() * vb.x() - da.x() * vb.z()) +
                       (va.z() * db.x() - va.x() * db.z());
              dx.z() = (da.x() * vb.y() - da.y() * vb.x()) +
                       (va.x() * db.y() - va.y() * db.x());
            })
TRACTOR_D_T(reverse, vec3, cross,
            (const Vector3<T> &va, const Vector3<T> &vb, Vector3<T> &da,
             Vector3<T> &db, const Vector3<T> &dx),
            {
              da.x() = vb.y() * dx.z() - vb.z() * dx.y();
              da.y() = vb.z() * dx.x() - vb.x() * dx.z();
              da.z() = vb.x() * dx.y() - vb.y() * dx.x();
              db.x() = va.z() * dx.y() - va.y() * dx.z();
              db.y() = va.x() * dx.z() - va.z() * dx.x();
              db.z() = va.y() * dx.x() - va.x() * dx.y();
            })

TRACTOR_OP(vec3_unpack, (const Vector3<T> &v, T &x, T &y, T &z),
           { vec3_unpack(v, x, y, z); })
TRACTOR_D(prepare, vec3_unpack,
          (const Vector3<T> &v, const T &x, const T &y, const T &z), {})
TRACTOR_D(forward, vec3_unpack, (const Vector3<T> &dv, T &dx, T &dy, T &dz), {
  dx = dv.x();
  dy = dv.y();
  dz = dv.z();
})
TRACTOR_D(reverse, vec3_unpack,
          (Vector3<T> & dv, const T &dx, const T &dy, const T &dz), {
            dv.x() = dx;
            dv.y() = dy;
            dv.z() = dz;
          })

TRACTOR_OP(vec3_pack, (const T &x, const T &y, const T &z, Vector3<T> &vec),
           { vec3_pack(x, y, z, vec); })
TRACTOR_D(prepare, vec3_pack,
          (const T &x, const T &y, const T &z, const Vector3<T> &v), {})
TRACTOR_D(forward, vec3_pack,
          (const T &x, const T &y, const T &z, Vector3<T> &v), {
            v.x() = x;
            v.y() = y;
            v.z() = z;
          })
TRACTOR_D(reverse, vec3_pack, (T & x, T &y, T &z, const Vector3<T> &v), {
  x = v.x();
  y = v.y();
  z = v.z();
})

/*
template <class Scalar> void goal(const Var<Vector3<Scalar>> &v) {
  Var<Scalar> x, y, z;
  vec3_unpack(v, x, y, z);
  goal(x);
  goal(y);
  goal(z);
}
*/

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
           const Quaternion<T> &x, Quaternion<T> &v),
          { v = x; })
TRACTOR_D(forward, quat_pack,
          (const Quaternion<T> &v, const T &da, const T &db, const T &dc,
           const T &dd, Vector3<T> &dx),
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

            dx = quat_pack_forward(v, Quaternion<T>(da, db, dc, dd));
          })
TRACTOR_D(reverse, quat_pack,
          (const Quaternion<T> &v, T &da, T &db, T &dc, T &dd,
           const Vector3<T> &dx),
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

            Quaternion d = quat_pack_reverse(v, dx);
            da = d.x();
            db = d.y();
            dc = d.z();
            dd = d.w();
          })

// -------------------------------------------------------------------------

// TRACTOR_OP(quat_residual, (const Quaternion<T> &a), { return a.vec() *
// T(2);
// })

// TRACTOR_OP(quat_residual, (const Quaternion<T> &a),
//            { return quat_residual(a); })
// TRACTOR_D(prepare, quat_residual, (const Quaternion<T> &a, const Vector3<T>
// &x),
//           {})
// TRACTOR_D(forward, quat_residual, (const Vector3<T> &a, Vector3<T> &x),
//           { x = a; })
// TRACTOR_D(reverse, quat_residual, (Vector3<T> & a, const Vector3<T> &x),
//           { a = x; })

TRACTOR_OP(quat_residual, (const Quaternion<T> &a),
           { return quat_residual(a); })
TRACTOR_D(forward, quat_residual,
          (const Quaternion<T> &va, const Vector3<T> &vx, const Vector3<T> &a,
           Vector3<T> &x),
          { x = va * a; })
TRACTOR_D(reverse, quat_residual,
          (const Quaternion<T> &va, const Vector3<T> &vx, Vector3<T> &a,
           const Vector3<T> &x),
          { a = x; })

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

TRACTOR_GRADIENT_TYPE_TEMPLATE(Pose<T>, Twist<T>);

TRACTOR_OP_T(pose, zero, (Pose<T> & x), { x.setZero(); })
TRACTOR_D_T(prepare, pose, zero, (const Pose<T> &x), {})
TRACTOR_D_T(forward, pose, zero, (Pose<T> & dx), { dx.setZero(); })
TRACTOR_D_T(reverse, pose, zero, (const Pose<T> &dx), {})

TRACTOR_OP_T(twist, zero, (Twist<T> & x), { x.setZero(); })
TRACTOR_D_T(prepare, twist, zero, (const Twist<T> &x), {})
TRACTOR_D_T(forward, twist, zero, (Twist<T> & dx), { dx.setZero(); })
TRACTOR_D_T(reverse, twist, zero, (const Twist<T> &dx), {})

TRACTOR_OP_T(twist, minus, (const Twist<T> &a), { return -a; })
TRACTOR_D_T(prepare, twist, minus, (const Twist<T> &a, const Twist<T> &x), {})
TRACTOR_D_T(forward, twist, minus, (const Twist<T> &da, Twist<T> &dx),
            { dx = -da; })
TRACTOR_D_T(reverse, twist, minus, (Twist<T> & da, const Twist<T> &dx),
            { da = -dx; })

TRACTOR_OP_T(pose, move, (const Pose<T> &v), { return Pose<T>(v); })
TRACTOR_D_T(prepare, pose, move, (const Pose<T> &a, const Pose<T> &x), {})
TRACTOR_D_T(forward, pose, move, (const Twist<T> &da, Twist<T> &dx),
            { dx = da; })
TRACTOR_D_T(reverse, pose, move, (Twist<T> & da, const Twist<T> &dx),
            { da = dx; })

TRACTOR_OP_T(twist, move, (const Twist<T> &v), { return Twist<T>(v); })
TRACTOR_D_T(prepare, twist, move, (const Twist<T> &a, const Twist<T> &x), {})
TRACTOR_D_T(forward, twist, move, (const Twist<T> &da, Twist<T> &dx),
            { dx = da; })
TRACTOR_D_T(reverse, twist, move, (Twist<T> & da, const Twist<T> &dx),
            { da = dx; })

TRACTOR_OP_T(twist, add, (const Twist<T> &a, const Twist<T> &b),
             { return a + b; })
TRACTOR_D_T(prepare, twist, add,
            (const Twist<T> &a, const Twist<T> &b, const Twist<T> &x), {})
TRACTOR_D_T(forward, twist, add,
            (const Twist<T> &da, const Twist<T> &db, Twist<T> &dx),
            { dx = da + db; })
TRACTOR_D_T(reverse, twist, add,
            (Twist<T> & da, Twist<T> &db, const Twist<T> &dx), {
              da = dx;
              db = dx;
            })

TRACTOR_OP_T(twist_s, mul, (const Twist<T> &a, const T &b), { return a * b; })
TRACTOR_D_T(prepare, twist_s, mul,
            (const Twist<T> &a, const T &b, const Twist<T> &x, Twist<T> &va,
             T &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, twist_s, mul,
            (const Twist<T> &va, const T &vb, const Twist<T> &da, const T &db,
             Twist<T> &dx),
            { dx = da * vb + va * db; })
TRACTOR_D_T(reverse, twist_s, mul,
            (const Twist<T> &va, const T &vb, Twist<T> &da, T &db,
             const Twist<T> &dx),
            {
              da = dx * vb;
              db = dot(dx.translation(), va.translation()) +
                   dot(dx.rotation(), va.rotation());
            })

TRACTOR_OP_T(s_twist, mul, (const T &a, const Twist<T> &b), { return a * b; })
TRACTOR_D_T(prepare, s_twist, mul,
            (const T &a, const Twist<T> &b, const Twist<T> &x, T &va,
             Twist<T> &vb),
            {
              va = a;
              vb = b;
            })
TRACTOR_D_T(forward, s_twist, mul,
            (const T &va, const Twist<T> &vb, const T &da, const Twist<T> &db,
             Twist<T> &dx),
            { dx = da * vb + va * db; })
TRACTOR_D_T(reverse, s_twist, mul,
            (const T &va, const Twist<T> &vb, T &da, Twist<T> &db,
             const Twist<T> &dx),
            {
              db = dx * va;
              da = dot(dx.translation(), vb.translation()) +
                   dot(dx.rotation(), vb.rotation());
            })

TRACTOR_OP_T(twist, sub, (const Twist<T> &a, const Twist<T> &b),
             { return a - b; })
TRACTOR_D_T(prepare, twist, sub,
            (const Twist<T> &a, const Twist<T> &b, const Twist<T> &x), {})
TRACTOR_D_T(forward, twist, sub,
            (const Twist<T> &da, const Twist<T> &db, Twist<T> &dx),
            { dx = da - db; })
TRACTOR_D_T(reverse, twist, sub,
            (Twist<T> & da, Twist<T> &db, const Twist<T> &dx), {
              da = dx;
              db.translation() = -dx.translation();
              db.rotation() = -dx.rotation();
            })

template <class T> struct PoseMulState {
  Quaternion<T> ar;
  Vector3<T> arbt;
  Quaternion<T> arinv;
};
TRACTOR_OP_T(pose, mul, (const Pose<T> &a, const Pose<T> &b), { return a * b; })
TRACTOR_D_T(prepare, pose, mul,
            (const Pose<T> &a, const Pose<T> &b, const Pose<T> &x,
             PoseMulState<T> &v),
            {
              // v.at = at;
              v.ar = a.orientation();
              // v.bt = bt;
              // v.br = br;
              v.arbt = a.orientation() * b.translation();
              v.arinv = a.orientation().inverse();
            })
TRACTOR_D_T(forward, pose, mul,
            (const PoseMulState<T> &v, const Twist<T> &da, const Twist<T> &db,
             Twist<T> &dx),
            {
              // xt = at + ar * bt
              // dxt = dat + v.ar * dbt + cross(dar, v.ar * v.bt);
              dx.translation() = da.translation() + v.ar * db.translation() +
                                 cross(da.rotation(), v.arbt);

              // xr = ar * br
              dx.rotation() = v.ar * db.rotation() + da.rotation();
            })
TRACTOR_D_T(reverse, pose, mul,
            (const PoseMulState<T> &v, Twist<T> &da, Twist<T> &db,
             const Twist<T> &dx),
            {
              // xt = at + ar * bt

              // dat = dxt;
              // dar = cross(v.ar * v.bt, dxt);
              // dbt = v.ar.inverse() * dxt;

              da.translation() = dx.translation();
              da.rotation() = cross(v.arbt, dx.translation());
              db.translation() = v.arinv * dx.translation();

              // xr = ar * br

              // dar = dar + dxr;
              // dbr = v.ar.inverse() * dxr;

              da.rotation() = da.rotation() + dx.rotation();
              db.rotation() = v.arinv * dx.rotation();
            })

template <class T> struct PoseVec3MulState {
  Quaternion<T> ar;
  Vector3<T> arbt;
  Quaternion<T> arinv;
};
TRACTOR_OP_T(pose_vec3, mul, (const Pose<T> &a, const Vector3<T> &b),
             { return a * b; })
TRACTOR_D_T(prepare, pose_vec3, mul,
            (const Pose<T> &a, const Vector3<T> &b, const Vector3<T> &x,
             PoseVec3MulState<T> &v),
            {
              v.ar = a.orientation();
              v.arbt = a.orientation() * b;
              v.arinv = a.orientation().inverse();
            })
TRACTOR_D_T(forward, pose_vec3, mul,
            (const PoseVec3MulState<T> &v, const Twist<T> &da,
             const Vector3<T> &db, Vector3<T> &dx),
            {
              dx = da.translation() + v.ar * db + cross(da.rotation(), v.arbt);
            })
TRACTOR_D_T(reverse, pose_vec3, mul,
            (const PoseVec3MulState<T> &v, Twist<T> &da, Vector3<T> &db,
             const Vector3<T> &dx),
            {
              da.translation() = dx;
              da.rotation() = cross(v.arbt, dx);
              db = v.arinv * dx;
            })

// template <class T> struct FGAngleAxisPoseState {
//   T angle;
//   Vector3<T> axis;
// };
// TRACTOR_OP(angle_axis_pose, (const T &angle, const Vector3<T> &axis),
//            { return angle_axis_pose(angle, axis); })
// TRACTOR_D(prepare, angle_axis_pose,
//           (const T &angle, const Vector3<T> &axis, const Pose<T> &pose,
//            FGAngleAxisPoseState<T> &v),
//           {
//             v.angle = angle;
//             v.axis = axis;
//           })
// TRACTOR_D(forward, angle_axis_pose,
//           (const FGAngleAxisPoseState<T> &v, const T &d_angle,
//            const Vector3<T> &d_axis, Twist<T> &d_pose),
//           {
//             d_pose.translation().setZero();
//             d_pose.rotation() = v.axis * d_angle + d_axis * v.angle;
//           })
// TRACTOR_D(reverse, angle_axis_pose,
//           (const FGAngleAxisPoseState<T> &v, T &d_angle, Vector3<T> &d_axis,
//            const Twist<T> &d_pose),
//           {
//             d_angle = dot(d_pose.rotation(), v.axis);
//             d_axis = d_pose.rotation() * v.angle;
//           })

TRACTOR_OP(angle_axis_pose, (const T &angle, const Vector3<T> &axis),
           { return angle_axis_pose(angle, axis); })
TRACTOR_D(prepare, angle_axis_pose,
          (const T &angle, const Vector3<T> &axis, const Pose<T> &pose,
           AngleAxisQuatLinerization<T> &v),
          {
            v.axis_normalized = normalized(axis);
            v.sin_angle_by_axis_length = T(sin(angle)) / norm(axis);
            v.cos_angle_minus_one_by_axis_length =
                (T(cos(angle)) - T(1)) / norm(axis);
          })
TRACTOR_D(forward, angle_axis_pose,
          (const AngleAxisQuatLinerization<T> &v, const T &d_angle,
           const Vector3<T> &d_axis, Twist<T> &d_pose),
          {
            Vector3<T> d_axis_p =
                (d_axis - v.axis_normalized * dot(v.axis_normalized, d_axis));
            d_pose.rotation() = v.axis_normalized * d_angle             //
                                + d_axis_p * v.sin_angle_by_axis_length //
                                + cross(d_axis_p, v.axis_normalized) *
                                      v.cos_angle_minus_one_by_axis_length;
            d_pose.translation().setZero();
          })
TRACTOR_D(reverse, angle_axis_pose,
          (const AngleAxisQuatLinerization<T> &v, T &d_angle,
           Vector3<T> &d_axis, const Twist<T> &d_pose),
          {
            Vector3<T> d_rot = d_pose.rotation();
            Vector3<T> d_rot_p =
                (d_rot - v.axis_normalized * dot(v.axis_normalized, d_rot));
            d_angle = dot(v.axis_normalized, d_rot);
            d_axis = d_rot_p * v.sin_angle_by_axis_length +
                     cross(v.axis_normalized, d_rot_p) *
                         v.cos_angle_minus_one_by_axis_length;
          })

// template <class T> struct PoseAngleAxisPoseState {
//   // Pose<T> parent;
//   Quaternion<T> parent_orientation;
//   Quaternion<T> parent_orientation_inverse;
//   Vector3<T> parent_orientation_axis;
//   T angle;
//   Vector3<T> axis;
// };
// TRACTOR_OP(pose_angle_axis_pose,
//            (const Pose<T> &parent, const T &angle, const Vector3<T> &axis),
//            { return pose_angle_axis_pose(parent, angle, axis); })
// TRACTOR_D(prepare, pose_angle_axis_pose,
//           (const Pose<T> &parent, const T &angle, const Vector3<T> &axis,
//            const Pose<T> &pose, PoseAngleAxisPoseState<T> &v),
//           {
//             // v.parent = parent;
//             v.parent_orientation = parent.orientation();
//             v.parent_orientation_inverse = parent.orientation().inverse();
//             v.parent_orientation_axis = parent.orientation() * axis;
//             v.angle = angle;
//             v.axis = axis;
//           })
// TRACTOR_D(forward, pose_angle_axis_pose,
//           (const PoseAngleAxisPoseState<T> &v, const Twist<T> &d_parent,
//            const T &d_angle, const Vector3<T> &d_axis, Twist<T> &d_pose),
//           {
//             // d_pose.translation() = d_parent.translation();
//             // d_pose.rotation() = d_parent.rotation() +
//             //                    (v.parent.orientation() * v.axis) * d_angle
//             +
//             //                    (v.parent.orientation() * d_axis) *
//             v.angle;
//
//             // d_pose.translation() = d_parent.translation();
//             // d_pose.rotation() = d_parent.rotation() +
//             //                    (v.parent_orientation_axis) * d_angle +
//             //                    (v.parent_orientation * d_axis) * v.angle;
//
//             // dx.translation() = da.translation() + v.ar * db.translation()
//             +
//             //                   cross(da.rotation(), v.arbt);
//             // dx.rotation() = v.ar * db.rotation() + da.rotation();
//
//             d_pose.translation() = d_parent.translation();
//             d_pose.rotation() = d_parent.rotation() +
//                                 (v.parent_orientation_axis) * d_angle +
//                                 (v.parent_orientation * d_axis) * v.angle;
//           })
// TRACTOR_D(reverse, pose_angle_axis_pose,
//           (const PoseAngleAxisPoseState<T> &v, Twist<T> &d_parent, T
//           &d_angle,
//            Vector3<T> &d_axis, const Twist<T> &d_pose),
//           {
//             // d_parent.translation() = d_pose.translation();
//             // d_parent.rotation() = d_pose.rotation();
//             // d_angle = dot(v.parent.orientation().inverse() *
//             // d_pose.rotation(),
//             //              v.axis);
//             // d_axis = (v.parent.orientation().inverse() *
//             d_pose.rotation()) *
//             //         v.angle;
//
//             d_parent.translation() = d_pose.translation();
//             d_parent.rotation() = d_pose.rotation();
//             d_angle =
//                 dot(v.parent_orientation_inverse * d_pose.rotation(),
//                 v.axis);
//             d_axis =
//                 (v.parent_orientation_inverse * d_pose.rotation()) * v.angle;
//           })

TRACTOR_OP(pose_translation, (const Pose<T> &pose),
           { return pose_translation(pose); })
TRACTOR_D(prepare, pose_translation,
          (const Pose<T> &pose, const Vector3<T> &vec), {})
TRACTOR_D(forward, pose_translation,
          (const Twist<T> &twist, Vector3<T> &translation),
          { translation = twist.translation(); })
TRACTOR_D(reverse, pose_translation,
          (Twist<T> & twist, const Vector3<T> &translation), {
            twist.translation() = translation;
            twist.rotation().setZero();
          })

TRACTOR_OP(pose_orientation, (const Pose<T> &pose),
           { return pose_orientation(pose); })
TRACTOR_D(prepare, pose_orientation,
          (const Pose<T> &pose, const Quaternion<T> &orientation), {})
TRACTOR_D(forward, pose_orientation,
          (const Twist<T> &twist, Vector3<T> &rotation),
          { rotation = twist.rotation(); })
TRACTOR_D(reverse, pose_orientation,
          (Twist<T> & twist, const Vector3<T> &rotation), {
            twist.rotation() = rotation;
            twist.translation().setZero();
          })

TRACTOR_OP(translation_pose, (const Vector3<T> &translation),
           { return translation_pose(translation); })
TRACTOR_D(prepare, translation_pose,
          (const Vector3<T> &translation, const Pose<T> &pose), {})
TRACTOR_D(forward, translation_pose,
          (const Vector3<T> &translation, Twist<T> &twist), {
            twist.translation() = translation;
            twist.rotation().setZero();
          })
TRACTOR_D(reverse, translation_pose,
          (Vector3<T> & translation, const Twist<T> &twist),
          { translation = twist.translation(); })

TRACTOR_OP(translation_twist, (const Vector3<T> &translation),
           { return translation_twist(translation); })
TRACTOR_D(prepare, translation_twist,
          (const Vector3<T> &translation, const Twist<T> &pose), {})
TRACTOR_D(forward, translation_twist,
          (const Vector3<T> &translation, Twist<T> &twist), {
            twist.translation() = translation;
            twist.rotation().setZero();
          })
TRACTOR_D(reverse, translation_twist,
          (Vector3<T> & translation, const Twist<T> &twist),
          { translation = twist.translation(); })

TRACTOR_OP(make_twist,
           (const Vector3<T> &translation, const Vector3<T> &rotation),
           { return make_twist(translation, rotation); })
TRACTOR_D(prepare, make_twist,
          (const Vector3<T> &translation, const Vector3<T> &rotation,
           const Twist<T> &twist),
          {})
TRACTOR_D(forward, make_twist,
          (const Vector3<T> &translation, const Vector3<T> &rotation,
           Twist<T> &twist),
          {
            twist.translation() = translation;
            twist.rotation() = rotation;
          })
TRACTOR_D(reverse, make_twist,
          (Vector3<T> & translation, Vector3<T> &rotation,
           const Twist<T> &twist),
          {
            translation = twist.translation();
            rotation = twist.rotation();
          })

TRACTOR_OP(orientation_pose, (const Quaternion<T> &orientation),
           { return orientation_pose(orientation); })
TRACTOR_D(prepare, orientation_pose,
          (const Quaternion<T> &orientation, const Pose<T> &pose), {})
TRACTOR_D(forward, orientation_pose,
          (const Vector3<T> &rotation, Twist<T> &twist), {
            twist.rotation() = rotation;
            twist.translation().setZero();
          })
TRACTOR_D(reverse, orientation_pose,
          (Vector3<T> & rotation, const Twist<T> &twist),
          { rotation = twist.rotation(); })

TRACTOR_OP(pose_translate,
           (const Pose<T> &parent, const Vector3<T> &translation),
           { return pose_translate(parent, translation); })
TRACTOR_D(prepare, pose_translate,
          (const Pose<T> &parent, const Vector3<T> &translation,
           const Pose<T> &pose, Quaternion<T> &parent_orientation),
          { parent_orientation = parent.orientation(); })
TRACTOR_D(forward, pose_translate,
          (const Quaternion<T> &parent_orientation, const Twist<T> &parent,
           const Vector3<T> &translation, Twist<T> &twist),
          {
            twist.rotation() = parent.rotation();
            twist.translation() =
                parent.translation() + parent_orientation * translation;
          })
TRACTOR_D(reverse, pose_translate,
          (const Quaternion<T> &parent_orientation, Twist<T> &parent,
           Vector3<T> &translation, const Twist<T> &twist),
          {
            parent = twist;
            translation = parent_orientation.inverse() * twist.translation();
          })

template <class T> struct PoseResidualState {
  Pose<T> a;
  Pose<T> b;
};
TRACTOR_OP(pose_residual, (const Pose<T> &a, const Pose<T> &b),
           { return pose_residual(a, b); })
TRACTOR_D(prepare, pose_residual,
          (const Pose<T> &a, const Pose<T> &b, const Twist<T> &x,
           PoseResidualState<T> &v),
          {
            v.a = a;
            v.b = b;
          })
TRACTOR_D(forward, pose_residual,
          (const PoseResidualState<T> &v, const Twist<T> &da,
           const Twist<T> &db, Twist<T> &dx),
          {
            // dx.translation() = da.translation() - db.translation();
            // dx.rotation() =
            //    v.a.orientation().inverse() * db.rotation() - da.rotation();
            dx = db - da;
          })
TRACTOR_D(reverse, pose_residual,
          (const PoseResidualState<T> &v, Twist<T> &da, Twist<T> &db,
           const Twist<T> &dx),
          {
            // da.translation() = dx.translation();
            // db.translation() = -dx.translation();
            // da.rotation() = -dx.rotation();
            // db.rotation() = v.a.orientation() * dx.rotation();
            da = -dx;
            db = dx;
          })

TRACTOR_OP(twist_unpack,
           (const Twist<T> &v, T &tx, T &ty, T &tz, T &rx, T &ry, T &rz), {
             tx = v.translation().x();
             ty = v.translation().y();
             tz = v.translation().z();
             rx = v.rotation().x();
             ry = v.rotation().y();
             rz = v.rotation().z();
           })
TRACTOR_D(prepare, twist_unpack,
          (const Twist<T> &v, const T &tx, const T &ty, const T &tz,
           const T &rx, const T &ry, const T &rz),
          {})
TRACTOR_D(forward, twist_unpack,
          (const Twist<T> &v, T &tx, T &ty, T &tz, T &rx, T &ry, T &rz), {
            tx = v.translation().x();
            ty = v.translation().y();
            tz = v.translation().z();
            rx = v.rotation().x();
            ry = v.rotation().y();
            rz = v.rotation().z();
          })
TRACTOR_D(reverse, twist_unpack,
          (Twist<T> & v, const T &tx, const T &ty, const T &tz, const T &rx,
           const T &ry, const T &rz),
          {
            v.translation().x() = tx;
            v.translation().y() = ty;
            v.translation().z() = tz;
            v.rotation().x() = rx;
            v.rotation().y() = ry;
            v.rotation().z() = rz;
          })

TRACTOR_OP(twist_translation, (const Twist<T> &twist),
           { return twist_translation(twist); })
TRACTOR_D(prepare, twist_translation,
          (const Twist<T> &twist, const Vector3<T> &translation), {})
TRACTOR_D(forward, twist_translation,
          (const Twist<T> &twist, Vector3<T> &translation),
          { translation = twist.translation(); })
TRACTOR_D(reverse, twist_translation,
          (Twist<T> & twist, const Vector3<T> &translation), {
            twist.translation() = translation;
            twist.rotation().setZero();
          })

TRACTOR_OP(twist_rotation, (const Twist<T> &twist),
           { return twist_rotation(twist); })
TRACTOR_D(prepare, twist_rotation,
          (const Twist<T> &twist, const Vector3<T> &rotation), {})
TRACTOR_D(forward, twist_rotation,
          (const Twist<T> &twist, Vector3<T> &rotation),
          { rotation = twist.rotation(); })
TRACTOR_D(reverse, twist_rotation,
          (Twist<T> & twist, const Vector3<T> &rotation), {
            twist.translation().setZero();
            twist.rotation() = rotation;
          })

/*
template <class Scalar> void goal(const Var<Twist<Scalar>> &v) {
  Var<Scalar> tx, ty, tz, rx, ry, rz;
  twist_unpack(v, tx, ty, tz, rx, ry, rz);
  goal(tx);
  goal(ty);
  goal(tz);
  goal(rx);
  goal(ry);
  goal(rz);
}
*/

/*
template <class T> void parameter(Pose<T> &pose) {
  parameter(pose.translation());
  parameter(pose.orientation());
}
*/

// -------------------------------------------------------------------------

template <class T> Pose<T> operator+(const Pose<T> &a, const Twist<T> &b) {
  // Pose<T> ret;
  // ret.translation() =
  //     a.translation() + b.translation() + cross(b.rotation(),
  //     a.translation());
  // ret.orientation() =
  //     normalized(normalized(Quaternion<T>(b.rotation().x() * T(0.5),
  //                                         b.rotation().y() * T(0.5),
  //                                         b.rotation().z() * T(0.5), T(1.0)))
  //                                         *
  //                a.orientation());
  // return ret;
  Quaternion<T> qb =
      normalized(normalized(Quaternion<T>(b.rotation().x() * T(0.5), //
                                          b.rotation().y() * T(0.5), //
                                          b.rotation().z() * T(0.5), //
                                          T(1.0)                     //
                                          )));
  Pose<T> pb = Pose(b.translation(), qb);
  return pb * a;
}

TRACTOR_OP_T(pose_twist, add, (const Pose<T> &a, const Twist<T> &b), {
  Pose<T> ret = a + b;
  // TRACTOR_DEBUG("add pose twist " << ret);
  return ret;
})

// TRACTOR_D_T(prepare, pose_twist, add,
//             (const Pose<T> &a, const Twist<T> &b, const Pose<T> &x), {})
// TRACTOR_D_T(forward, pose_twist, add,
//             (const Twist<T> &a, const Twist<T> &b, Twist<T> &x), { x = a + b;
//             })
// TRACTOR_D_T(reverse, pose_twist, add,
//             (Twist<T> & a, Twist<T> &b, const Twist<T> &x), {
//               a = x;
//               b = x;
//             })

template <class T> struct AddPoseTwistLinearization {
  Quaternion<T> bqn;
  Vector3<T> at;
  T bqfh;
};

TRACTOR_D_T(prepare, pose_twist, add,
            (const Pose<T> &a, const Twist<T> &b, const Pose<T> &x,
             AddPoseTwistLinearization<T> &v),
            {
              Quaternion<T> bq = Quaternion<T>(b.rotation().x() * T(0.5), //
                                               b.rotation().y() * T(0.5), //
                                               b.rotation().z() * T(0.5), //
                                               T(1)                       //
              );
              T bqf = T(1) / norm(bq);
              v.bqn = normalized(bq);
              v.bqfh = bqf * T(0.5);
              v.at = a.translation();
            })

TRACTOR_D_T(forward, pose_twist, add,
            (const AddPoseTwistLinearization<T> &v, const Twist<T> &da,
             const Twist<T> &db, Twist<T> &dx),
            {
              Vector3 dqb = quat_pack_forward(v.bqn,
                                              Quaternion<T>(                  //
                                                  db.rotation().x() * v.bqfh, //
                                                  db.rotation().y() * v.bqfh, //
                                                  db.rotation().z() * v.bqfh, //
                                                  T(0)                        //
                                                  ));

              dx.rotation() = dqb + v.bqn * da.rotation();

              dx.translation() = db.translation() + v.bqn * da.translation() +
                                 cross(dqb, v.bqn * v.at);

              // dx = va * db + cross(da, va * vb);

              // auto qvb = Quaternion<T>(vb.rotation().x() * T(0.5), //
              //                          vb.rotation().y() * T(0.5), //
              //                          vb.rotation().z() * T(0.5), //
              //                          T(1)                        //
              // );
              //
              // Quaternion<T> qvbn = normalized(qvb);
              // T qvbf = T(1) / norm(qvb);
              //
              // Quaternion<T> qdbn =
              //     Quaternion<T>(db.rotation().x() * qvbf * T(0.5), //
              //                   db.rotation().y() * qvbf * T(0.5), //
              //                   db.rotation().z() * qvbf * T(0.5), //
              //                   T(0)                               //
              //     );
              //
              // auto db_r = quat_pack_forward(qvbn, qdbn);
              //
              // auto v_ar = va.orientation();
              // auto v_arbt = va.orientation() * vb.translation();
              // auto v_arinv = va.orientation().inverse();
              //
              // dx.translation() = da.translation() + v_ar * db.translation() +
              //                    cross(da.rotation(), v_arbt);
              //
              // dx.rotation() = v_ar * db_r + da.rotation();
            })
TRACTOR_D_T(reverse, pose_twist, add,
            (const AddPoseTwistLinearization<T> &v,
             // const Pose<T> &va, const Twist<T> &vb, const Pose<T> &vx,
             Twist<T> &da, Twist<T> &db, const Twist<T> &dx),
            {
              // a = x;
              // b = x;

              da.rotation() = v.bqn.inverse() * dx.rotation();

              Vector3<T> rot = dx.rotation();

              rot += cross(v.bqn * v.at, dx.translation());

              Quaternion<T> qdb = quat_pack_reverse(v.bqn, rot * v.bqfh);

              db.rotation().x() = qdb.x();
              db.rotation().y() = qdb.y();
              db.rotation().z() = qdb.z();

              db.translation() = dx.translation();

              da.translation() = v.bqn.inverse() * dx.translation();
            })

// -------------------------------------------------------------------------

template <class T>
Quaternion<T> operator+(const Quaternion<T> &a, const Vector3<T> &b) {

  // return normalized(normalized(Quaternion<T>(b.x() * T(0.5), b.y() * T(0.5),
  //                                            b.z() * T(0.5), T(1.0))) *
  //                   a);

  // return normalized(angle_axis_quat(norm(b), normalized(b)) * a);

  return normalized(normalized(Quaternion<T>(b.x() * T(0.5), b.y() * T(0.5),
                                             b.z() * T(0.5), T(1))) *
                    a);

  // Quaternion<T> qb = normalized(
  //     Quaternion<T>(b.x() * T(0.5), b.y() * T(0.5), b.z() * T(0.5), T(1)));
  // return normalized(qb * a);

  // Quaternion<T> qb =
  //     Quaternion<T>(b.x() * T(0.5), b.y() * T(0.5), b.z() * T(0.5), T(1));
  // return qb * a;

  // T angle = norm(b);
  // Vector3<T> axis_n = b / angle;
  //
  // T s = sin(angle * T(0.5));
  // T c = cos(angle * T(0.5));
  //
  // Quaternion<T> quat;
  //
  // quat.x() = axis_n.x() * s;
  // quat.y() = axis_n.y() * s;
  // quat.z() = axis_n.z() * s;
  // quat.w() = c;
  //
  // return quat * a;

  // Quaternion<T> b_quat(a * b.x() * T(0.5), //
  //                      a * b.y() * T(0.5), //
  //                      a * b.z() * T(0.5), //
  //                      T(1));

  // T norm = b.x() * b.x() + b.y() * b.y() + b.z() * b.z();
  // T angle = norm;
  // Vector3<T> axis = v / norm;
  // Quaternion<T> quat;
  // T s = sin(angle * T(0.5));
  // T c = cos(angle * T(0.5));
  // quat.x() = axis.x() * s;
  // quat.y() = axis.y() * s;
  // quat.z() = axis.z() * s;
  // quat.w() = c;
  // return quat;
}
template <class T>
Quaternion<T> &operator+=(Quaternion<T> &a, const Vector3<T> &b) {
  a = a + b;
  return a;
}
TRACTOR_OP_T(quat_vec3, add, (const Quaternion<T> &a, const Vector3<T> &b),
             { return a + b; })
// TRACTOR_D_T(prepare, quat_vec3, add,
//             (const Quaternion<T> &a, const Vector3<T> &b,
//              const Quaternion<T> &x, Quaternion<T> &va, Vector3<T> &vb),
//             {
//               va = a;
//               vb = b;
//               // T angle = norm(b);
//               // Vector3<T> axis = normalized(b);
//               // vb.axis_normalized = normalized(axis);
//               // vb.sin_angle_by_axis_length = T(sin(angle)) / norm(axis);
//               // vb.cos_angle_minus_one_by_axis_length =
//               //     (T(cos(angle)) - T(1)) / norm(axis);
//             })
template <class T> struct QuatVec3AddLinearization {
  Quaternion<T> bqn;
  T bqfh;
};
TRACTOR_D_T(prepare, quat_vec3, add,
            (const Quaternion<T> &a, const Vector3<T> &b,
             const Quaternion<T> &x, QuatVec3AddLinearization<T> &v),
            {
              Quaternion<T> bq = Quaternion<T>(b.x() * T(0.5), //
                                               b.y() * T(0.5), //
                                               b.z() * T(0.5), //
                                               T(1)            //
              );
              T bqf = T(1) / norm(bq);
              v.bqn = normalized(bq);
              v.bqfh = bqf * T(0.5);
            })
TRACTOR_D_T(forward, quat_vec3, add,
            (
                // const Quaternion<T> &va, const Vector3<T> &vb,
                // const Quaternion<T> &vx,
                const QuatVec3AddLinearization<T> &v, const Vector3<T> &da,
                const Vector3<T> &db, Vector3<T> &dx),
            {
              Vector3 dqb = quat_pack_forward(v.bqn,
                                              Quaternion<T>(       //
                                                  db.x() * v.bqfh, //
                                                  db.y() * v.bqfh, //
                                                  db.z() * v.bqfh, //
                                                  T(0)             //
                                                  ));
              dx = dqb + v.bqn * da;

              // Quaternion<T> vbq = Quaternion<T>(vb.x() * T(0.5), //
              //                                   vb.y() * T(0.5), //
              //                                   vb.z() * T(0.5), //
              //                                   T(1)             //
              // );
              // T vbqf = T(1) / norm(vbq);
              // Quaternion<T> vbqn = normalized(vbq);
              // Vector3 dqb = quat_pack_forward(vbqn,
              //                                 Quaternion<T>(              //
              //                                     db.x() * T(0.5) * vbqf, //
              //                                     db.y() * T(0.5) * vbqf, //
              //                                     db.z() * T(0.5) * vbqf, //
              //                                     T(0)                    //
              //                                     ));
              // dx = dqb + vbqn * da;

              // T f = T(1) / norm(Quaternion<T>(vb.x() * T(0.5), //
              //                                 vb.y() * T(0.5), //
              //                                 vb.z() * T(0.5), //
              //                                 T(1)             //
              //                                 ));
              // Quaternion<T> vbq = normalized(Quaternion<T>(vb.x() * T(0.5),
              // //
              //                                              vb.y() * T(0.5),
              //                                              // vb.z() *
              //                                              T(0.5), // T(1) //
              //                                              ));
              // Vector3 dqb = quat_pack_forward(
              //     vbq, Quaternion<T>(db.x() * T(0.5) * f, db.y() * T(0.5) *
              //     f,
              //                        db.z() * T(0.5) * f, T(0)));
              // dx = dqb + vbq * da;

              // Vector3<T> d_axis_p =
              //     (d_axis - v.axis_normalized * dot(v.axis_normalized,
              //     d_axis));
              //
              // d_rot = v.axis_normalized * d_angle             //
              //         + d_axis_p * v.sin_angle_by_axis_length //
              //         + cross(d_axis_p, v.axis_normalized) *
              //               v.cos_angle_minus_one_by_axis_length;

              // x = a + b;
              // Quaternion<T> qvb = normalized(Quaternion<T>(
              //     vb.x() * T(0.5), vb.y() * T(0.5), vb.z() * T(0.5), T(1)));
              //
              // // dx = qvb * da + db * va;
              //
              // // dx = qvb * da + va.inverse() * db;
              // // dx = qvb * da + db;
              //
              // dx = qvb * da + db * va;

              // static auto q = [](const Vector3<T> &v) {
              //   return Quaternion<T>(v.x() * T(0.5), v.y() * T(0.5),
              //                        v.z() * T(0.5), T(1.0));
              // };
              //
              // Quaternion<T> qdx = q(vb + db) * q(da) * va * vx.inverse();
              //
              // dx.x() = qdx.x() * T(2.0);
              // dx.y() = qdx.y() * T(2.0);
              // dx.z() = qdx.z() * T(2.0);
            })
TRACTOR_D_T(reverse, quat_vec3, add,
            (const QuatVec3AddLinearization<T> &v, Vector3<T> &da,
             Vector3<T> &db, const Vector3<T> &dx),
            {
              // da.setZero();
              // db.setZero();

              da = v.bqn.inverse() * dx;
              Quaternion<T> qdb = quat_pack_reverse(v.bqn, dx * v.bqfh);
              db.x() = qdb.x();
              db.y() = qdb.y();
              db.z() = qdb.z();

              // Vector3<T> xdb = dx;
              // Vector3<T> xda = v.bqn.inverse() * dx;
              //
              // Quaternion<T> qdb = quat_pack_reverse(v.bqn,
              //                                       Vector3<T>(           //
              //                                           xdb.x() * v.bqfh, //
              //                                           xdb.y() * v.bqfh, //
              //                                           xdb.z() * v.bqfh  //
              //                                           ));
              //
              // db.x() = qdb.x();
              // db.y() = qdb.y();
              // db.z() = qdb.z();
              //
              // da = xda;
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

template <class T> T gate(const T &a, const T &b) { return a; }
TRACTOR_OP(gate, (const T &a, const T &b), { return a; })
TRACTOR_D(prepare, gate, (const T &a, const T &b, const T &x, T &p), { p = b; })
TRACTOR_D(forward, gate, (const T &p, const T &da, const T &db, T &dx),
          { dx = da * p; })
TRACTOR_D(reverse, gate, (const T &p, T &da, T &db, const T &dx),
          { da = dx * p; })

template <class T> Pose<T> gate(const Pose<T> &a, const T &b) { return a; }
TRACTOR_OP_T(pose_gate, gate, (const Pose<T> &a, const T &b), { return a; })
TRACTOR_D_T(prepare, pose_gate, gate,
            (const Pose<T> &a, const T &b, const Pose<T> &x, T &p), { p = b; })
TRACTOR_D_T(forward, pose_gate, gate,
            (const T &p, const Twist<T> &da, const T &db, Twist<T> &dx), {
              dx.translation() = da.translation() * p;
              dx.rotation() = da.rotation() * p;
            })
TRACTOR_D_T(reverse, pose_gate, gate,
            (const T &p, Twist<T> &da, T &db, const Twist<T> &dx), {
              da.translation() = dx.translation() * p;
              da.rotation() = dx.rotation() * p;
              db = T(0);
            })

// -------------------------------------------------------------------------

TRACTOR_OP(pose_trust_region_constraint, (const Pose<T> &a, const T &tr),
           { return T(0); })
TRACTOR_D(prepare, pose_trust_region_constraint,
          (const Pose<T> &a, const T &tr, const T &x), {})
TRACTOR_D(forward, pose_trust_region_constraint,
          (const Twist<T> &a, const T &tr, T &x), { x = T(0); })
TRACTOR_D(reverse, pose_trust_region_constraint,
          (Twist<T> & a, T &tr, const T &x), {
            a.setZero();
            tr = T(0);
          })
TRACTOR_D(project, pose_trust_region_constraint,
          (const Pose<T> &a, const T &tr, const Twist<T> &da, const T &padding,
           Twist<T> &dx),
          {
            dx = da;
            // for (size_t i = 0; i < 6; i++) {
            //  dx[i] = std::max(-tr, std::min(tr, da[i]));
            //}
          })
TRACTOR_D(barrier_init, pose_trust_region_constraint,
          (const Pose<T> &a, const T &tr, const Twist<T> &da, Twist<T> &dx,
           Twist<T> &ddx),
          {
            for (size_t i = 0; i < 6; i++) {
              T p = da[i];
              T lo2 = -tr;
              T hi2 = +tr;
              T u = T(-1) / std::max(T(0), p - lo2);
              T v = T(+1) / std::max(T(0), hi2 - p);
              dx[i] = u + v;
              ddx[i] = (u * u) + (v * v);
            }
          })
TRACTOR_D(barrier_step, pose_trust_region_constraint,
          (const Twist<T> &dda, const Twist<T> &da, Twist<T> &dx), {
            for (size_t i = 0; i < 6; i++) {
              dx[i] = dda[i] * da[i];
            }
          })
TRACTOR_D(barrier_diagonal, pose_trust_region_constraint,
          (const Twist<T> &dda, Twist<T> &ddx), { ddx = dda; })

// -------------------------------------------------------------------------

template <class T>
auto vector3_trust_region_constraint(const Vector3<T> &a, const T &tr) {
  return T(0);
}
TRACTOR_OP(vector3_trust_region_constraint, (const Vector3<T> &a, const T &tr),
           { return T(0); })
TRACTOR_D(prepare, vector3_trust_region_constraint,
          (const Vector3<T> &a, const T &tr, const T &x), {})
TRACTOR_D(forward, vector3_trust_region_constraint,
          (const Vector3<T> &a, const T &tr, T &x), { x = T(0); })
TRACTOR_D(reverse, vector3_trust_region_constraint,
          (Vector3<T> & a, T &tr, const T &x), {
            a.setZero();
            tr = T(0);
          })
TRACTOR_D(project, vector3_trust_region_constraint,
          (const Vector3<T> &a, const T &tr, const Vector3<T> &da,
           const T &padding, Vector3<T> &dx),
          { dx = da; })
TRACTOR_D(barrier_init, vector3_trust_region_constraint,
          (const Vector3<T> &a, const T &tr, const Vector3<T> &da,
           Vector3<T> &dx, Vector3<T> &ddx),
          {
            for (size_t i = 0; i < 3; i++) {
              T p = da[i];
              T lo2 = -tr;
              T hi2 = +tr;
              T u = T(-1) / std::max(T(0), p - lo2);
              T v = T(+1) / std::max(T(0), hi2 - p);
              dx[i] = u + v;
              ddx[i] = (u * u) + (v * v);
            }
          })
TRACTOR_D(barrier_step, vector3_trust_region_constraint,
          (const Vector3<T> &dda, const Twist<T> &da, Vector3<T> &dx), {
            for (size_t i = 0; i < 6; i++) {
              dx[i] = dda[i] * da[i];
            }
          })
TRACTOR_D(barrier_diagonal, vector3_trust_region_constraint,
          (const Vector3<T> &dda, Vector3<T> &ddx), { ddx = dda; })

// -------------------------------------------------------------------------

template <class T>
auto quaternion_trust_region_constraint(const Quaternion<T> &a, const T &tr) {
  return T(0);
}
TRACTOR_OP(quaternion_trust_region_constraint,
           (const Quaternion<T> &a, const T &tr), { return T(0); })
TRACTOR_D(prepare, quaternion_trust_region_constraint,
          (const Quaternion<T> &a, const T &tr, const T &x), {})
TRACTOR_D(forward, quaternion_trust_region_constraint,
          (const Vector3<T> &a, const T &tr, T &x), { x = T(0); })
TRACTOR_D(reverse, quaternion_trust_region_constraint,
          (Vector3<T> & a, T &tr, const T &x), {
            a.setZero();
            tr = T(0);
          })
TRACTOR_D(project, quaternion_trust_region_constraint,
          (const Quaternion<T> &a, const T &tr, const Vector3<T> &da,
           const T &padding, Vector3<T> &dx),
          { dx = da; })
TRACTOR_D(barrier_init, quaternion_trust_region_constraint,
          (const Quaternion<T> &a, const T &tr, const Vector3<T> &da,
           Vector3<T> &dx, Vector3<T> &ddx),
          {
            for (size_t i = 0; i < 6; i++) {
              T p = da[i];
              T lo2 = -tr;
              T hi2 = +tr;
              T u = T(-1) / std::max(T(0), p - lo2);
              T v = T(+1) / std::max(T(0), hi2 - p);
              dx[i] = u + v;
              ddx[i] = (u * u) + (v * v);
            }
          })
TRACTOR_D(barrier_step, quaternion_trust_region_constraint,
          (const Vector3<T> &dda, const Vector3<T> &da, Vector3<T> &dx), {
            for (size_t i = 0; i < 6; i++) {
              dx[i] = dda[i] * da[i];
            }
          })
TRACTOR_D(barrier_diagonal, quaternion_trust_region_constraint,
          (const Vector3<T> &dda, Vector3<T> &ddx), { ddx = dda; })

// -------------------------------------------------------------------------

template <class T> inline auto squaredNorm(const Var<Vector3<T>> &v) {
  return Var<T>(dot(v, v));
}

template <class T> inline auto norm(const Var<Vector3<T>> &v) {
  return Var<T>(sqrt(dot(v, v)));
}

// template <class T> inline auto normalized(const Var<Vector3<T>> &v) {
//   return v * (Var<T>(T(1)) / norm(v));
// }

TRACTOR_OP(normalized, (const Vector3<T> &a), { return a * (T(1) / norm(a)); })
TRACTOR_D(prepare, normalized,
          (const Vector3<T> &a, const Vector3<T> &x, Vector3<T> &va, T &vf), {
            va = a;
            vf = T(1) / norm(a);
          })
TRACTOR_D(forward, normalized,
          (const Vector3<T> &va, const T &vf, const Vector3<T> &da,
           Vector3<T> &dx),
          { dx = (da - va * (dot(da, va) * vf * vf)) * vf; })
TRACTOR_D(reverse, normalized,
          (const Vector3<T> &va, const T &vf, Vector3<T> &da,
           const Vector3<T> &dx),
          { da = (dx - va * (dot(dx, va) * vf * vf)) * vf; })

} // namespace tractor
