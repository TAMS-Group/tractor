// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>
#include <tractor/geometry/vector3.h>

namespace tractor {

TRACTOR_VAR_OP(dot)
TRACTOR_VAR_OP(cross)

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
            { dx = da * vb + va * db; })
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
            { dx = dot(va, db) + dot(da, vb); })
TRACTOR_D_T(reverse, vec3, dot,
            (const Vector3<T> &va, const Vector3<T> &vb, Vector3<T> &da,
             Vector3<T> &db, const T &dx),
            {
              da = vb * dx;
              db = va * dx;
            })

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

template <class T>
inline auto squaredNorm(const Var<Vector3<T>> &v) {
  return Var<T>(dot(v, v));
}

template <class T>
inline auto norm(const Var<Vector3<T>> &v) {
  return Var<T>(sqrt(dot(v, v)));
}

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

}  // namespace tractor
