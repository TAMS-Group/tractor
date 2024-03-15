// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>
#include <tractor/geometry/twist.h>
#include <tractor/geometry/vector3_ops.h>

namespace tractor {

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

}  // namespace tractor
