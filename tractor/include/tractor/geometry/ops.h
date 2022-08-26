// (c) 2020-2022 Philipp Ruppel

#pragma once

// #include <tractor/core/operator.h>
// #include <tractor/geometry/matrix3.h>
// #include <tractor/geometry/pose.h>
// #include <tractor/geometry/quaternion.h>
// #include <tractor/geometry/twist.h>
// #include <tractor/geometry/vector3.h>

namespace tractor {

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

// template <class T> T gate(const T &a, const T &b) { return a; }
// TRACTOR_OP(gate, (const T &a, const T &b), { return a; })
// TRACTOR_D(prepare, gate, (const T &a, const T &b, const T &x, T &p), { p = b;
// }) TRACTOR_D(forward, gate, (const T &p, const T &da, const T &db, T &dx),
//           { dx = da * p; })
// TRACTOR_D(reverse, gate, (const T &p, T &da, T &db, const T &dx),
//           { da = dx * p; })
//
// template <class T> Pose<T> gate(const Pose<T> &a, const T &b) { return a; }
// TRACTOR_OP_T(pose_gate, gate, (const Pose<T> &a, const T &b), { return a; })
// TRACTOR_D_T(prepare, pose_gate, gate,
//             (const Pose<T> &a, const T &b, const Pose<T> &x, T &p), { p = b;
//             })
// TRACTOR_D_T(forward, pose_gate, gate,
//             (const T &p, const Twist<T> &da, const T &db, Twist<T> &dx), {
//               dx.translation() = da.translation() * p;
//               dx.rotation() = da.rotation() * p;
//             })
// TRACTOR_D_T(reverse, pose_gate, gate,
//             (const T &p, Twist<T> &da, T &db, const Twist<T> &dx), {
//               da.translation() = dx.translation() * p;
//               da.rotation() = dx.rotation() * p;
//               db = T(0);
//             })

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------

// -------------------------------------------------------------------------

} // namespace tractor
