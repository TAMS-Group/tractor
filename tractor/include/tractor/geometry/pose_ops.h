// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>
#include <tractor/geometry/pose.h>
#include <tractor/geometry/quaternion_ops.h>
#include <tractor/geometry/twist_ops.h>
#include <tractor/geometry/vector3_ops.h>

namespace tractor {

TRACTOR_GRADIENT_TYPE_TEMPLATE(Pose<T>, Twist<T>);

TRACTOR_OP_T(pose, zero, (Pose<T> & x), { x.setZero(); })
TRACTOR_D_T(prepare, pose, zero, (const Pose<T> &x), {})
TRACTOR_D_T(forward, pose, zero, (Twist<T> & dx), { dx.setZero(); })
TRACTOR_D_T(reverse, pose, zero, (const Twist<T> &dx), {})

TRACTOR_OP_T(pose, move, (const Pose<T> &v), { return Pose<T>(v); })
TRACTOR_D_T(prepare, pose, move, (const Pose<T> &a, const Pose<T> &x), {})
TRACTOR_D_T(forward, pose, move, (const Twist<T> &da, Twist<T> &dx),
            { dx = da; })
TRACTOR_D_T(reverse, pose, move, (Twist<T> & da, const Twist<T> &dx),
            { da = dx; })

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

template <class T>
inline Pose<T> make_pose(const Vector3<T> &a, const Quaternion<T> &b) {
  return Pose<T>(a, b);
}
TRACTOR_OP(make_pose, (const Vector3<T> &a, const Quaternion<T> &b),
           { return Pose<T>(a, b); })
TRACTOR_D(prepare, make_pose,
          (const Vector3<T> &a, const Quaternion<T> &b, const Pose<T> &x), {})
TRACTOR_D(forward, make_pose,
          (const Vector3<T> &da, const Vector3<T> &db, Twist<T> &dx), {
            dx.translation() = da;
            dx.rotation() = db;
          })
TRACTOR_D(reverse, make_pose,
          (Vector3<T> & da, Vector3<T> &db, const Twist<T> &dx), {
            da = dx.translation();
            db = dx.rotation();
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

// -------------------------------------------------------------------------

// template <class Pose>
// auto pose_residual(const Pose &a)
//     -> decltype(make_twist(a.position(), a.position())) {
//   return make_twist(a.position(), quat_residual(a.orientation()));
// }

template <class T> Twist<T> pose_residual(const Pose<T> &a) {
  Twist<T> x;
  x.translation() = a.translation();
  x.rotation() = quat_residual(a.orientation());
  return x;
}

// template <class T> Twist<T> pose_residual(const Pose<T> &a, const Pose<T> &b)
// {
//   Twist<T> x;
//
//   // x.translation() = a.translation() - b.translation();
//   // x.rotation() = quat_residual(a.orientation().inverse() *
//   // b.orientation());
//
//   // residual * a = b
//   // (residual * a)^-1 = b^-1
//   // a^-1 * residual^-1 = b^-1
//   // residual  = a * b^-1
//   x.translation() = b.translation() - a.translation();
//   x.rotation() = -quat_residual(a.orientation() * b.orientation().inverse());
//
//   return x;
// }

template <class T> struct PoseResidualLinearization {
  T vec_f;
  T d_vec_f;
  Quaternion<T> va;
  Vector3<T> pa;
};

TRACTOR_OP(pose_residual, (const Pose<T> &a), { return pose_residual(a); })
TRACTOR_D(prepare, pose_residual,
          (const Pose<T> &pose, const Twist<T> &vx,
           PoseResidualLinearization<T> &v),
          {
            auto &va = pose.orientation();

            // v.vec_f = T(2) * acos(va.w()) / sqrt(T(1) - va.w() * va.w());
            //
            // T d_va_w_r = T(1) - va.w() * va.w();
            // v.d_vec_f =
            //     T(2) * va.w() * acos(va.w()) / (d_va_w_r * sqrt(d_va_w_r)) -
            //     T(2) / (T(1) - va.w() * va.w());

            v.vec_f = quat_residual_factor(va.w());
            v.d_vec_f = quat_residual_gradient(va.w());

            v.va = va;

            v.pa = pose.position();
          })
TRACTOR_D(forward, pose_residual,
          (const PoseResidualLinearization<T> &v, //
           const Twist<T> &twist_a, Twist<T> &twist_x),
          {
            auto &vec_f = v.vec_f;
            auto &d_vec_f = v.d_vec_f;
            auto &va = v.va;

            auto &da = twist_a.rotation();

            Quaternion<T> dqda = Quaternion<T>(da.x() * T(0.5), da.y() * T(0.5),
                                               da.z() * T(0.5), T(0));

            auto dqa = dqda * va;

            T d_vec_f_w = d_vec_f * dqa.w();

            T d_vec_x = dqa.x() * vec_f + va.x() * d_vec_f_w;
            T d_vec_y = dqa.y() * vec_f + va.y() * d_vec_f_w;
            T d_vec_z = dqa.z() * vec_f + va.z() * d_vec_f_w;

            twist_x.rotation() = Vector3<T>(d_vec_x, d_vec_y, d_vec_z);

            twist_x.translation() = twist_a.translation();
          })
TRACTOR_D(reverse, pose_residual,
          (const PoseResidualLinearization<T> &v, //
           Twist<T> &twist_a, const Twist<T> &twist_x),
          {
            auto &vec_f = v.vec_f;
            auto &d_vec_f = v.d_vec_f;
            auto &va = v.va;

            auto &dx = twist_x.rotation();

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

            twist_a.rotation().x() = dqda.x() * T(0.5);
            twist_a.rotation().y() = dqda.y() * T(0.5);
            twist_a.rotation().z() = dqda.z() * T(0.5);

            twist_a.translation() = twist_x.translation();
          })

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

  // Quaternion<T> qb =
  //     normalized(normalized(Quaternion<T>(b.rotation().x() * T(0.5), //
  //                                         b.rotation().y() * T(0.5), //
  //                                         b.rotation().z() * T(0.5), //
  //                                         T(1.0)                     //
  //                                         )));
  // Pose<T> pb = Pose(b.translation(), qb);
  // return pb * a;

  return Pose<T>(a.position() + b.translation(),
                 a.orientation() + b.rotation());
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
//
// template <class T> struct AddPoseTwistLinearization {
//   Quaternion<T> bqn;
//   Vector3<T> at;
//   T bqfh;
// };
//
// TRACTOR_D_T(prepare, pose_twist, add,
//             (const Pose<T> &a, const Twist<T> &b, const Pose<T> &x,
//              AddPoseTwistLinearization<T> &v),
//             {
//               Quaternion<T> bq = Quaternion<T>(b.rotation().x() * T(0.5), //
//                                                b.rotation().y() * T(0.5), //
//                                                b.rotation().z() * T(0.5), //
//                                                T(1)                       //
//               );
//               T bqf = T(1) / norm(bq);
//               v.bqn = normalized(bq);
//               v.bqfh = bqf * T(0.5);
//               v.at = a.translation();
//             })
//
// TRACTOR_D_T(forward, pose_twist, add,
//             (const AddPoseTwistLinearization<T> &v, const Twist<T> &da,
//              const Twist<T> &db, Twist<T> &dx),
//             {
//               Vector3 dqb = quat_pack_forward(v.bqn, T(1),
//                                               Quaternion<T>( //
//                                                   db.rotation().x() * v.bqfh,
//                                                   // db.rotation().y() *
//                                                   v.bqfh, //
//                                                   db.rotation().z() * v.bqfh,
//                                                   // T(0) //
//                                                   ));
//
//               dx.rotation() = dqb + v.bqn * da.rotation();
//
//               // dx.translation() = db.translation() + v.bqn *
//               da.translation()
//               // +
//               //                    cross(dqb, v.bqn * v.at);
//
//               dx.translation() = da.translation() + db.translation();
//
//               // dx = va * db + cross(da, va * vb);
//
//               // auto qvb = Quaternion<T>(vb.rotation().x() * T(0.5), //
//               //                          vb.rotation().y() * T(0.5), //
//               //                          vb.rotation().z() * T(0.5), //
//               //                          T(1)                        //
//               // );
//               //
//               // Quaternion<T> qvbn = normalized(qvb);
//               // T qvbf = T(1) / norm(qvb);
//               //
//               // Quaternion<T> qdbn =
//               //     Quaternion<T>(db.rotation().x() * qvbf * T(0.5), //
//               //                   db.rotation().y() * qvbf * T(0.5), //
//               //                   db.rotation().z() * qvbf * T(0.5), //
//               //                   T(0)                               //
//               //     );
//               //
//               // auto db_r = quat_pack_forward(qvbn, qdbn);
//               //
//               // auto v_ar = va.orientation();
//               // auto v_arbt = va.orientation() * vb.translation();
//               // auto v_arinv = va.orientation().inverse();
//               //
//               // dx.translation() = da.translation() + v_ar *
//               db.translation() +
//               //                    cross(da.rotation(), v_arbt);
//               //
//               // dx.rotation() = v_ar * db_r + da.rotation();
//             })
// TRACTOR_D_T(reverse, pose_twist, add,
//             (const AddPoseTwistLinearization<T> &v,
//              // const Pose<T> &va, const Twist<T> &vb, const Pose<T> &vx,
//              Twist<T> &da, Twist<T> &db, const Twist<T> &dx),
//             {
//               // a = x;
//               // b = x;
//
//               da.rotation() = v.bqn.inverse() * dx.rotation();
//
//               Vector3<T> rot = dx.rotation();
//
//               // rot += cross(v.bqn * v.at, dx.translation());
//
//               Quaternion<T> qdb = quat_pack_reverse(v.bqn, T(1), rot *
//               v.bqfh);
//
//               db.rotation().x() = qdb.x();
//               db.rotation().y() = qdb.y();
//               db.rotation().z() = qdb.z();
//
//               db.translation() = dx.translation();
//
//               // da.translation() = v.bqn.inverse() * dx.translation();
//               da.translation() = dx.translation();
//             })

template <class T> struct AddPoseTwistLinearization {
  Vector3<T> at;
  Vector3<T> b;
  Quaternion<T> quat;
  T sgradn;
  T f;
};

TRACTOR_D_T(prepare, pose_twist, add,
            (const Pose<T> &a, const Twist<T> &b, const Pose<T> &x,
             AddPoseTwistLinearization<T> &v),
            {
              T angle = norm(b.rotation());
              T f = sinc(angle * T(0.5)) * T(0.5);
              T c = cos(angle * T(0.5));
              Quaternion<T> quat;
              quat.x() = b.rotation().x() * f;
              quat.y() = b.rotation().y() * f;
              quat.z() = b.rotation().z() * f;
              quat.w() = c;
              v.b = b.rotation();
              v.quat = quat;
              v.sgradn = quat_vec_add_gradient(angle);
              v.f = f;

              v.at = a.translation();
            })

TRACTOR_D_T(forward, pose_twist, add,
            (const AddPoseTwistLinearization<T> &v, const Twist<T> &da,
             const Twist<T> &db, Twist<T> &dx),
            {
              T d_angle = dot(v.b, db.rotation());
              T d_f = d_angle * v.sgradn;
              T d_c = d_angle * v.f * T(-0.5);

              Quaternion<T> d_quat;
              d_quat.x() = v.b.x() * d_f + db.rotation().x() * v.f;
              d_quat.y() = v.b.y() * d_f + db.rotation().y() * v.f;
              d_quat.z() = v.b.z() * d_f + db.rotation().z() * v.f;
              d_quat.w() = d_c;

              Vector3 d_vec = quat_pack_forward(v.quat, T(1), d_quat);

              dx.rotation() = d_vec + v.quat * da.rotation();

              dx.translation() = da.translation() + db.translation();
            })
TRACTOR_D_T(reverse, pose_twist, add,
            (const AddPoseTwistLinearization<T> &v,
             // const Pose<T> &va, const Twist<T> &vb, const Pose<T> &vx,
             Twist<T> &da, Twist<T> &db, const Twist<T> &dx),
            {
              da.rotation() = v.quat.inverse() * dx.rotation();

              Quaternion<T> d_quat =
                  quat_pack_reverse(v.quat, T(1), dx.rotation());

              db.rotation().x() = d_quat.x() * v.f;
              db.rotation().y() = d_quat.y() * v.f;
              db.rotation().z() = d_quat.z() * v.f;

              T d_f = v.b.x() * d_quat.x() + v.b.y() * d_quat.y() +
                      v.b.z() * d_quat.z();

              T d_c = d_quat.w();

              T d_angle = d_f * v.sgradn + d_c * v.f * T(-0.5);

              db.rotation() += v.b * d_angle;

              db.translation() = dx.translation();
              da.translation() = dx.translation();
            })

// TRACTOR_OP(pose_trust_region_constraint, (const Pose<T> &a, const T &tr),
//            { return T(0); })
// TRACTOR_D(prepare, pose_trust_region_constraint,
//           (const Pose<T> &a, const T &tr, const T &x), {})
// TRACTOR_D(forward, pose_trust_region_constraint,
//           (const Twist<T> &a, const T &tr, T &x), { x = T(0); })
// TRACTOR_D(reverse, pose_trust_region_constraint,
//           (Twist<T> & a, T &tr, const T &x), {
//             a.setZero();
//             tr = T(0);
//           })
// TRACTOR_D(project, pose_trust_region_constraint,
//           (const Pose<T> &a, const T &tr, const Twist<T> &da, const T
//           &padding,
//            Twist<T> &dx),
//           {
//             dx = da;
//             // for (size_t i = 0; i < 6; i++) {
//             //  dx[i] = std::max(-tr, std::min(tr, da[i]));
//             //}
//           })
// TRACTOR_D(barrier_init, pose_trust_region_constraint,
//           (const Pose<T> &a, const T &tr, const Twist<T> &da, Twist<T> &dx,
//            Twist<T> &ddx),
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
// TRACTOR_D(barrier_step, pose_trust_region_constraint,
//           (const Twist<T> &dda, const Twist<T> &da, Twist<T> &dx), {
//             for (size_t i = 0; i < 6; i++) {
//               dx[i] = dda[i] * da[i];
//             }
//           })
// TRACTOR_D(barrier_diagonal, pose_trust_region_constraint,
//           (const Twist<T> &dda, Twist<T> &ddx), { ddx = dda; })

} // namespace tractor
