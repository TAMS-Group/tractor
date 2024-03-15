// 2020-2024 Philipp Ruppel

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

template <class T>
struct PoseMulState {
  Quaternion<T> ar;
  Vector3<T> arbt;
  Quaternion<T> arinv;
};
TRACTOR_OP_T(pose, mul, (const Pose<T> &a, const Pose<T> &b), { return a * b; })
TRACTOR_D_T(prepare, pose, mul,
            (const Pose<T> &a, const Pose<T> &b, const Pose<T> &x,
             PoseMulState<T> &v),
            {
              v.ar = a.orientation();
              v.arbt = a.orientation() * b.translation();
              v.arinv = a.orientation().inverse();
            })
TRACTOR_D_T(forward, pose, mul,
            (const PoseMulState<T> &v, const Twist<T> &da, const Twist<T> &db,
             Twist<T> &dx),
            {
              dx.translation() = da.translation() + v.ar * db.translation() +
                                 cross(da.rotation(), v.arbt);
              dx.rotation() = v.ar * db.rotation() + da.rotation();
            })
TRACTOR_D_T(reverse, pose, mul,
            (const PoseMulState<T> &v, Twist<T> &da, Twist<T> &db,
             const Twist<T> &dx),
            {
              da.translation() = dx.translation();
              da.rotation() = cross(v.arbt, dx.translation());
              db.translation() = v.arinv * dx.translation();
              da.rotation() = da.rotation() + dx.rotation();
              db.rotation() = v.arinv * dx.rotation();
            })

template <class T>
struct PoseVec3MulState {
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
            d_pose.rotation() = v.axis_normalized * d_angle              //
                                + d_axis_p * v.sin_angle_by_axis_length  //
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

template <class T>
Twist<T> pose_residual(const Pose<T> &a) {
  Twist<T> x;
  x.translation() = a.translation();
  x.rotation() = quat_residual(a.orientation());
  return x;
}

template <class T>
struct PoseResidualLinearization {
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

            v.vec_f = quat_residual_factor(va.w());
            v.d_vec_f = quat_residual_gradient(va.w());

            v.va = va;

            v.pa = pose.position();
          })
TRACTOR_D(forward, pose_residual,
          (const PoseResidualLinearization<T> &v,  //
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
          (const PoseResidualLinearization<T> &v,  //
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

template <class T>
Pose<T> operator+(const Pose<T> &a, const Twist<T> &b) {
  return Pose<T>(a.position() + b.translation(),
                 a.orientation() + b.rotation());
}

TRACTOR_OP_T(pose_twist, add, (const Pose<T> &a, const Twist<T> &b), {
  Pose<T> ret = a + b;
  return ret;
})

template <class T>
struct AddPoseTwistLinearization {
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

}  // namespace tractor
