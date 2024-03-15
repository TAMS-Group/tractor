// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/recorder.h>
#include <tractor/geometry/matrix3_ops.h>
#include <tractor/geometry/ops.h>
#include <tractor/geometry/pose_ops.h>
#include <tractor/geometry/quaternion_ops.h>
#include <tractor/geometry/twist_ops.h>
#include <tractor/geometry/vector3_ops.h>

namespace tractor {

template <class ScalarType>
struct GeometryFastBase {
  typedef ScalarType Scalar;
  typedef ScalarType Value;
  typedef tractor::Pose<ScalarType> Pose;
  typedef tractor::Vector3<ScalarType> Vector3;
  typedef tractor::Quaternion<ScalarType> Orientation;
  typedef tractor::Twist<ScalarType> Twist;
  typedef tractor::Matrix3<ScalarType> Matrix3;
};

template <class ValueType>
struct GeometryFastBase<Var<ValueType>> {
  typedef Var<ValueType> Scalar;
  typedef ValueType Value;
  typedef Var<tractor::Pose<ValueType>> Pose;
  typedef Var<tractor::Vector3<ValueType>> Vector3;
  typedef Var<tractor::Quaternion<ValueType>> Orientation;
  typedef Var<tractor::Twist<ValueType>> Twist;
  typedef Var<tractor::Matrix3<ValueType>> Matrix3;
};

template <class ScalarType>
struct GeometryScalarBase {
  typedef ScalarType Scalar;
  typedef ScalarType Value;
  typedef tractor::Pose<ScalarType> Pose;
  typedef tractor::Vector3<ScalarType> Vector3;
  typedef tractor::Quaternion<ScalarType> Orientation;
  typedef tractor::Twist<ScalarType> Twist;
  typedef tractor::Matrix3<ScalarType> Matrix3;
};

template <class ValueType>
struct GeometryScalarBase<Var<ValueType>> {
  typedef Var<ValueType> Scalar;
  typedef ValueType Value;
  typedef tractor::Pose<Var<ValueType>> Pose;
  typedef tractor::Vector3<Var<ValueType>> Vector3;
  typedef tractor::Quaternion<Var<ValueType>> Orientation;
  typedef tractor::Twist<Var<ValueType>> Twist;
  typedef tractor::Matrix3<Var<ValueType>> Matrix3;
};

template <class Base>
struct GeometryImpl : Base {
  typedef typename Base::Scalar Scalar;
  typedef typename Base::Value Value;
  typedef typename Base::Pose Pose;
  typedef typename Base::Vector3 Vector3;
  typedef typename Base::Orientation Orientation;
  typedef typename Base::Twist Twist;
  typedef typename Base::Matrix3 Matrix3;

  static auto Vector3Zero() { return Vector3(tractor::Vector3<Value>::Zero()); }
  static auto TwistZero() { return Twist(tractor::Twist<Value>::Zero()); }
  static auto ScalarZero() { return Scalar(Value(0)); }
  static auto PoseIdentity() { return Pose(tractor::Pose<Value>::Identity()); }
  static auto Matrix3Zero() { return Matrix3(tractor::Matrix3<Value>::Zero()); }
  static auto Matrix3Identity() {
    return Matrix3(tractor::Matrix3<Value>::Identity());
  }
  static auto OrientationIdentity() {
    return Orientation(tractor::Quaternion<Value>::Identity());
  }
  static auto UnitX() { return pack(Value(1), Value(0), Value(0)); }
  static auto UnitY() { return pack(Value(0), Value(1), Value(0)); }
  static auto UnitZ() { return pack(Value(0), Value(0), Value(1)); }

  static Orientation angleAxisOrientation(const Scalar &angle,
                                          const Vector3 &axis) {
    return angle_axis_quat(angle, axis);
  }

  static Pose angleAxisPose(const Scalar &angle, const Vector3 &axis) {
    return angle_axis_pose(angle, axis);
  }

  static Pose angleAxisPose(const Pose &parent, const Scalar &angle,
                            const Vector3 &axis) {
    // return parent * angleAxisPose(angle, axis);
    return pose_angle_axis_pose(parent, angle, axis);
  }

  static Pose identityPose() { return Pose(tractor::Pose<Value>::Identity()); }

  static Pose translationPose(const Vector3 &pos) {
    return translation_pose(pos);
  }

  static Pose translationPose(const Pose &parent, const Vector3 &pos) {
    // return parent * translationPose(pos);
    return pose_translate(parent, pos);
  }

  static Pose translationPose(const Pose &parent, const Scalar &x,
                              const Scalar &y, const Scalar &z) {
    Vector3 translation;
    vec3_pack(x, y, z, translation);
    return pose_translate(parent, translation);
  }

  // static Matrix3 inverse(const Matrix3 &mat) { return tractor::inverse(mat);
  // }

  static Twist translationTwist(const Vector3 &translation) {
    return translation_twist(translation);
  }

  static Twist twist(const Vector3 &translation, const Vector3 &rotation) {
    return make_twist(translation, rotation);
  }

  static Twist pack(const Vector3 &translation, const Vector3 &rotation) {
    return make_twist(translation, rotation);
  }

  static Pose orientationPose(const Orientation &orientation) {
    return orientation_pose(orientation);
  }

  static Vector3 translation(const Pose &pose) {
    return pose_translation(pose);
  }

  static Vector3 position(const Pose &pose) { return pose_translation(pose); }

  static Orientation orientation(const Pose &pose) {
    return pose_orientation(pose);
  }

  static Vector3 translation(const Twist &twist) {
    return twist_translation(twist);
  }

  static Vector3 rotation(const Twist &twist) { return twist_rotation(twist); }

  static Vector3 pack(const Scalar &x, const Scalar &y, const Scalar &z) {
    Vector3 ret;
    vec3_pack(x, y, z, ret);
    return ret;
  }

  static void unpack(const Vector3 &v, Scalar &x, Scalar &y, Scalar &z) {
    vec3_unpack(v, x, y, z);
  }

  static void unpack(const Orientation &v, Scalar &x, Scalar &y, Scalar &z,
                     Scalar &w) {
    quat_unpack(v, x, y, z, w);
  }

  static void unpack(const Pose &pose, Scalar &px, Scalar &py, Scalar &pz,
                     Scalar &qx, Scalar &qy, Scalar &qz, Scalar &qw) {
    unpack(translation(pose), px, py, pz);
    unpack(orientation(pose), qx, qy, qz, qw);
  }

  static void unpack(const Twist &twist, Vector3 &t, Vector3 &r) {
    t = translation(twist);
    r = rotation(twist);
  }

  static void unpack(const Pose &pose, Vector3 &p, Orientation &o) {
    p = position(pose);
    o = orientation(pose);
  }

  // static void unpack(const Twist &twist, Scalar &px, Scalar &py, Scalar &pz,
  //                    Scalar &rx, Scalar &ry, Scalar &rz) {
  //   twist_unpack(twist, px, py, pz, rx, ry, rz);
  // }

  static Pose pack(const Scalar &px, const Scalar &py, const Scalar &pz,
                   const Scalar &qx, const Scalar &qy, const Scalar &qz,
                   const Scalar &qw) {
    Vector3 p;
    vec3_pack(px, py, pz, p);
    Orientation q;
    quat_pack(qx, qy, qz, qw, q);
    // return translationPose(p) * orientationPose(q);
    return make_pose(p, q);
  }

  static Orientation pack(const Scalar &qx, const Scalar &qy, const Scalar &qz,
                          const Scalar &qw) {
    Orientation q;
    quat_pack(qx, qy, qz, qw, q);
    return q;
  }

  static Pose pack(const Vector3 &position, const Orientation &orientation) {
    return make_pose(position, orientation);
  }

  static Orientation inverse(const Orientation &q) { return quat_inverse(q); }

  static Pose inverse(const Pose &pose) {
    Orientation ret_orientation = inverse(orientation(pose));
    Vector3 ret_translation = ret_orientation * -translation(pose);
    return translationPose(ret_translation) * orientationPose(ret_orientation);
  }

  static Vector3 residual(const Orientation &a) { return quat_residual(a); }

  static Vector3 residual(const Orientation &a, const Orientation &b) {
    return residual(inverse(a) * b);
  }

  static Twist residual(const Pose &a) {
    // return pose_residual(a);
    return twist(position(a), residual(orientation(a)));
  }

  static Twist residual(const Pose &a, const Pose &b) {
    return residual(inverse(a) * b);
  }

  static Scalar dot(const Vector3 &a, const Vector3 &b) {
    return tractor::dot(a, b);
  }

  static Vector3 cross(const Vector3 &a, const Vector3 &b) {
    return tractor::cross(a, b);
  }

  static Scalar norm(const Vector3 &a) { return tractor::norm(a); }
  static Vector3 normalized(const Vector3 &a) { return tractor::normalized(a); }
  static Scalar squaredNorm(const Vector3 &a) {
    return tractor::squaredNorm(a);
  }

  template <class T>
  static Vector3 importVector3(const T &p) {
    return Vector3(
        tractor::Vector3<Value>(Value(p.x()), Value(p.y()), Value(p.z())));
  }

  template <class T>
  static Orientation importQuaternion(const T &q) {
    return Orientation(tractor::Quaternion<Value>(Value(q.x()), Value(q.y()),
                                                  Value(q.z()), Value(q.w())));
  }

  template <class T>
  static Matrix3 importMatrix3(const T &p) {
    typename tractor::Matrix3<Value> ret;
    for (size_t row = 0; row < 3; row++) {
      for (size_t col = 0; col < 3; col++) {
        ret(row, col) = Value(p(row, col));
      }
    }
    return Matrix3(ret);
  }

  template <class T, class R>
  static Twist importTwist(const T &p, const R &r) {
    tractor::Twist<Value> ret;
    ret.translation().x() = Value(p.x());
    ret.translation().y() = Value(p.y());
    ret.translation().z() = Value(p.z());
    ret.rotation().x() = Value(r.x());
    ret.rotation().y() = Value(r.y());
    ret.rotation().z() = Value(r.z());
    return Twist(ret);
  }

  template <class T>
  static Value importValue(const T &v) {
    return Value(v);
  }

  template <class T>
  static Scalar importScalar(const T &value) {
    return Scalar(Value(value));
  }
};

template <class ScalarType>
struct GeometryFast : GeometryImpl<GeometryFastBase<ScalarType>> {};

template <class ScalarType>
struct GeometryScalar : GeometryImpl<GeometryScalarBase<ScalarType>> {};

}  // namespace tractor
