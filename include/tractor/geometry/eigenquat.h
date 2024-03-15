// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>

namespace tractor {

template <class ScalarType>
class PoseEigenQuat {
 public:
  typedef ScalarType Scalar;
  typedef Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> Vector3;
  typedef Eigen::Quaternion<Scalar, Eigen::DontAlign> Orientation;

  auto &translation() const { return _translation; }
  auto &translation() { return _translation; }
  auto &orientation() const { return _orientation; }
  auto &orientation() { return _orientation; }
  PoseEigenQuat operator*(const PoseEigenQuat &other) const {
    PoseEigenQuat ret;
    ret.translation() = translation() + orientation() * other.translation();
    ret.orientation() = orientation() * other.orientation();
    return ret;
  }
  static PoseEigenQuat Identity() {
    PoseEigenQuat ret;
    ret.translation().setZero();
    ret.orientation().setIdentity();
    return ret;
  }

 private:
  Vector3 _translation;
  Orientation _orientation;
};

template <class ScalarType>
struct GeometryEigenQuat {
  typedef ScalarType Scalar;
  typedef PoseEigenQuat<ScalarType> Pose;
  typedef typename Pose::Vector3 Vector3;
  typedef typename Pose::Orientation Orientation;
  typedef Eigen::Matrix<Scalar, 3, 3, Eigen::DontAlign> Matrix3;

  static auto ScalarZero() { return Scalar(0); }
  static auto Vector3Zero() { return Vector3::Zero(); }
  static auto OrientationZero() { return Orientation::Zero(); }
  static auto Matrix3Zero() { return Matrix3::Zero(); }

  static auto import(const Matrix3 &m) { return m; }

  static Pose angleAxisPose(const Scalar &angle, const Vector3 &axis) {
    Pose pose;
    pose.translation().setZero();
    Scalar s = sin(angle * Scalar(0.5));
    Scalar c = cos(angle * Scalar(0.5));
    pose.orientation().x() = axis.x() * s;
    pose.orientation().y() = axis.y() * s;
    pose.orientation().z() = axis.z() * s;
    pose.orientation().w() = c;
    return pose;
  }

  static void unpack(const Pose &pose, Scalar &px, Scalar &py, Scalar &pz,
                     Scalar &qx, Scalar &qy, Scalar &qz, Scalar &qw) {
    Vector3 pos = pose.translation();
    px = pos.x();
    py = pos.y();
    pz = pos.z();

    Orientation quat = pose.orientation();
    qx = quat.x();
    qy = quat.y();
    qz = quat.z();
    qw = quat.w();
  }

  static Pose angleAxisPose(const Pose &parent, const Scalar &angle,
                            const Vector3 &axis) {
    return parent * angleAxisPose(angle, axis);
  }

  static Pose pose(const Scalar &px, const Scalar &py, const Scalar &pz,
                   const Scalar &qx, const Scalar &qy, const Scalar &qz,
                   const Scalar &qw) {
    Pose pose;
    pose.translation().x() = px;
    pose.translation().y() = py;
    pose.translation().z() = pz;
    pose.orientation().x() = qx;
    pose.orientation().y() = qy;
    pose.orientation().z() = qz;
    pose.orientation().w() = qw;
    return pose;
  }

  static Pose identityPose() {
    Pose pose;
    pose.translation().setZero();
    pose.orientation().setIdentity();
    return pose;
  }

  static Pose translationPose(const Vector3 &pos) {
    Pose pose;
    pose.translation() = pos;
    pose.orientation().setIdentity();
    return pose;
  }

  static Pose translationPose(const Pose &parent, const Vector3 &pos) {
    return parent * translationPose(pos);
  }

  static Pose translationPose(const Pose &parent, const Scalar &x,
                              const Scalar &y, const Scalar &z) {
    return parent * translationPose(Vector3(x, y, z));
  }

  static const Vector3 &translation(const Pose &pose) {
    return pose.translation();
  }

  static const Orientation &orientation(const Pose &pose) {
    return pose.orientation();
  }

  static Orientation inverse(const Orientation &q) {
    return Orientation(q.w(), -q.x(), -q.y(), -q.z());
  }

  static Vector3 residual(const Orientation &a, const Orientation &b) {
    return (inverse(a) * b).vec();
  }

  template <class T, int Mode, int Flags>
  static Pose import(const Eigen::Transform<T, 3, Mode, Flags> &pose) {
    Pose t =
        translationPose(Vector3(pose.translation().x(), pose.translation().y(),
                                pose.translation().z()));

    Eigen::AngleAxisd angle_axis(pose.linear());

    Pose orientation =
        angleAxisPose(angle_axis.angle(),
                      Vector3(angle_axis.axis().x(), angle_axis.axis().y(),
                              angle_axis.axis().z()));

    return t * orientation;
  }

  template <class T, int Flags>
  static Vector3 import(const Eigen::Matrix<T, 3, 1, Flags> &p) {
    return Vector3(Scalar(p.x()), Scalar(p.y()), Scalar(p.z()));
  }
};

template <class Scalar>
void parameter(Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> &v) {
  parameter(v.x());
  parameter(v.y());
  parameter(v.z());
}

template <class Scalar>
void goal(const Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> &v) {
  goal(v.x());
  goal(v.y());
  goal(v.z());
}

template <class Scalar>
void parameter(PoseEigenQuat<Scalar> &pose) {
  parameter(pose.translation().x());
  parameter(pose.translation().y());
  parameter(pose.translation().z());
  parameter(pose.orientation().x());
  parameter(pose.orientation().y());
  parameter(pose.orientation().z());
  parameter(pose.orientation().w());
}

}  // namespace tractor
