// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>

namespace tractor {

template <class ScalarType>
struct GeometryEigenMat {
  typedef ScalarType Scalar;
  typedef Eigen::Matrix<Scalar, 3, 1, Eigen::DontAlign> Vector3;
  typedef Eigen::Quaternion<Scalar, Eigen::DontAlign> Orientation;
  typedef Eigen::Transform<Scalar, 3, Eigen::Isometry, Eigen::DontAlign> Pose;
  typedef Eigen::Matrix<Scalar, 3, 3, Eigen::DontAlign> Matrix3;

  static auto ScalarZero() { return Scalar(0); }
  static auto Vector3Zero() { return Vector3::Zero(); }
  static auto OrientationZero() { return Orientation::Zero(); }
  static auto Matrix3Zero() { return Matrix3::Zero(); }

  static auto import(const Matrix3 &m) { return m; }

  static Pose angleAxisPose(const Scalar &angle, const Vector3 &axis) {
    return Pose(Eigen::AngleAxis<Scalar>(angle, axis));
  }

  static Pose angleAxisPose(const Pose &parent, const Scalar &angle,
                            const Vector3 &axis) {
    return parent * Pose(Eigen::AngleAxis<Scalar>(angle, axis));
  }

  static Pose pose(const Scalar &px, const Scalar &py, const Scalar &pz,
                   const Scalar &qx, const Scalar &qy, const Scalar &qz,
                   const Scalar &qw) {
    Orientation quat;
    quat.x() = qx;
    quat.y() = qy;
    quat.z() = qz;
    quat.w() = qw;

    Pose pose;
    pose.translation().x() = px;
    pose.translation().y() = py;
    pose.translation().z() = pz;
    pose.linear() = quat.toRotationMatrix();

    return pose;
  }

  static void unpack(const Pose &pose, Scalar &px, Scalar &py, Scalar &pz,
                     Scalar &qx, Scalar &qy, Scalar &qz, Scalar &qw) {
    Vector3 pos = pose.translation();
    px = pos.x();
    py = pos.y();
    pz = pos.z();

    Orientation quat(pose.linear());
    qx = quat.x();
    qy = quat.y();
    qz = quat.z();
    qw = quat.w();
  }

  static Pose identityPose() { Pose::Identity(); }

  static Pose translationPose(const Vector3 &pos) {
    Pose pose;
    pose.setIdentity();
    pose.translation() = pos;
    return pose;
  }

  static Pose translationPose(const Pose &parent, const Vector3 &pos) {
    return parent * translationPose(pos);
  }

  static Pose translationPose(const Pose &parent, const Scalar &x,
                              const Scalar &y, const Scalar &z) {
    return parent * translationPose(Vector3(x, y, z));
  }

  static Vector3 translation(const Pose &pose) { return pose.translation(); }

  static Orientation orientation(const Pose &pose) {
    // return Orientation(pose.linear());
    // return Orientation::Identity();
    auto &mat = pose.linear();
    Orientation ret;
    ret.w() = sqrt(Scalar(1) + mat(0, 0) + mat(1, 1) + mat(2, 2)) * Scalar(0.5);
    ret.x() = (mat(2, 1) - mat(1, 2)) / (Scalar(4) * ret.w());
    ret.y() = (mat(0, 2) - mat(2, 0)) / (Scalar(4) * ret.w());
    ret.z() = (mat(1, 0) - mat(0, 1)) / (Scalar(4) * ret.w());
    return ret;
  }

  static Orientation inverse(const Orientation &q) {
    return Orientation(q.w(), -q.x(), -q.y(), -q.z());
  }

  static Vector3 residual(const Orientation &a, const Orientation &b) {
    // return (a.inverse() * b).vec();
    return (inverse(a) * b).vec();
    // return Vector3::Zero();
  }

  template <class T, int Mode, int Flags>
  static Pose import(const Eigen::Transform<T, 3, Mode, Flags> &p) {
    Pose t = translationPose(
        Vector3(p.translation().x(), p.translation().y(), p.translation().z()));

    Eigen::AngleAxisd angle_axis(p.linear());

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
void parameter(
    Eigen::Transform<Scalar, 3, Eigen::Isometry, Eigen::DontAlign> &pose) {
  for (size_t row = 0; row < pose.rows(); row++) {
    for (size_t col = 0; col < pose.cols(); col++) {
      parameter(pose.matrix()(row, col));
    }
  }
}

}  // namespace tractor
