// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>

#include "pose.h"

namespace tractor {

template <class T> static Vector3<T> toVector3(const Eigen::Vector3d &vec) {
  return Vector3<T>(T(vec.x()), T(vec.y()), T(vec.z()));
}

template <class T>
static Eigen::Vector3d toEigenVector3d(const Vector3<T> &vector) {
  return Eigen::Vector3d(vector.x(), vector.y(), vector.z());
}

template <class T>
static Eigen::Isometry3d toEigenIsometry3d(const Pose<T> &pose) {
  auto translation = pose.translation();
  auto orientation = pose.orientation();
  auto p = Eigen::Vector3d(translation.x(), translation.y(), translation.z());
  auto q = Eigen::Quaterniond(orientation.w(), orientation.x(), orientation.y(),
                              orientation.z());
  Eigen::Isometry3d ret = Eigen::Isometry3d(Eigen::AngleAxisd(q));
  ret.translation() = p;
  return ret;
}

} // namespace tractor
