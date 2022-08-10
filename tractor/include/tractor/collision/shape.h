// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>
#include <tractor/core/profiler.h>
#include <tractor/geometry/types.h>

#include <stdexcept>
#include <vector>

namespace tractor {

template <class Scalar> class CollisionShape {
protected:
  Vector3<Scalar> _center = Vector3<Scalar>::Zero();

  // static inline void _capturePlane(const Vector3<Scalar> &point,
  //                                  const Plane<Scalar> &plane,
  //                                  Eigen::Matrix<Scalar, 3, 1> &gradient,
  //                                  Eigen::Matrix<Scalar, 3, 3> &hessian) {
  //   Vector3<Scalar> gradient_vector =
  //       plane.normal() *
  //       (Scalar(1) / std::max(Scalar(0), -plane.signedDistance(point)));
  //   gradient += Eigen::Matrix<Scalar, 3, 1>(
  //       gradient_vector.x(), gradient_vector.y(), gradient_vector.z());
  //   hessian += gradient * gradient.transpose();
  // }

public:
  inline auto &center() const { return _center; }

  virtual void support(const Vector3<Scalar> &direction,
                       Vector3<Scalar> &point) const = 0;

  virtual Vector3<Scalar> support(const Vector3<Scalar> &direction) const {
    Vector3<Scalar> point;
    support(direction, point);
    return point;
  }

  virtual void project(const Vector3<Scalar> &point, Vector3<Scalar> &normal,
                       Scalar &distance);
};

} // namespace tractor
