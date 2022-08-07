// (c) 2020-2022 Philipp Ruppel

#pragma once

//#define EIGEN_NO_STATIC_ASSERT

#include <tractor/collision/link.h>
#include <tractor/core/profiler.h>

namespace tractor {

namespace internal {

class CollisionSupportInterface {
public:
  virtual void support(double dx, double dy, double dz, double &px, double &py,
                       double &pz) const = 0;
  virtual void center(double &px, double &py, double &pz) const = 0;
};
struct CollisionResult {
  double ax = 0, ay = 0, az = 0;
  double bx = 0, by = 0, bz = 0;
  double nx = 1, ny = 0, nz = 0;
  double d = 0;
};
void doCollisionQuery(const CollisionSupportInterface &a,
                      const CollisionSupportInterface &b,
                      CollisionResult &result);

template <class Scalar>
class CollisionShapeSupport
    : public tractor::internal::CollisionSupportInterface {
  Pose<Scalar> _pose;
  const CollisionShape<Scalar> *_shape = nullptr;

public:
  CollisionShapeSupport(const Pose<Scalar> &pose,
                        const CollisionShape<Scalar> *shape)
      : _pose(pose), _shape(shape) {}
  virtual void support(double dx, double dy, double dz, double &px, double &py,
                       double &pz) const override {
    if (!_shape) {
      throw std::runtime_error("shape is null");
    }
    Vector3<Scalar> d = Vector3<Scalar>(Scalar(dx), Scalar(dy), Scalar(dz));
    d = _pose.orientation().inverse() * d;
    Vector3<Scalar> p;
    _shape->support(d, p);
    p = _pose * p;
    px = double(p.x());
    py = double(p.y());
    pz = double(p.z());
  }
  virtual void center(double &px, double &py, double &pz) const override {
    auto p = _shape->center();
    p = _pose * p;
    px = p.x();
    py = p.y();
    pz = p.z();
  }
};

template <class Scalar>
class PointSupport : public tractor::internal::CollisionSupportInterface {
  Vector3<Scalar> _point;

public:
  PointSupport(const Vector3<Scalar> &point) : _point(point) {}
  virtual void support(double dx, double dy, double dz, double &px, double &py,
                       double &pz) const override {
    px = double(_point.x());
    py = double(_point.y());
    pz = double(_point.z());
  }
  virtual void center(double &px, double &py, double &pz) const override {
    px = double(_point.x());
    py = double(_point.y());
    pz = double(_point.z());
  }
};

} // namespace internal

} // namespace tractor
