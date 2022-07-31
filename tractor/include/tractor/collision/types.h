// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/collision/shape.h>

namespace tractor {

template <class Scalar>
class CollisionSphereShape : public CollisionShape<Scalar> {
  Scalar _radius = 0;

public:
  CollisionSphereShape(const Vector3<Scalar> &center, const Scalar &radius) {
    this->_center = center;
    _radius = radius;
  }

  virtual void support(const Vector3<Scalar> &direction,
                       Vector3<Scalar> &point) const override {
    point = this->_center + normalized(direction) * _radius;
  }
};

template <class Scalar>
class CollisionCylinderShape : public CollisionShape<Scalar> {
  Scalar _radius = 0;
  Scalar _length = 0;

  static inline Scalar _sign(const Scalar &v) {
    if (v < 0) {
      return Scalar(-1);
    }
    if (v > 0) {
      return Scalar(1);
    }
    return Scalar(0);
  }

public:
  CollisionCylinderShape(const Scalar &radius, const Scalar &length) {
    _radius = radius;
    _length = length;
  }

  auto &radius() const { return _radius; }
  auto &length() const { return _length; }

  virtual void support(const Vector3<Scalar> &direction,

                       Vector3<Scalar> &point) const override {
    Vector3<Scalar> axis = Vector3<Scalar>(Scalar(0), Scalar(0), Scalar(1));
    Vector3<Scalar> a = direction - dot(axis, direction) * axis;
    if (norm(a) != Scalar(0)) {
      point = Scalar(_sign(direction.z())) * (_length * Scalar(0.5)) * axis +
              _radius * normalized(a);
    } else {
      point = Scalar(_sign(direction.z())) * (_length * Scalar(0.5)) * axis;
    }
  }
};

template <class Scalar>
class ConvexPolyhedralCollisionShape : public CollisionShape<Scalar> {
  std::vector<Vector3<Scalar>> _points;
  std::vector<Plane<Scalar>> _planes;

public:
  ConvexPolyhedralCollisionShape(const std::vector<Vector3<Scalar>> &points,
                                 const std::vector<Plane<Scalar>> &planes)
      : _points(points), _planes(planes) {
    if (points.empty()) {
      throw std::invalid_argument("points");
    }
    if (planes.empty()) {
      throw std::invalid_argument("planes");
    }
    this->_center = Vector3<Scalar>::Zero();
    for (auto &p : points) {
      this->_center += p;
    }
    this->_center *= Scalar(1.0 / points.size());
  }

  virtual void support(const Vector3<Scalar> &direction,
                       Vector3<Scalar> &point) const override {
    point = _points.front();
    for (auto &p : _points) {
      if (dot(p, direction) > dot(point, direction)) {
        point = p;
      }
    }
  }

  const std::vector<Vector3<Scalar>> &points() const { return _points; }

  auto &planes() const { return _planes; }

  virtual void project(const Vector3<Scalar> &point, Vector3<Scalar> &normal,
                       Scalar &distance) override {
    bool inside = true;
    for (size_t i = 0; i < _planes.size(); i++) {
      auto &plane = _planes[i];
      auto dist = plane.signedDistance(point);
      if (dist > 0) {
        inside = false;
      }
      if (i == 0 || dist > distance) {
        distance = dist;
        normal = -plane.normal();
      }
    }
    if (!inside) {
      CollisionShape<Scalar>::project(point, normal, distance);
    }
  }
};

} // namespace tractor
