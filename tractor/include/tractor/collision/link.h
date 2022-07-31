// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/collision/shape.h>

#include <memory>

namespace tractor {

class CollisionLinkBase {
public:
  virtual void
  addConvexPolyhedron(const std::vector<Vector3<double>> &points,
                      const std::vector<Plane<double>> &planes) = 0;
  virtual void addSphere(const Vector3<double> &center, double radius) = 0;
  virtual void addCylinder(double radius, double length) = 0;
};

template <class Scalar> class CollisionLink : public CollisionLinkBase {
  std::string _name;
  std::vector<std::shared_ptr<const CollisionShape<Scalar>>> _shapes;

public:
  CollisionLink() {}
  CollisionLink(const std::string &name) : _name(name) {}
  const auto &shapes() const { return _shapes; }
  const std::string &name() const { return _name; }
  virtual void
  addConvexPolyhedron(const std::vector<Vector3<double>> &points,
                      const std::vector<Plane<double>> &planes) override;
  virtual void addSphere(const Vector3<double> &center, double radius) override;
  virtual void addCylinder(double radius, double length) override;
};

} // namespace tractor
