// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/link.h>
#include <tractor/collision/types.h>

namespace tractor {

template <class Scalar>
void CollisionLink<Scalar>::addConvexPolyhedron(
    const std::vector<Vector3<double>> &points,
    const std::vector<Plane<double>> &planes) {
  if (points.empty() || planes.empty()) {
    return;
  }
  std::vector<Vector3<Scalar>> points2;
  for (auto &p : points) {
    points2.emplace_back(Scalar(p.x()), Scalar(p.y()), Scalar(p.z()));
  }
  std::vector<Plane<Scalar>> planes2;
  for (auto &plane : planes) {
    planes2.emplace_back(Vector3<Scalar>(Scalar(plane.normal().x()),
                                         Scalar(plane.normal().y()),
                                         Scalar(plane.normal().z())),
                         Scalar(plane.offset()));
  }
  _shapes.push_back(std::make_shared<ConvexPolyhedralCollisionShape<Scalar>>(
      points2, planes2));
}

template <class Scalar>
void CollisionLink<Scalar>::addSphere(const Vector3<double> &center,
                                      double radius) {
  _shapes.push_back(std::make_shared<CollisionSphereShape<Scalar>>(
      Vector3<Scalar>(Scalar(center.x()), Scalar(center.y()),
                      Scalar(center.z())),
      Scalar(radius)));
}

template <class Scalar>
void CollisionLink<Scalar>::addCylinder(double radius, double length) {
  _shapes.push_back(std::make_shared<CollisionCylinderShape<Scalar>>(
      Scalar(radius), Scalar(length)));
}

template class CollisionLink<double>;
template class CollisionLink<float>;

} // namespace tractor
