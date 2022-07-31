// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/shape.h>

#include <tractor/collision/query.h>

namespace tractor {

template <class Scalar>
void CollisionShape<Scalar>::project(const Vector3<Scalar> &point,
                                     Vector3<Scalar> &normal,
                                     Scalar &distance) {
  auto a = tractor::internal::CollisionShapeSupport<Scalar>(
      Pose<Scalar>::Identity(), this);
  auto b = tractor::internal::PointSupport<Scalar>(point);
  tractor::internal::CollisionResult r;
  tractor::internal::doCollisionQuery(a, b, r);
  normal = Vector3<Scalar>(Scalar(r.nx), Scalar(r.ny), Scalar(r.nz));
  distance = r.d;
}

template class CollisionShape<double>;
template class CollisionShape<float>;

} // namespace tractor
