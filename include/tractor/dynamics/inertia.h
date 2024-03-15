// 2022-2024 Philipp Ruppel

#pragma once

namespace tractor {

template <class Geometry>
class Inertia {
  typename Geometry::Vector3 _center = Geometry::Vector3Zero();
  typename Geometry::Scalar _mass = Geometry::ScalarZero();
  typename Geometry::Scalar _mass_inverse = Geometry::ScalarZero();
  typename Geometry::Matrix3 _moment = Geometry::Matrix3Zero();
  typename Geometry::Matrix3 _moment_inverse = Geometry::Matrix3Zero();

 public:
  Inertia() {}
  Inertia(const typename Geometry::Vector3 &center,
          const typename Geometry::Scalar &mass,
          const typename Geometry::Scalar &inverse_mass,
          const typename Geometry::Matrix3 &moment,
          const typename Geometry::Matrix3 &inverse_moment)
      : _center(center),
        _mass(mass),
        _mass_inverse(inverse_mass),
        _moment(moment),
        _moment_inverse(inverse_moment) {}
  const typename Geometry::Vector3 &center() const { return _center; }
  const typename Geometry::Scalar &mass() const { return _mass; }
  const typename Geometry::Scalar &massInverse() const { return _mass_inverse; }
  const typename Geometry::Matrix3 &moment() const { return _moment; }
  const typename Geometry::Matrix3 &momentInverse() const {
    return _moment_inverse;
  }
};

}  // namespace tractor
