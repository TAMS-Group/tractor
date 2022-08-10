// (c) 2022 Philipp Ruppel

#pragma once

#include "inertia.h"

namespace tractor {

template <class Geometry> class RigidBody {
  Inertia<Geometry> _inertia;
  typename Geometry::Pose _pose = Geometry::PoseIdentity();
  typename Geometry::Vector3 _global_linear_momentum = Geometry::Vector3Zero();
  typename Geometry::Vector3 _global_angular_momentum = Geometry::Vector3Zero();
  bool _has_force = false;
  typename Geometry::Vector3 _sum_force = Geometry::Vector3Zero();
  bool _has_torque = false;
  typename Geometry::Vector3 _sum_torque = Geometry::Vector3Zero();
  bool _has_damping = false;
  typename Geometry::Scalar _sum_linear_damping = Geometry::ScalarZero();
  typename Geometry::Scalar _sum_angular_damping = Geometry::ScalarZero();

public:
  RigidBody(const typename Geometry::Pose &pose,
            const Inertia<Geometry> &inertia)
      : _inertia(inertia), _pose(pose) {}

  auto &inertia() const { return _inertia; }

  auto &pose() const { return _pose; }

  void applyAcceleration(const typename Geometry::Vector3 &acceleration) {
    _has_force = true;
    _sum_force += acceleration * _inertia.mass();
  }

  void applyForce(const typename Geometry::Vector3 &force) {
    _has_force = true;
    _sum_force += force;
  }

  void applyForce(const typename Geometry::Vector3 &point,
                  const typename Geometry::Vector3 &force) {
    _has_force = true;
    _has_torque = true;
    _sum_force += force;
    _sum_torque += cross(point - Geometry::translation(_pose), force);
  }

  void applyDamping(const typename Geometry::Scalar &linear_damping,
                    const typename Geometry::Scalar &angular_damping) {
    _has_damping = true;
    _sum_linear_damping += linear_damping;
    _sum_angular_damping += angular_damping;
  }

  void integrate(const typename Geometry::Scalar &delta_time) {

    if (_has_force) {
      _has_force = false;
      _global_linear_momentum += _sum_force * delta_time;
      _sum_force = Geometry::Vector3Zero();
    }

    if (_has_torque) {
      _has_torque = false;
      _global_angular_momentum += _sum_torque * delta_time;
      _sum_torque = Geometry::Vector3Zero();
    }

    if (_has_damping) {
      _has_damping = false;
      auto neg_time = -delta_time;
      _global_linear_momentum *= exp(_sum_linear_damping * neg_time);
      _global_angular_momentum *= exp(_sum_angular_damping * neg_time);
      _sum_linear_damping = Geometry::ScalarZero();
      _sum_angular_damping = Geometry::ScalarZero();
    }

    auto orientation = Geometry::orientation(_pose);
    auto local_angular_momentum =
        Geometry::inverse(orientation) * _global_angular_momentum;
    auto global_angular_velocity =
        orientation * (_inertia.momentInverse() * local_angular_momentum);
    _global_angular_momentum = orientation * local_angular_momentum;

    auto global_linear_velocity =
        _global_linear_momentum * _inertia.massInverse() -
        cross(global_angular_velocity, orientation * _inertia.center());

    _pose += Geometry::twist(global_linear_velocity, global_angular_velocity) *
             delta_time;
  }
};

} // namespace tractor
