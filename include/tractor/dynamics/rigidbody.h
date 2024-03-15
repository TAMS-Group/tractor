// 2022-2024 Philipp Ruppel

#pragma once

#include "inertia.h"

namespace tractor {

template <class Geometry>
class RigidBody {
  typedef typename Geometry::Vector3 Vector3;
  typedef typename Geometry::Orientation Orientation;
  typedef typename Geometry::Scalar Scalar;

  Inertia<Geometry> _local_inertia;

  Vector3 _position = Geometry::Vector3Zero();
  Orientation _orientation = Geometry::OrientationIdentity();

  Vector3 _global_linear_momentum = Geometry::Vector3Zero();
  Vector3 _global_angular_momentum = Geometry::Vector3Zero();

  Vector3 _global_linear_velocity = Geometry::Vector3Zero();
  Vector3 _global_angular_velocity = Geometry::Vector3Zero();

  bool _has_force = false;
  Vector3 _sum_force = Geometry::Vector3Zero();

  bool _has_torque = false;
  Vector3 _sum_torque = Geometry::Vector3Zero();

  bool _has_damping = false;
  Scalar _sum_linear_damping = Geometry::ScalarZero();
  Scalar _sum_angular_damping = Geometry::ScalarZero();

  void _update(const Scalar &delta_time, bool integrate) {
    if (_has_force) {
      _has_force = false;
      if (integrate) {
        _global_linear_momentum += _sum_force * delta_time;
      }
      _sum_force = Geometry::Vector3Zero();
    }

    if (_has_torque) {
      _has_torque = false;
      if (integrate) {
        _global_angular_momentum += _sum_torque * delta_time;
      }
      _sum_torque = Geometry::Vector3Zero();
    }

    if (_has_damping) {
      _has_damping = false;
      if (integrate) {
        auto neg_time = -delta_time;
        _global_linear_momentum *= exp(_sum_linear_damping * neg_time);
        _global_angular_momentum *= exp(_sum_angular_damping * neg_time);
      }
      _sum_linear_damping = Geometry::ScalarZero();
      _sum_angular_damping = Geometry::ScalarZero();
    }

    auto local_angular_momentum =
        Geometry::inverse(_orientation) * _global_angular_momentum;
    auto local_angular_velocity =
        _local_inertia.momentInverse() * local_angular_momentum;
    _global_angular_velocity = _orientation * local_angular_velocity;

    _global_linear_velocity =
        _global_linear_momentum * _local_inertia.massInverse() -
        cross(_global_angular_velocity, _orientation * _local_inertia.center());

    if (integrate) {
      _position += _global_linear_velocity * delta_time;
      _orientation += _global_angular_velocity * delta_time;
      _global_angular_momentum = _orientation * local_angular_momentum;
    }
  }

 public:
  RigidBody(const typename Geometry::Pose &pose,
            const Inertia<Geometry> &inertia)
      : _local_inertia(inertia),
        _position(Geometry::position(pose)),
        _orientation(Geometry::orientation(pose)) {}

  RigidBody(const typename Geometry::Pose &pose,
            const Inertia<Geometry> &inertia,
            const typename Geometry::Twist &local_velocity)
      : _local_inertia(inertia),
        _position(Geometry::position(pose)),
        _orientation(Geometry::orientation(pose)) {
    auto local_linear_velocity = Geometry::translation(local_velocity);
    auto global_linear_velocity = _orientation * local_linear_velocity;
    _global_linear_momentum = global_linear_velocity * inertia.mass();

    auto local_angular_velocity = Geometry::rotation(local_velocity);
    auto local_angular_momentum =
        _local_inertia.moment() * local_angular_velocity;
    _global_angular_momentum = _orientation * local_angular_momentum;

    _update(Geometry::ScalarZero(), false);
  }

  auto &inertia() const { return _local_inertia; }

  Vector3 center() const {
    return _position + _orientation * _local_inertia.center();
  }

  auto localVelocity() const {
    auto orientation_inverse = Geometry::inverse(_orientation);
    return Geometry::pack(orientation_inverse * _global_linear_velocity,
                          orientation_inverse * _global_angular_velocity);
  }

  auto globalVelocity() const {
    return Geometry::pack(_global_linear_velocity, _global_angular_velocity);
  }

  auto globalLinearVelocity() const { return _global_linear_velocity; }
  auto globalAngularVelocity() const { return _global_angular_velocity; }

  auto pose() const { return Geometry::pack(_position, _orientation); }
  // void pose(const typename Geometry::Pose &pose) {
  //   _position = Geometry::position(pose);
  //   _orientation = Geometry::orientation(pose);
  // }

  const auto &position() const { return _position; }
  // void position(const Vector3 &position) {
  //   _position = position;
  // }

  const auto &orientation() const { return _orientation; }
  // void orientation(const Orientation &orientation) {
  //   _orientation = orientation;
  // }

  void applyAcceleration(const Vector3 &acceleration) {
    _has_force = true;
    _sum_force += acceleration * _local_inertia.mass();
  }

  void applyForce(const Vector3 &force) {
    _has_force = true;
    _sum_force += force;
  }

  void applyForce(const Vector3 &point, const Vector3 &force) {
    _has_force = true;
    _has_torque = true;
    _sum_force += force;
    _sum_torque += cross(point - center(), force);
  }

  void applyWrench(const typename Geometry::Pose &pose,
                   const typename Geometry::Twist &wrench) {
    auto point = Geometry::position(pose);
    auto orientation = Geometry::orientation(pose);
    applyForce(point, orientation * Geometry::translation(wrench));
    applyTorque(orientation * Geometry::rotation(wrench));
  }

  void applyDamping(const Scalar &linear_damping,
                    const Scalar &angular_damping) {
    _has_damping = true;
    _sum_linear_damping += linear_damping;
    _sum_angular_damping += angular_damping;
  }

  void applyTorque(const Vector3 &torque) {
    _has_torque = true;
    _sum_torque += torque;
  }

  void integrate(const Scalar &delta_time) { _update(delta_time, true); }
};

}  // namespace tractor
