// (c) 2022 Philipp Ruppel

#pragma once

#include <Eigen/Dense>
#include <memory>

namespace tractor {

class CollisionEngine;

struct CollisionResponse {
  Eigen::Vector3d point_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d point_b = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  double distance = 0.0;
};

struct CollisionShape {
  virtual ~CollisionShape() {}
  virtual const CollisionEngine *engine() const = 0;
  virtual void project(const Eigen::Vector3d &point,
                       Eigen::Vector3d &closest_point,
                       Eigen::Vector3d &surface_normal) const = 0;
  virtual void sample(Eigen::Vector3d &point,
                      Eigen::Vector3d &normal) const = 0;
  virtual const std::string &name() const = 0;
};

} // namespace tractor
