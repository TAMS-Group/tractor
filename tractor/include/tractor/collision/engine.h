// (c) 2022 Philipp Ruppel

#pragma once

#include <Eigen/Dense>
#include <memory>

namespace shapes {
class Shape;
}

namespace tractor {

class CollisionEngine;

struct CollisionShape {
  virtual const CollisionEngine *engine() const = 0;
  virtual ~CollisionShape() {}
};

struct CollisionRequest {
  Eigen::Isometry3d pose_a = Eigen::Isometry3d::Identity();
  const CollisionShape *shape_a = nullptr;
  Eigen::Isometry3d pose_b = Eigen::Isometry3d::Identity();
  const CollisionShape *shape_b = nullptr;
};

struct CollisionResponse {
  Eigen::Vector3d point_a = Eigen::Vector3d::Zero();
  Eigen::Vector3d point_b = Eigen::Vector3d::Zero();
  Eigen::Vector3d normal = Eigen::Vector3d::Zero();
  double distance = 0.0;
};

class CollisionEngine {
public:
  virtual std::shared_ptr<CollisionShape>
  create(const Eigen::Affine3d &pose, const shapes::Shape *shape) const = 0;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const = 0;

  virtual ~CollisionEngine() {}
};

} // namespace tractor
