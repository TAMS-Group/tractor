// (c) 2022 Philipp Ruppel

#pragma once

#include "shape.h"

namespace shapes {
class Shape;
class Mesh;
} // namespace shapes

namespace tractor {

struct CollisionRequest {
  Eigen::Isometry3d pose_a = Eigen::Isometry3d::Identity();
  const CollisionShape *shape_a = nullptr;
  Eigen::Isometry3d pose_b = Eigen::Isometry3d::Identity();
  const CollisionShape *shape_b = nullptr;
};

class CollisionEngine {
public:
  virtual std::shared_ptr<CollisionShape>
  create(const std::string &name, const Eigen::Affine3d &pose,
         const shapes::Shape *shape) const = 0;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const = 0;

  virtual ~CollisionEngine() {}
};

} // namespace tractor
