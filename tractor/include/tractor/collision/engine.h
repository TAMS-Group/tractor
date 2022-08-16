// (c) 2022 Philipp Ruppel

#pragma once

#include "shape.h"

#include <tractor/geometry/pose.h>

namespace shapes {
class Shape;
class Mesh;
} // namespace shapes

namespace tractor {

class ConvexCollisionMesh;

struct CollisionRequest {
  Pose3d pose_a = Pose3d::Identity();
  const CollisionShape *shape_a = nullptr;
  Pose3d pose_b = Pose3d::Identity();
  const CollisionShape *shape_b = nullptr;
};

class CollisionEngine {
public:
  virtual std::shared_ptr<ConvexCollisionMesh>
  createConvexMesh(const std::string &name, const shapes::Mesh *mesh) const = 0;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const = 0;

  virtual ~CollisionEngine() {}
};

} // namespace tractor
