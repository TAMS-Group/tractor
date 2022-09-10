// (c) 2022 Philipp Ruppel

#pragma once

#include "base.h"
#include "engine.h"

namespace tractor {

class BulletCollisionEngine : public CollisionEngine {

public:
  virtual std::shared_ptr<ConvexCollisionMesh>
  createConvexMesh(const std::string &name,
                   const shapes::Mesh *mesh) const override;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const override;

  virtual void collide(const ContinuousCollisionRequest &request,
                       ContinuousCollisionResponse &response) const override;

  // virtual void collide(const ContinuousCollisionManifoldRequest &request,
  //                      Vec3d &global_normal, Vec3d *local_points_a,
  //                      Vec3d *local_points_b) const override;
};

} // namespace tractor
