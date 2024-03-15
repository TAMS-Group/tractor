// 2022-2024 Philipp Ruppel

#pragma once

#include "base.h"
#include "engine.h"

namespace tractor {

class BulletCollisionEngine : public CollisionEngine {
 public:
  virtual std::shared_ptr<ConvexCollisionMesh> createConvexMesh(
      const std::string &name, const shapes::Mesh *mesh) const override;

  virtual std::shared_ptr<CollisionShape> createSphere(
      const std::string &name, const Pose3d &pose,
      double radius) const override;

  virtual std::shared_ptr<CollisionShape> createCylinder(
      const std::string &name, const Pose3d &pose, double length,
      double radius) const override;

  virtual std::shared_ptr<CollisionShape> createBox(
      const std::string &name, const Pose3d &pose,
      const Vec3d &size) const override;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const override;

  virtual void collide(const ContinuousCollisionRequest &request,
                       ContinuousCollisionResponse &response) const override;
};

}  // namespace tractor
