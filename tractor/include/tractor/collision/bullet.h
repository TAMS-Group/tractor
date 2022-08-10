// (c) 2022 Philipp Ruppel

#pragma once

#include "engine.h"

namespace tractor {

class BulletCollisionEngine : public CollisionEngine {
public:
  virtual std::shared_ptr<CollisionShape>
  create(const Eigen::Affine3d &pose,
         const shapes::Shape *shape) const override;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const override;
};

} // namespace tractor
