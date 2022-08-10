// (c) 2022 Philipp Ruppel

#pragma once

#include "engine.h"

namespace tractor {

class SurfaceSampler;

struct MeshCollisionShapeBase : public CollisionShape {

public:
  virtual void sample(Eigen::Vector3d &point,
                      Eigen::Vector3d &normal) const override;

protected:
  std::shared_ptr<const SurfaceSampler> surface_sampler;
  void initMeshBase(const Eigen::Affine3d &pose, const shapes::Mesh *mesh);
};

} // namespace tractor
