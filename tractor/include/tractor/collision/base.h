// (c) 2022 Philipp Ruppel

#pragma once

#include "engine.h"

#include <tractor/geometry/plane.h>

namespace tractor {

class SurfaceSampler;

struct ConvexPolyhedralCollisionShape : public CollisionShape {
  std::string _name;

public:
  virtual void sample(Eigen::Vector3d &point,
                      Eigen::Vector3d &normal) const override;
  virtual const std::string &name() const override { return _name; }
  virtual const std::vector<Plane<double>> &planes() const = 0;

protected:
  std::shared_ptr<const SurfaceSampler> surface_sampler;
  void initMeshBase(const std::string &name, const Eigen::Affine3d &pose,
                    const shapes::Mesh *mesh);
};

} // namespace tractor
