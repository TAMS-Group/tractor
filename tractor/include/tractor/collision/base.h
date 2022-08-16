// (c) 2022 Philipp Ruppel

#pragma once

#include "shape.h"

#include <tractor/geometry/plane.h>

namespace shapes {
class Shape;
class Mesh;
} // namespace shapes

namespace tractor {

class SurfaceSampler;

class ConvexCollisionMesh : public CollisionShape {
  std::string _name;
  std::vector<Vec3d> _vertices;

public:
  virtual void sample(Vec3d &point, Vec3d &normal) const override;
  virtual const std::string &name() const override { return _name; }
  virtual const std::vector<Plane3d> &planes() const = 0;
  const std::vector<Vec3d> &vertices() const { return _vertices; };

protected:
  std::shared_ptr<const SurfaceSampler> surface_sampler;
  void initConvexMesh(const std::string &name, const shapes::Mesh *mesh);
};

} // namespace tractor
