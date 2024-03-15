// 2022-2024 Philipp Ruppel

#pragma once

#include "shape.h"

#include <tractor/geometry/plane.h>

namespace shapes {
class Shape;
class Mesh;
}  // namespace shapes

namespace tractor {

class SurfaceSampler;

typedef std::vector<Plane3d> PlaneList;
typedef std::vector<Vec3d> VertexList;
typedef std::vector<std::vector<size_t>> FaceList;

class ConvexCollisionMesh : virtual public CollisionShape {
 protected:
  std::string _name;
  VertexList _vertices;
  FaceList _faces;
  PlaneList _planes;

 public:
  virtual void sample(Vec3d &point, Vec3d &normal) const override;
  virtual const std::string &name() const override { return _name; }
  const PlaneList &planes() const { return _planes; }
  const VertexList &vertices() const { return _vertices; }
  const FaceList &faces() const { return _faces; }
};

}  // namespace tractor
