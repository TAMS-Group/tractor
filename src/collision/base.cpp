// 2020-2024 Philipp Ruppel

#include <tractor/collision/base.h>

#include <tractor/core/log.h>
#include <tractor/geometry/eigen.h>
#include <tractor/geometry/plane.h>

#include <geometric_shapes/mesh_operations.h>

#include <random>

namespace tractor {

void ConvexCollisionMesh::sample(Vec3d &point, Vec3d &normal) const {
  throw std::runtime_error("not implemented");
}

}  // namespace tractor
