// (c) 2022 Philipp Ruppel

#pragma once

#include <memory>
#include <tractor/geometry/vector3.h>

namespace tractor {

class CollisionEngine;

struct CollisionResponse {
  Vec3d point_a = Vec3d::Zero();
  Vec3d point_b = Vec3d::Zero();
  Vec3d normal = Vec3d::Zero();
  double distance = 0.0;
};

struct CollisionShape {
  CollisionShape(const CollisionShape &) = delete;
  CollisionShape &operator=(const CollisionShape &) = delete;
  CollisionShape() {}
  virtual ~CollisionShape() {}
  virtual const CollisionEngine *engine() const = 0;
  virtual void project(const Vec3d &point, Vec3d &closest_point,
                       Vec3d &surface_normal) const = 0;
  virtual void sample(Vec3d &point, Vec3d &normal) const = 0;
  virtual const std::string &name() const = 0;
};

} // namespace tractor
