// 2022-2024 Philipp Ruppel

#pragma once

#include "shape.h"

#include <tractor/geometry/pose.h>

namespace shapes {
class Shape;
class Mesh;
}  // namespace shapes

namespace tractor {

class ConvexCollisionMesh;

struct CollisionRequest {
  Pose3d pose_a = Pose3d::Identity();
  const CollisionShape *shape_a = nullptr;
  Pose3d pose_b = Pose3d::Identity();
  const CollisionShape *shape_b = nullptr;
  const Vec3d *guess = nullptr;
};

struct CollisionResponse {
  Vec3d point_a = Vec3d::Zero();
  Vec3d point_b = Vec3d::Zero();
  Vec3d normal = Vec3d::Zero();
  double distance = 0.0;
  Vec3d guess = Vec3d::Zero();
};

struct ContinuousCollisionRequest {
  Pose3d pose_a_0 = Pose3d::Identity();
  Pose3d pose_a_1 = Pose3d::Identity();
  const CollisionShape *shape_a = nullptr;
  Pose3d pose_b_0 = Pose3d::Identity();
  Pose3d pose_b_1 = Pose3d::Identity();
  const CollisionShape *shape_b = nullptr;
};

struct ContinuousCollisionResponse {
  Vec3d point_a_0 = Vec3d::Zero();
  Vec3d point_a_1 = Vec3d::Zero();
  Vec3d point_b_0 = Vec3d::Zero();
  Vec3d point_b_1 = Vec3d::Zero();
  Vec3d normal = Vec3d::Zero();
  double distance = 0.0;
};

class CollisionEngine {
 public:
  virtual std::shared_ptr<ConvexCollisionMesh> createConvexMesh(
      const std::string &name, const shapes::Mesh *mesh) const = 0;

  virtual std::shared_ptr<CollisionShape> createSphere(const std::string &name,
                                                       const Pose3d &pose,
                                                       double radius) const = 0;

  virtual std::shared_ptr<CollisionShape> createCylinder(
      const std::string &name, const Pose3d &pose, double length,
      double radius) const = 0;

  virtual std::shared_ptr<CollisionShape> createBox(
      const std::string &name, const Pose3d &pose, const Vec3d &size) const = 0;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const = 0;

  virtual void collide(const ContinuousCollisionRequest &request,
                       ContinuousCollisionResponse &response) const = 0;

  virtual ~CollisionEngine() {}
};

}  // namespace tractor
