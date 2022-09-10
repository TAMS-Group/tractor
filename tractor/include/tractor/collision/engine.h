// (c) 2022 Philipp Ruppel

#pragma once

#include "shape.h"

#include <tractor/geometry/pose.h>

namespace shapes {
class Shape;
class Mesh;
} // namespace shapes

namespace tractor {

class ConvexCollisionMesh;

struct CollisionRequest {
  Pose3d pose_a = Pose3d::Identity();
  const CollisionShape *shape_a = nullptr;
  Pose3d pose_b = Pose3d::Identity();
  const CollisionShape *shape_b = nullptr;
};

struct CollisionResponse {
  Vec3d point_a = Vec3d::Zero();
  Vec3d point_b = Vec3d::Zero();
  Vec3d normal = Vec3d::Zero();
  double distance = 0.0;
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

// struct ContinuousCollisionManifoldRequest {
//   Pose3d pose_a_0 = Pose3d::Identity();
//   Pose3d pose_a_1 = Pose3d::Identity();
//   const CollisionShape *shape_a = nullptr;
//   Pose3d pose_b_0 = Pose3d::Identity();
//   Pose3d pose_b_1 = Pose3d::Identity();
//   const CollisionShape *shape_b = nullptr;
//   size_t point_count = 0;
// };

// struct ContinuousCollisionRequest {
//   Pose3d pose_a_0 = Pose3d::Identity();
//   Pose3d pose_a_1 = Pose3d::Identity();
//   const CollisionShape *shape_a = nullptr;
//   Pose3d pose_b_0 = Pose3d::Identity();
//   Pose3d pose_b_1 = Pose3d::Identity();
//   const CollisionShape *shape_b = nullptr;
// };

// struct ContinuousCollisionResponse {
//   Vec3d normal = Vec3d::Zero();
// };

class CollisionEngine {
public:
  virtual std::shared_ptr<ConvexCollisionMesh>
  createConvexMesh(const std::string &name, const shapes::Mesh *mesh) const = 0;

  virtual void collide(const CollisionRequest &request,
                       CollisionResponse &response) const = 0;

  virtual void collide(const ContinuousCollisionRequest &request,
                       ContinuousCollisionResponse &response) const = 0;

  // virtual void collide(const ContinuousCollisionManifoldRequest &request,
  //                      Vec3d &global_normal, Vec3d *local_points_a,
  //                      Vec3d *local_points_b) const = 0;

  virtual ~CollisionEngine() {}
};

} // namespace tractor
