// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/bullet.h>

#include <tractor/core/log.h>

#include <BulletCollision/CollisionShapes/btConvexHullShape.h>
#include <BulletCollision/CollisionShapes/btConvexShape.h>
#include <BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h>
#include <BulletCollision/NarrowPhaseCollision/btGjkEpa3.h>
#include <BulletCollision/NarrowPhaseCollision/btMprPenetration.h>
#include <LinearMath/btConvexHullComputer.h>
#include <LinearMath/btGeometryUtil.h>
#include <geometric_shapes/mesh_operations.h>

namespace tractor {

btVector3 toBulletVector3(const Eigen::Vector3d &v) {
  return btVector3(v.x(), v.y(), v.z());
}

btMatrix3x3 toBulletMatrix3x3(const Eigen::Matrix3d &m) {
  return btMatrix3x3(m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2),
                     m(2, 0), m(2, 1), m(2, 2));
}

btTransform toBulletTransform(const Eigen::Affine3d &a) {
  btTransform r;
  r.setOrigin(toBulletVector3(a.translation()));
  r.setBasis(toBulletMatrix3x3(a.linear()));
  return r;
}

btTransform toBulletTransform(const Eigen::Isometry3d &a) {
  btTransform r;
  r.setOrigin(toBulletVector3(a.translation()));
  r.setBasis(toBulletMatrix3x3(a.linear()));
  return r;
}

Eigen::Vector3d toEigenVector3d(const btVector3 &v) {
  return Eigen::Vector3d(v.x(), v.y(), v.z());
}

struct BulletCollisionShape : public CollisionShape {
  const CollisionEngine *collision_engine = nullptr;
  // btTransform bullet_pose = toBulletTransform(Eigen::Isometry3d::Identity());
  std::shared_ptr<btConvexShape> bullet_shape = nullptr;
  virtual const CollisionEngine *engine() const override {
    return collision_engine;
  }
  BulletCollisionShape(const CollisionEngine *engine,
                       const Eigen::Affine3d &pose, const shapes::Shape *shape)
      : collision_engine(engine) {
    {
      const shapes::Mesh *mesh = dynamic_cast<const shapes::Mesh *>(shape);
      const shapes::Mesh *mesh_cleanup = nullptr;
      if (!mesh) {
        mesh_cleanup = mesh = shapes::createMeshFromShape(shape);
      }

      auto sh = std::make_shared<btConvexHullShape>();

      // for (size_t vertex_index = 0; vertex_index < mesh->vertex_count;
      //      vertex_index++) {
      //   Eigen::Vector3d vertex(mesh->vertices[vertex_index * 3 + 0],
      //                          mesh->vertices[vertex_index * 3 + 1],
      //                          mesh->vertices[vertex_index * 3 + 2]);
      //   vertex = pose * vertex;
      //   sh->addPoint(btVector3(vertex.x(), vertex.y(), vertex.z()));
      // }

      btConvexHullComputer hull_computer;
      hull_computer.compute(mesh->vertices, sizeof(double) * 3,
                            mesh->vertex_count, btScalar(0), btScalar(0));
      for (size_t i = 0; i < hull_computer.vertices.size(); i++) {
        auto &v = hull_computer.vertices[i];
        Eigen::Vector3d vertex(v.x(), v.y(), v.z());
        vertex = pose * vertex;
        sh->addPoint(btVector3(vertex.x(), vertex.y(), vertex.z()));
      }

      bullet_shape = sh;

      delete mesh_cleanup;
      return;
    }
  }
};

struct BulletCollisionWrapper {
  btConvexShape *shape = nullptr;
  btTransform pose;
  BulletCollisionWrapper(const Eigen::Isometry3d &pose,
                         const BulletCollisionShape *shape)
      : pose(toBulletTransform(pose)), shape(shape->bullet_shape.get()) {}
  inline btScalar getMargin() const {
    return 0.0;
    // return shape->getMargin();
  }
  inline btVector3 getObjectCenterInWorld() const { return pose.getOrigin(); }
  inline const btTransform &getWorldTransform() const { return pose; }
  inline btVector3 getLocalSupportWithMargin(const btVector3 &dir) const {
    return shape->localGetSupportingVertexWithoutMargin(dir);
  }
  inline btVector3 getLocalSupportWithoutMargin(const btVector3 &dir) const {
    return shape->localGetSupportingVertexWithoutMargin(dir);
  }
};

std::shared_ptr<CollisionShape>
BulletCollisionEngine::create(const Eigen::Affine3d &pose,
                              const shapes::Shape *shape) const {
  return std::make_shared<BulletCollisionShape>(this, pose, shape);
}

void BulletCollisionEngine::collide(const CollisionRequest &request,
                                    CollisionResponse &response) const {

  auto wa = BulletCollisionWrapper(
      request.pose_a, (const BulletCollisionShape *)request.shape_a);
  auto wb = BulletCollisionWrapper(
      request.pose_b, (const BulletCollisionShape *)request.shape_b);

  response = CollisionResponse();

  btVector3 guess = btVector3(1, 0, 0);

  btGjkEpaSolver3::sResults results;
  bool ok = btGjkEpaSolver3_Distance(wa, wb, guess, results);
  if (!ok) {
    ok = btGjkEpaSolver3_Penetration(wa, wb, guess, results);
    if (!ok) {
      TRACTOR_DEBUG("collision detection failed");
    }
  }

  if (ok) {
    response.point_a = toEigenVector3d(results.witnesses[0]);
    response.point_b = toEigenVector3d(results.witnesses[1]);
    response.normal =
        request.pose_a.rotation() * toEigenVector3d(results.normal);
    response.distance = results.distance;
  } else {
    TRACTOR_ERROR("collision detection failed");
  }
}

} // namespace tractor
