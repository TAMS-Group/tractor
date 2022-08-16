// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/bullet.h>

#include <tractor/collision/base.h>
#include <tractor/core/log.h>
#include <tractor/core/profiler.h>
#include <tractor/geometry/convert.h>
#include <tractor/geometry/plane.h>

#include <BulletCollision/CollisionShapes/btConvexHullShape.h>
#include <BulletCollision/CollisionShapes/btConvexShape.h>
#include <BulletCollision/CollisionShapes/btSphereShape.h>
#include <BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h>
#include <BulletCollision/NarrowPhaseCollision/btGjkEpa3.h>
#include <BulletCollision/NarrowPhaseCollision/btMprPenetration.h>
#include <LinearMath/btConvexHullComputer.h>
#include <LinearMath/btGeometryUtil.h>
#include <geometric_shapes/mesh_operations.h>

#include <mutex>

namespace tractor {

static std::mutex &bulletMutex() {
  static std::mutex m;
  return m;
}

static Vector3<double> toVector3(const btVector3 &v) {
  return Vector3<double>(v.x(), v.y(), v.z());
}

static btVector3 toBulletVector3(const Eigen::Vector3d &v) {
  return btVector3(v.x(), v.y(), v.z());
}

static btMatrix3x3 toBulletMatrix3x3(const Eigen::Matrix3d &m) {
  return btMatrix3x3(m(0, 0), m(0, 1), m(0, 2), m(1, 0), m(1, 1), m(1, 2),
                     m(2, 0), m(2, 1), m(2, 2));
}

static btTransform toBulletTransform(const Eigen::Affine3d &a) {
  btTransform r;
  r.setOrigin(toBulletVector3(a.translation()));
  r.setBasis(toBulletMatrix3x3(a.linear()));
  return r;
}

static btTransform toBulletTransform(const Eigen::Isometry3d &a) {
  btTransform r;
  r.setOrigin(toBulletVector3(a.translation()));
  r.setBasis(toBulletMatrix3x3(a.linear()));
  return r;
}

static Eigen::Vector3d toEigenVector3d(const btVector3 &v) {
  return Eigen::Vector3d(v.x(), v.y(), v.z());
}

struct BulletCollisionWrapper {
  btTransform pose = btTransform::getIdentity();
  const btConvexShape *shape = nullptr;
  BulletCollisionWrapper(const btTransform &pose, const btConvexShape *shape)
      : pose(pose), shape(shape) {}
  inline btScalar getMargin() const { return shape->getMargin(); }
  inline btVector3 getObjectCenterInWorld() const { return pose.getOrigin(); }
  inline const btTransform &getWorldTransform() const { return pose; }
  inline btVector3 getLocalSupportWithMargin(const btVector3 &dir) const {
    return shape->localGetSupportingVertex(dir);
  }
  inline btVector3 getLocalSupportWithoutMargin(const btVector3 &dir) const {
    return shape->localGetSupportingVertexWithoutMargin(dir);
  }
};

static void bulletCollide(const char *name_a, const btTransform &pose_a,
                          const btConvexShape *shape_a, const char *name_b,
                          const btTransform &pose_b,
                          const btConvexShape *shape_b,
                          CollisionResponse &response) {
  TRACTOR_PROFILER("bullet gjk");
  BulletCollisionWrapper wa = BulletCollisionWrapper(pose_a, shape_a);
  BulletCollisionWrapper wb = BulletCollisionWrapper(pose_b, shape_b);
  btVector3 guess = btVector3(1, 2, 3).normalized();
  btGjkEpaSolver3::sResults results;
  bool ok = btGjkEpaSolver3_Distance(wa, wb, guess, results);
  if (!ok) {
    ok = btGjkEpaSolver3_Penetration(wa, wb, guess, results);
  }
  if (ok) {
    response.point_a = toEigenVector3d(results.witnesses[0]);
    response.point_b = toEigenVector3d(results.witnesses[1]);
    response.normal = toEigenVector3d(pose_a.getBasis() * results.normal);
    response.distance = results.distance;
  } else {
    TRACTOR_WARN("collision detection failed " << name_a << " " << name_b);
    response = CollisionResponse();
  }
}

struct BulletCollisionShape : public ConvexPolyhedralCollisionShape {

  const btScalar margin = 0.005;
  std::vector<Plane<double>> bounding_planes;
  const CollisionEngine *collision_engine = nullptr;
  std::shared_ptr<btConvexHullShape> bullet_shape = nullptr;

  virtual const CollisionEngine *engine() const override {
    return collision_engine;
  }

  virtual const std::vector<Plane<double>> &planes() const override {
    return bounding_planes;
  }

  BulletCollisionShape(const CollisionEngine *engine, const std::string &name,
                       const Eigen::Affine3d &pose, const shapes::Shape *shape)
      : collision_engine(engine) {
    {
      TRACTOR_DEBUG("bullet collision shape " << typeid(*shape).name());

      const shapes::Mesh *mesh = dynamic_cast<const shapes::Mesh *>(shape);
      const shapes::Mesh *mesh_cleanup = nullptr;
      if (!mesh) {
        mesh_cleanup = mesh = shapes::createMeshFromShape(shape);
      }

      initMeshBase(name, pose, mesh);

      auto sh = std::make_shared<btConvexHullShape>();

      btConvexHullComputer hull_computer;
      hull_computer.compute(mesh->vertices, sizeof(double) * 3,
                            mesh->vertex_count, btScalar(margin), btScalar(0));
      for (size_t i = 0; i < hull_computer.vertices.size(); i++) {
        auto &v = hull_computer.vertices[i];
        Eigen::Vector3d vertex(v.x(), v.y(), v.z());
        vertex = pose * vertex;
        sh->addPoint(btVector3(vertex.x(), vertex.y(), vertex.z()));
      }

      sh->setMargin(margin);

      bounding_planes.clear();
      for (size_t face_index = 0; face_index < hull_computer.faces.size();
           face_index++) {
        auto *edge1 = &hull_computer.edges[hull_computer.faces[face_index]];
        auto *edge2 = edge1->getNextEdgeOfFace();
        auto *edge3 = edge2->getNextEdgeOfFace();
        auto v0 = toVector3<double>(
            pose *
            toEigenVector3d(hull_computer.vertices[edge1->getSourceVertex()]));
        auto v1 = toVector3<double>(
            pose *
            toEigenVector3d(hull_computer.vertices[edge2->getSourceVertex()]));
        auto v2 = toVector3<double>(
            pose *
            toEigenVector3d(hull_computer.vertices[edge3->getSourceVertex()]));
        bounding_planes.emplace_back(normalized(cross(v1 - v0, v2 - v0)),
                                     (v0 + v1 + v2) * (1.0 / 3.0));
      }

      bullet_shape = sh;

      delete mesh_cleanup;
      return;
    }
  }

  virtual void project(const Eigen::Vector3d &in_point,
                       Eigen::Vector3d &closest_point,
                       Eigen::Vector3d &surface_normal) const override {

    TRACTOR_PROFILER("bullet project");

    std::lock_guard<std::mutex>(bulletMutex());

    btSphereShape point_shape(margin);
    // btSphereShape point_shape(0.01);
    CollisionResponse response;
    bulletCollide(
        name().c_str(), btTransform::getIdentity(), bullet_shape.get(), "point",
        btTransform(btMatrix3x3::getIdentity(), toBulletVector3(in_point)),
        &point_shape, response);
    closest_point = response.point_a;
    surface_normal = -response.normal;

    // bulletCollide(
    //     btTransform::getIdentity(), bullet_shape.get(),
    //     btTransform(btMatrix3x3::getIdentity(),
    //     toBulletVector3(closest_point)), &point_shape, response);

    // double min_dist = -1;
    // TRACTOR_DEBUG("plane count " << bullet_shape->getNumPlanes());
    // for (size_t i = 0; i < bullet_shape->getNumPlanes(); i++) {
    //   TRACTOR_DEBUG("plane " << i);
    //   btVector3 normal;
    //   btVector3 support;
    //   bullet_shape->getPlane(normal, support, i);
    //   double dist =
    //       std::abs((toBulletVector3(closest_point) - support).dot(normal));
    //   if (min_dist < 0 || dist < min_dist) {
    //     min_dist = dist;
    //     surface_normal = toEigenVector3d(normal);
    //   }
    // }

    // auto closest_point_t = toVector3<double>(closest_point);
    // double min_dist = -1;
    // // TRACTOR_DEBUG("closest point " << closest_point_t);
    // for (size_t i = 0; i < bounding_planes.size(); i++) {
    //   auto &plane = bounding_planes[i];
    //   double dist = std::abs(plane.signedDistance(closest_point_t) - margin);
    //   // double dist = std::abs((closest_point -
    //   toEigenVector3d(plane.point()))
    //   //                            .dot(toEigenVector3d(plane.normal())));
    //   // TRACTOR_DEBUG("plane " << i << " " << plane << " dist " << dist);
    //   if (min_dist < -0.5 || dist < min_dist) {
    //     min_dist = dist;
    //     surface_normal = toEigenVector3d(plane.normal());
    //   }
    // }
  }
};

std::shared_ptr<CollisionShape>
BulletCollisionEngine::create(const std::string &name,
                              const Eigen::Affine3d &pose,
                              const shapes::Shape *shape) const {

  std::lock_guard<std::mutex>(bulletMutex());

  return std::make_shared<BulletCollisionShape>(this, name, pose, shape);
}

void BulletCollisionEngine::collide(const CollisionRequest &request,
                                    CollisionResponse &response) const {

  TRACTOR_PROFILER("bullet collide");

  std::lock_guard<std::mutex>(bulletMutex());

  bulletCollide(
      request.shape_a->name().c_str(), toBulletTransform(request.pose_a),
      ((BulletCollisionShape *)request.shape_a)->bullet_shape.get(),
      request.shape_b->name().c_str(), toBulletTransform(request.pose_b),
      ((BulletCollisionShape *)request.shape_b)->bullet_shape.get(), response);
}

} // namespace tractor
