// 2020-2024 Philipp Ruppel

#include <tractor/collision/bullet.h>

#include <tractor/collision/base.h>
#include <tractor/core/log.h>
#include <tractor/core/profiler.h>
#include <tractor/geometry/eigen.h>
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
#include <BulletCollision/CollisionShapes/btCylinderShape.h>
#include <BulletCollision/CollisionShapes/btBoxShape.h>

#include <mutex>

namespace tractor {

static std::mutex &bulletMutex() {
  static std::mutex m;
  return m;
}

static Vector3<double> toVec3d(const btVector3 &v) {
  return Vector3<double>(v.x(), v.y(), v.z());
}

static btVector3 toBulletVector3(const Eigen::Vector3d &v) {
  return btVector3(v.x(), v.y(), v.z());
}

static btVector3 toBulletVector3(const Vec3d &v) {
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

static btQuaternion toBulletQuaternion(const Quat3d &a) {
  return btQuaternion(a.x(), a.y(), a.z(), a.w());
}

static btTransform toBulletTransform(const Pose3d &a) {
  return btTransform(toBulletQuaternion(a.orientation()),
                     toBulletVector3(a.translation()));
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
  btVector3 center = btVector3(0, 0, 0);
  BulletCollisionWrapper(const btTransform &pose, const btConvexShape *shape,
                         const btVector3 &center)
      : pose(pose), shape(shape), center(center) {}
  inline btScalar getMargin() const { return shape->getMargin(); }
  inline btVector3 getObjectCenterInWorld() const { return pose * center; }
  inline const btTransform &getWorldTransform() const { return pose; }
  inline btVector3 getLocalSupportWithMargin(const btVector3 &dir) const {
    return shape->localGetSupportingVertex(dir);
  }
  inline btVector3 getLocalSupportWithoutMargin(const btVector3 &dir) const {
    return shape->localGetSupportingVertexWithoutMargin(dir);
  }
};

static void bulletCollide(
    const char *name_a, const btTransform &pose_a, const btConvexShape *shape_a,
    const btVector3 &center_a, const char *name_b, const btTransform &pose_b,
    const btConvexShape *shape_b, const btVector3 &center_b,
    CollisionResponse &response, const Vec3d *guess_opt = nullptr) {
  TRACTOR_PROFILER("bullet gjk");
  BulletCollisionWrapper wa = BulletCollisionWrapper(pose_a, shape_a, center_a);
  BulletCollisionWrapper wb = BulletCollisionWrapper(pose_b, shape_b, center_b);
  btVector3 guess = btVector3(0, 0, 0);
  if (guess_opt) {
    guess = toBulletVector3(*guess_opt);
  }
  if (guess.isZero()) {
    guess = btVector3(1, 2, 3).normalized();
  }
  btGjkEpaSolver3::sResults results;
  bool ok = btGjkEpaSolver3_Distance(wa, wb, guess, results);
  if (!ok) {
    ok = btGjkEpaSolver3_Penetration(wa, wb, guess, results);
  }
  if (ok) {
    response.point_a = toVec3d(results.witnesses[0]);
    response.point_b = toVec3d(results.witnesses[1]);
    response.normal = toVec3d(pose_a.getBasis() * results.normal);
    response.distance = results.distance;
    response.guess = toVec3d(guess);
  } else {
    TRACTOR_WARN("collision detection failed " << name_a << " " << name_b);
    response = CollisionResponse();
  }
}

struct ContinuousBulletCollisionWrapper {
  const char *name = nullptr;
  btTransform pose_0 = btTransform::getIdentity();
  btTransform pose_1 = btTransform::getIdentity();
  btMatrix3x3 rot_inv_0 = btMatrix3x3::getIdentity();
  btMatrix3x3 rot_inv_1 = btMatrix3x3::getIdentity();
  const btConvexShape *shape = nullptr;
  btVector3 center = btVector3(0, 0, 0);
  ContinuousBulletCollisionWrapper(const char *name, const btTransform &pose_0,
                                   const btTransform &pose_1,
                                   const btConvexShape *shape,
                                   const btVector3 &center)
      : name(name),
        pose_0(pose_0),
        pose_1(pose_1),
        shape(shape),
        center(center) {
    rot_inv_0 = pose_0.getBasis().inverse();
    rot_inv_1 = pose_1.getBasis().inverse();
  }
  inline btScalar getMargin() const { return shape->getMargin(); }
  inline btVector3 getObjectCenterInWorld() const {
    return (pose_0 * center + pose_1 * center) * btScalar(0.5);
  }
  inline const btTransform &getWorldTransform() const {
    return btTransform::getIdentity();
  }
  inline btVector3 getLocalSupportWithMargin(const btVector3 &dir) const {
    btVector3 sup_0 = pose_0 * shape->localGetSupportingVertex(rot_inv_0 * dir);
    btVector3 sup_1 = pose_1 * shape->localGetSupportingVertex(rot_inv_1 * dir);
    btScalar dot_0 = dir.dot(sup_0);
    btScalar dot_1 = dir.dot(sup_1);
    return (dot_0 > dot_1) ? sup_0 : sup_1;
  }
  inline btVector3 getLocalSupportWithoutMargin(const btVector3 &dir) const {
    btVector3 sup_0 =
        pose_0 * shape->localGetSupportingVertexWithoutMargin(rot_inv_0 * dir);
    btVector3 sup_1 =
        pose_1 * shape->localGetSupportingVertexWithoutMargin(rot_inv_1 * dir);
    btScalar dot_0 = dir.dot(sup_0);
    btScalar dot_1 = dir.dot(sup_1);
    return (dot_0 > dot_1) ? sup_0 : sup_1;
  }
};

static void bulletCollideContinuous(const ContinuousBulletCollisionWrapper &wa,
                                    const ContinuousBulletCollisionWrapper &wb,
                                    btVector3 &normal  //
) {
  TRACTOR_PROFILER("bullet gjk continuous");
  btVector3 guess = btVector3(1, 2, 3).normalized();
  btGjkEpaSolver3::sResults results;
  bool ok = btGjkEpaSolver3_Distance(wa, wb, guess, results);
  if (!ok) {
    ok = btGjkEpaSolver3_Penetration(wa, wb, guess, results);
  }
  if (ok) {
    normal = results.normal;
  } else {
    TRACTOR_WARN("collision detection failed " << wa.name << " " << wb.name);
    normal = btVector3(0, 0, 0);
  }
}

struct BulletCollisionShape : public virtual CollisionShape {
  const btScalar margin = 0.005;
  const CollisionEngine *collision_engine = nullptr;
  btVector3 center = btVector3(0, 0, 0);
  std::shared_ptr<btConvexShape> bullet_shape = nullptr;
  btTransform transform = btTransform::getIdentity();
  virtual const CollisionEngine *engine() const override {
    return collision_engine;
  }
  BulletCollisionShape(const CollisionEngine *engine)
      : collision_engine(engine) {}
  virtual void project(const Vec3d &in_point, Vec3d &closest_point,
                       Vec3d &surface_normal) const override {
    TRACTOR_PROFILER("bullet project");

    std::lock_guard<std::mutex>(tractor::bulletMutex());

    btSphereShape point_shape(margin);
    CollisionResponse response;
    bulletCollide(
        name().c_str(), transform, bullet_shape.get(), center, "point",
        btTransform(btMatrix3x3::getIdentity(), toBulletVector3(in_point)),
        &point_shape, btVector3(0, 0, 0), response);
    closest_point = response.point_a;
    surface_normal = -response.normal;
  }
};

struct BulletConvexMesh : public ConvexCollisionMesh,
                          public BulletCollisionShape {
  BulletConvexMesh(const CollisionEngine *engine, const std::string &name,
                   const shapes::Mesh *mesh)
      : BulletCollisionShape(engine) {
    {
      TRACTOR_DEBUG("bullet collision shape " << typeid(*mesh).name());

      btConvexHullComputer hull_computer;
      hull_computer.compute(mesh->vertices, sizeof(double) * 3,
                            mesh->vertex_count, btScalar(margin), btScalar(0));

      auto sh = std::make_shared<btConvexHullShape>();
      for (size_t i = 0; i < hull_computer.vertices.size(); i++) {
        auto &v = hull_computer.vertices[i];
        sh->addPoint(btVector3(v.x(), v.y(), v.z()));
      }
      sh->setMargin(margin);

      center = btVector3(0, 0, 0);
      for (size_t i = 0; i < hull_computer.vertices.size(); i++) {
        auto &v = hull_computer.vertices[i];
        center += v;
      }
      center /= hull_computer.vertices.size();

      _planes.clear();
      for (size_t face_index = 0; face_index < hull_computer.faces.size();
           face_index++) {
        auto *edge1 = &hull_computer.edges[hull_computer.faces[face_index]];
        auto *edge2 = edge1->getNextEdgeOfFace();
        auto *edge3 = edge2->getNextEdgeOfFace();
        auto v0 = toVec3d(hull_computer.vertices[edge1->getSourceVertex()]);
        auto v1 = toVec3d(hull_computer.vertices[edge2->getSourceVertex()]);
        auto v2 = toVec3d(hull_computer.vertices[edge3->getSourceVertex()]);
        _planes.emplace_back(normalized(cross(v1 - v0, v2 - v0)),
                             (v0 + v1 + v2) * (1.0 / 3.0));
      }

      _name = name;

      _vertices.clear();
      for (size_t i = 0; i < hull_computer.vertices.size(); i++) {
        _vertices.push_back(toVec3d(hull_computer.vertices[i]));
      }

      _faces.clear();
      for (size_t face_index = 0; face_index < hull_computer.faces.size();
           face_index++) {
        _faces.emplace_back();
        const auto *edge =
            &hull_computer.edges[hull_computer.faces[face_index]];
        int v0 = edge->getSourceVertex();
        _faces.back().emplace_back(v0);
        int v2 = edge->getTargetVertex();
        while (v2 != v0) {
          _faces.back().emplace_back(v2);
          edge = edge->getNextEdgeOfFace();
          v2 = edge->getTargetVertex();
        }
      }
      bullet_shape = sh;
      return;
    }
  }
};
std::shared_ptr<ConvexCollisionMesh> BulletCollisionEngine::createConvexMesh(
    const std::string &name, const shapes::Mesh *mesh) const {
  std::lock_guard<std::mutex>(tractor::bulletMutex());
  return std::make_shared<BulletConvexMesh>(this, name, mesh);
}

struct BulletPrimitive : public BulletCollisionShape {
  std::string _name;
  BulletPrimitive(const CollisionEngine *engine, const std::string &name,
                  const Pose3d &pose,
                  const std::shared_ptr<btConvexShape> &shape)
      : BulletCollisionShape(engine), _name(name) {
    // this->
    this->transform = toBulletTransform(pose);
    this->bullet_shape = shape;
  }
  virtual void sample(Vec3d &point, Vec3d &normal) const override {
    throw std::logic_error("not implemented");
  }
  virtual const std::string &name() const override { return _name; }
};

std::shared_ptr<CollisionShape> BulletCollisionEngine::createSphere(
    const std::string &name, const Pose3d &pose, double radius) const {
  std::lock_guard<std::mutex>(tractor::bulletMutex());
  return std::make_shared<BulletPrimitive>(
      this, name, pose, std::make_shared<btSphereShape>(radius));
}

std::shared_ptr<CollisionShape> BulletCollisionEngine::createCylinder(
    const std::string &name, const Pose3d &pose, double length,
    double radius) const {
  return std::make_shared<BulletPrimitive>(
      this, name, pose,
      std::make_shared<btCylinderShapeZ>(
          btVector3(radius, radius, length * 0.5)));
}

std::shared_ptr<CollisionShape> BulletCollisionEngine::createBox(
    const std::string &name, const Pose3d &pose, const Vec3d &size) const {
  return std::make_shared<BulletPrimitive>(
      this, name, pose,
      std::make_shared<btBoxShape>(toBulletVector3(size * 0.5)));
}

void BulletCollisionEngine::collide(const CollisionRequest &request,
                                    CollisionResponse &response) const {
  TRACTOR_PROFILER("bullet collide");
  std::lock_guard<std::mutex>(tractor::bulletMutex());
  auto *shape_a = &dynamic_cast<const BulletCollisionShape &>(*request.shape_a);
  auto *shape_b = &dynamic_cast<const BulletCollisionShape &>(*request.shape_b);
  bulletCollide(
      shape_a->name().c_str(),
      toBulletTransform(request.pose_a) * shape_a->transform,
      shape_a->bullet_shape.get(), shape_a->center, shape_b->name().c_str(),
      toBulletTransform(request.pose_b) * shape_b->transform,
      shape_b->bullet_shape.get(), shape_b->center, response, request.guess);
}

void BulletCollisionEngine::collide(const ContinuousCollisionRequest &request,
                                    ContinuousCollisionResponse &res) const {
  TRACTOR_PROFILER("bullet collide continuous");
  std::lock_guard<std::mutex>(tractor::bulletMutex());

  auto *shape_a = &dynamic_cast<const BulletCollisionShape &>(*request.shape_a);
  auto *shape_b = &dynamic_cast<const BulletCollisionShape &>(*request.shape_b);

  ContinuousBulletCollisionWrapper wa(
      shape_a->name().c_str(),
      toBulletTransform(request.pose_a_0) * shape_a->transform,
      toBulletTransform(request.pose_a_1) * shape_a->transform,
      shape_a->bullet_shape.get(), shape_a->center);

  ContinuousBulletCollisionWrapper wb(
      shape_b->name().c_str(),
      toBulletTransform(request.pose_b_0) * shape_b->transform,
      toBulletTransform(request.pose_b_1) * shape_b->transform,
      shape_b->bullet_shape.get(), shape_b->center);

  btVector3 dir = btVector3(0, 0, 0);
  bulletCollideContinuous(wa, wb, dir);
  res.normal = toVec3d(dir);

  static auto pt = [](const btTransform &pose, const btMatrix3x3 &rot_inv,
                      const btConvexShape *shape, const btVector3 &dir) {
    return toVec3d(pose *
                   shape->localGetSupportingVertexWithoutMargin(rot_inv * dir));
  };

  res.point_a_0 = pt(wa.pose_0, wa.rot_inv_0, wa.shape, -dir);
  res.point_a_1 = pt(wa.pose_1, wa.rot_inv_1, wa.shape, -dir);

  res.point_b_0 = pt(wb.pose_0, wb.rot_inv_0, wb.shape, dir);
  res.point_b_1 = pt(wb.pose_1, wb.rot_inv_1, wb.shape, dir);
}

}  // namespace tractor
