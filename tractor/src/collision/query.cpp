// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/query.h>

#include <tractor/core/log.h>

#include <BulletCollision/NarrowPhaseCollision/btComputeGjkEpaPenetration.h>
#include <BulletCollision/NarrowPhaseCollision/btGjkEpa3.h>
#include <BulletCollision/NarrowPhaseCollision/btMprPenetration.h>

namespace tractor {
namespace internal {

class ConvexBulletWrapper {
  const CollisionSupportInterface *_data = nullptr;

public:
  ConvexBulletWrapper(const CollisionSupportInterface *data) : _data(data) {}
  inline const btTransform &getWorldTransform() const {
    return btTransform::getIdentity();
  }
  inline btVector3 getLocalSupportWithMargin(const btVector3 &dir) const {
    double px = 0, py = 0, pz = 0;
    _data->support(dir.x(), dir.y(), dir.z(), px, py, pz);
    return btVector3(px, py, pz);
  }
  inline btVector3 getObjectCenterInWorld() const {
    double px = 0, py = 0, pz = 0;
    _data->center(px, py, pz);
    return btVector3(px, py, pz);
  }
  inline btScalar getMargin() const { return 0; }
  inline btVector3 getLocalSupportWithoutMargin(const btVector3 &dir) const {
    return getLocalSupportWithMargin(dir);
  }
};

struct DistanceInfo {
  btVector3 m_pointOnA = btVector3(0, 0, 0);
  btVector3 m_pointOnB = btVector3(0, 0, 0);
  btVector3 m_normalBtoA = btVector3(0, 0, 0);
  btScalar m_distance = 0;
};

void doCollisionQuery(const CollisionSupportInterface &a,
                      const CollisionSupportInterface &b,
                      CollisionResult &result) {

  TRACTOR_PROFILER("collision query");

  auto wa = ConvexBulletWrapper(&a);
  auto wb = ConvexBulletWrapper(&b);

#if 0
  btGjkCollisionDescription cdesc;
  DistanceInfo info;
  int res = btComputeGjkDistance(wa, wb, cdesc, &info);
  if (res != 0) {
    btMprCollisionDescription mpr_desc;
    res = btComputeMprPenetration(wa, wb, mpr_desc, &info);
    // res = btComputeGjkEpaPenetration(mpr_desc, &info);
  }

  // btVoronoiSimplexSolver simplex_solver;
  // simplex_solver.reset();
  // int ret = btComputeGjkEpaPenetration(wa, wb, cdesc, simplex_solver, &info);

  // TRACTOR_DEBUG("normal ret " << ret);

  if (res == 0) {

    result.ax = info.m_pointOnA.x();
    result.ay = info.m_pointOnA.y();
    result.az = info.m_pointOnA.z();

    result.bx = info.m_pointOnB.x();
    result.by = info.m_pointOnB.y();
    result.bz = info.m_pointOnB.z();

    result.nx = info.m_normalBtoA.x();
    result.ny = info.m_normalBtoA.y();
    result.nz = info.m_normalBtoA.z();

    result.d = info.m_distance;

  } else {

    throw std::runtime_error("collision detection failed");
  }

#endif

#if 1

  // btVector3 guess = btVector3(result.nx, result.ny, result.nz);
  btVector3 guess = btVector3(1, 0, 0);

  btGjkEpaSolver3::sResults results;
  bool ok = btGjkEpaSolver3_Distance(wa, wb, guess, results);
  // TRACTOR_DEBUG("is_separated " << ok);

  // if (!ok || results.distance < 0) {
  if (!ok) {
    ok = btGjkEpaSolver3_Penetration(wa, wb, guess, results);

    if (!ok) {
      TRACTOR_DEBUG("collision detection failed");
    }

    /*result.ax = NAN;
    result.ay = NAN;
    result.az = NAN;
    result.bx = NAN;
    result.by = NAN;
    result.bz = NAN;
    result.nx = NAN;
    result.ny = NAN;
    result.nz = NAN;
    result.d = NAN;*/
  }

  if (ok) {

    result.ax = results.witnesses[0].x();
    result.ay = results.witnesses[0].y();
    result.az = results.witnesses[0].z();

    result.bx = results.witnesses[1].x();
    result.by = results.witnesses[1].y();
    result.bz = results.witnesses[1].z();

    result.nx = results.normal.x();
    result.ny = results.normal.y();
    result.nz = results.normal.z();

    result.d = results.distance;

  } else {

    result.d = 1;

    // throw std::runtime_error("collision detection failed");

    TRACTOR_DEBUG("ERROR COLLISION DETECTION FAILED !!!!");
  }
#endif

  /*
  TRACTOR_DEBUG(result.ax << " " << result.ay << " " << result.az);
  TRACTOR_DEBUG(result.bx << " " << result.by << " " << result.bz);
  TRACTOR_DEBUG(result.nx << " " << result.ny << " " << result.nz);
  TRACTOR_DEBUG(result.d);
  TRACTOR_DEBUG(ret);
  getchar();
  */
}
} // namespace internal
} // namespace tractor
