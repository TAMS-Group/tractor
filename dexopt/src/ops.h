// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/collision/query.h>
#include <tractor/core/eigen.h>
#include <tractor/core/operator.h>
#include <tractor/core/recorder.h>
#include <tractor/geometry/vector3.h>

#include <random>

namespace tractor {

template <class T> struct BarrierTypes {
  typedef Eigen::Matrix<T, 6, 1> GradientType;
  typedef Eigen::Matrix<T, 6, 6> HessianType;
  typedef Eigen::Matrix<T, 3, 1> Vector3;
  typedef Eigen::Matrix<T, 3, 3> Matrix3;
};

// ------------------------------------------

TRACTOR_OP(acos, (const T &a),
           { return std::max(T(-1), std::min(T(1), T(std::acos(a)))); })
TRACTOR_D(prepare, acos, (const T &a, const T &x, T &p), {
  p = T(-1) / std::max(T(1e-9), T(std::sqrt(std::max(T(0), T(1) - a * a))));
})
TRACTOR_D(forward, acos, (const T &p, const T &da, T &dx), { dx = da * p; })
TRACTOR_D(reverse, acos, (const T &p, T &da, const T &dx), { da = dx * p; })

// ------------------------------------------

template <class T>
static void collision_project_2(const Vector3<T> &point,
                                const uint64_t &shape_id, Vector3<T> &normal,
                                T &distance) {
  auto &shape = *(CollisionShape<T> *)shape_id;
  shape.project(point, normal, distance);
}

template <class T, size_t S>
static void
collision_project_2(const Vector3<Batch<T, S>> &point, const uint64_t &shape_id,
                    Vector3<Batch<T, S>> &normal, Batch<T, S> &distance) {
  auto &shape = *(CollisionShape<T> *)shape_id;
  for (size_t i = 0; i < S; i++) {
    Vector3<T> n;
    shape.project(Vector3<T>(point.x()[i], point.y()[i], point.z()[i]), n,
                  distance[i]);
    normal.x()[i] = n.x();
    normal.y()[i] = n.y();
    normal.z()[i] = n.z();
  }
}

TRACTOR_OP(collision_project_2,
           (const Vector3<T> &point, const uint64_t &shape_id,
            Vector3<T> &normal, T &distance),
           { collision_project_2(point, shape_id, normal, distance); })
TRACTOR_D(prepare, collision_project_2,
          (const Vector3<T> &point, const uint64_t &shape_id,
           const Vector3<T> &normal, const T &distance, Vector3<T> &n),
          { n = normal; })
TRACTOR_D(forward, collision_project_2,
          (const Vector3<T> &n, const Vector3<T> &point,
           const uint64_t &shape_id, Vector3<T> &normal, T &distance),
          {
            normal.setZero();
            distance = -dot(n, point);
          })
TRACTOR_D(reverse, collision_project_2,
          (const Vector3<T> &n, Vector3<T> &point, uint64_t &shape_id,
           const Vector3<T> &normal, const T &distance),
          {
            point = n * -distance;
            shape_id = 0;
          })

// ------------------------------------------

template <class T>
static void collision_project(const Vector3<T> &point, const uint64_t &shape_id,
                              Vector3<T> &normal, T &distance) {
  // auto &shape = *(CollisionShape<T> *)shape_id;
  // shape.project(point, normal, distance);
  throw std::runtime_error("NYI collision_project");
}
TRACTOR_OP(collision_project,
           (const Vector3<T> &point, const uint64_t &shape_id,
            Vector3<T> &normal, T &distance),
           { collision_project(point, shape_id, normal, distance); })
TRACTOR_D(prepare, collision_project,
          (const Vector3<T> &point, const uint64_t &shape_id,
           const Vector3<T> &normal, const T &distance),
          {})
TRACTOR_D(forward, collision_project,
          (const Vector3<T> &point, const uint64_t &shape_id,
           Vector3<T> &normal, T &distance),
          {
            normal.setZero();
            distance = T(0);
          })
TRACTOR_D(reverse, collision_project,
          (Vector3<T> & point, uint64_t &shape_id, const Vector3<T> &normal,
           const T &distance),
          {
            point.setZero();
            shape_id = 0;
          })

// ------------------------------------------

// TRACTOR_OP(capture_constraint, (const Vector3<T> &a, const uint64_t
// &shape_id),
//            { return Vector3<T>::Zero(); })
// TRACTOR_D(prepare, capture_constraint,
//           (const Vector3<T> &a, const uint64_t &shape_id, const Vector3<T>
//           &x),
//           {})
// TRACTOR_D(forward, capture_constraint,
//           (const Vector3<T> &a, const uint64_t &shape_id, Vector3<T> &x),
//           { x.setZero(); })
// TRACTOR_D(reverse, capture_constraint,
//           (Vector3<T> & a, uint64_t &shape_id, const Vector3<T> &x), {
//             a.setZero();
//             shape_id = 0;
//           })
// TRACTOR_D(project, capture_constraint,
//           (const Vector3<T> &a, const uint64_t &shape_id, const Vector3<T>
//           &da,
//            const T &padding, Vector3<T> &dx),
//           {
//             dx = da;
//             // dx = shape.center() - a;
//           })
// TRACTOR_D(barrier_init, capture_constraint,
//           (const Vector3<T> &a, const uint64_t &shape_id, const Vector3<T>
//           &da,
//            Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &dx,
//            Eigen::Matrix<T, 3, 3, Eigen::DontAlign> &ddx),
//           {
//             throw std::runtime_error("NYI capture_constraint");
//             /*
//           auto &shape = *(CollisionShape<T> *)shape_id;
//           BarrierTypes<T>::Vector3 rdx;
//           BarrierTypes<T>::Matrix3 rddx;
//           shape.capture(a + da, rdx, rddx);
//           dx = rdx;
//           ddx = rddx;
//           */
//           })
// TRACTOR_D(barrier_step, capture_constraint,
//           (const Eigen::Matrix<T, 3, 3, Eigen::DontAlign> &dda,
//            const Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &da,
//            Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &dx),
//           { dx = dda * da; })
// TRACTOR_D(barrier_diagonal, capture_constraint,
//           (const Eigen::Matrix<T, 3, 3, Eigen::DontAlign> &dda,
//            Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &ddx),
//           { ddx = dda.diagonal(); })

// ------------------------------------------

// TRACTOR_OP(simple_collision_constraint,
//            (const Vector3<T> &a, const Quaternion<T> &qa,
//             const Quaternion<T> &qb, const uint64_t &pair),
//            { return Vector3<T>::Zero(); })
// TRACTOR_D(prepare, simple_collision_constraint,
//           (const Vector3<T> &a, const Quaternion<T> &qa,
//            const Quaternion<T> &qb, const uint64_t &pair, const Vector3<T>
//            &x),
//           {})
// TRACTOR_D(forward, simple_collision_constraint,
//           (const Vector3<T> &a, const Vector3<T> &qa, const Vector3<T> &qb,
//            const uint64_t &pair, Vector3<T> &x),
//           { // x = T(0);
//             x.setZero();
//           })
// TRACTOR_D(reverse, simple_collision_constraint,
//           (Vector3<T> & a, Vector3<T> &qa, Vector3<T> &qb, uint64_t &pair,
//            const Vector3<T> &x),
//           {
//             a.setZero();
//             qa.setZero();
//             qb.setZero();
//             pair = 0;
//           })
// TRACTOR_D(project, simple_collision_constraint,
//           (const Vector3<T> &a, const Quaternion<T> &qa,
//            const Quaternion<T> &qb, const uint64_t &pair, const Vector3<T>
//            &da, const T &padding, Vector3<T> &dx),
//           { throw std::runtime_error("NYI simple_collision_constraint"); })
// template <class T> struct SimpleCollisionConstraintBarrier {
//   Eigen::Matrix<T, 6, 6, Eigen::DontAlign> hessian;
//   uint64_t test = 0;
// };
// TRACTOR_D(barrier_init, simple_collision_constraint,
//           (const Vector3<T> &a, const Quaternion<T> &qa,
//            const Quaternion<T> &qb, const uint64_t &pair, const Vector3<T>
//            &da, Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &dx,
//            SimpleCollisionConstraintBarrier<T> &ddx),
//           { throw std::runtime_error("NYI simple_collision_constraint"); })
// TRACTOR_D(barrier_step, simple_collision_constraint,
//           (const SimpleCollisionConstraintBarrier<T> &dda,
//            // const Twist<T> &da, Twist<T> &dx
//            const Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &da,
//            Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &dx),
//           {
//             // dx.setZero();
//
//             if (dda.test != 123456789) {
//               throw 1;
//             }
//
//             dx = dda.hessian.block(0, 0, 3, 3) * da;
//             // dx = da;
//             // dx = da * T(1);
//           })
// TRACTOR_D(barrier_diagonal, simple_collision_constraint,
//           (const SimpleCollisionConstraintBarrier<T> &dda,
//            Eigen::Matrix<T, 3, 1, Eigen::DontAlign> &ddx),
//           {
//             if (dda.test != 123456789) {
//               throw 1;
//             }
//
//             ddx = dda.hessian.diagonal().head(3);
//             // ddx = Twist<T>(Vector3<T>(1, 1, 1), Vector3<T>(1, 1, 1));
//             // ddx = Twist<T>(Vector3<T>(0, 0, 0), Vector3<T>(0, 0, 0));
//           })

// ------------------------------------------

// #if 1
// TRACTOR_OP(collision_constraint, (const Pose<T> &a, const uint64_t &pair),
//            { return T(0); })
// TRACTOR_D(prepare, collision_constraint,
//           (const Pose<T> &a, const uint64_t &pair, const T &x), {})
// TRACTOR_D(forward, collision_constraint,
//           (const Twist<T> &a, const uint64_t &pair, T &x), { x = T(0); })
// TRACTOR_D(reverse, collision_constraint,
//           (Twist<T> & a, uint64_t &pair, const T &x), {
//             a.setZero();
//             pair = 0;
//           })
// TRACTOR_D(project, collision_constraint,
//           (const Pose<T> &a, const uint64_t &pair, const Twist<T> &da,
//            const T &padding, Twist<T> &dx),
//           { throw std::runtime_error("NYI collision_constraint"); })
// TRACTOR_D(barrier_init, collision_constraint,
//           (const Pose<T> &a, const uint64_t &pair, const Twist<T> &da,
//            Twist<T> &dx, Eigen::Matrix<T, 6, 6, Eigen::DontAlign> &ddx),
//           { throw std::runtime_error("NYI collision_constraint"); })
// TRACTOR_D(barrier_step, collision_constraint,
//           (const Eigen::Matrix<T, 6, 6, Eigen::DontAlign> &dda,
//            const Eigen::Matrix<T, 6, 1, Eigen::DontAlign> &da,
//            Eigen::Matrix<T, 6, 1, Eigen::DontAlign> &dx),
//           {
//             TRACTOR_PROFILER("collision constraint step");
//             dx = dda * da;
//           })
// TRACTOR_D(barrier_diagonal, collision_constraint,
//           (const Eigen::Matrix<T, 6, 6, Eigen::DontAlign> &dda,
//            Eigen::Matrix<T, 6, 1, Eigen::DontAlign> &ddx),
//           {
//             TRACTOR_PROFILER("collision constraint diagonal");
//             ddx = dda.diagonal();
//           })
// #endif
//
// TRACTOR_OP(sphere_collision_constraint, (const Vector3<T> &a, const T
// &radius),
//            { return T(0); })
// TRACTOR_D(prepare, sphere_collision_constraint,
//           (const Vector3<T> &a, const T &radius, const T &x), {})
// TRACTOR_D(forward, sphere_collision_constraint,
//           (const Vector3<T> &a, const T &radius, T &x), { x = T(0); })
// TRACTOR_D(reverse, sphere_collision_constraint,
//           (Vector3<T> & a, T &radius, const T &x), {
//             a.setZero();
//             radius = T(0);
//           })
// TRACTOR_D(project, sphere_collision_constraint,
//           (const Vector3<T> &a, const T &radius, const Vector3<T> &da,
//            const T &padding, Vector3<T> &dx),
//           {
//             Vector3<T> normal = normalized(a);
//             dx = da + normal * std::max(T(0), radius - dot(normal, a + da));
//           })
// TRACTOR_D(barrier_init, sphere_collision_constraint,
//           (const Vector3<T> &a, const T &radius, const Vector3<T> &da,
//            Vector3<T> &dx, Vector3<T> &ddx),
//           {
//             Vector3<T> normal = normalized(a);
//             T d = T(-1) / std::max(T(0), dot(normal, a + da) - radius);
//             dx = normal * d;
//             ddx = normal * (d * d);
//             ddx = normal * d;
//           })
// TRACTOR_D(barrier_step, sphere_collision_constraint,
//           (const Vector3<T> &dda, const Vector3<T> &da, Vector3<T> &dx),
//           { dx = dda * dot(dda, da); })
// TRACTOR_D(barrier_diagonal, sphere_collision_constraint,
//           (const Vector3<T> &dda, Vector3<T> &ddx), {
//             ddx.x() = dda.x() * dda.x();
//             ddx.y() = dda.y() * dda.y();
//             ddx.z() = dda.z() * dda.z();
//           })

} // namespace tractor
