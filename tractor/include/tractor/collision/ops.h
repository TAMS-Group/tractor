// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/collision/query.h>

#include <memory>

namespace tractor {

template <class T>
void collision_axes(const Pose<T> &pose_a, const Pose<T> &pose_b,
                    const uint64_t &shape_a, const uint64_t &shape_b,
                    Vector3<T> &point_a, Vector3<T> &point_b, Vector3<T> &axis,
                    Vector3<T> &local_a, Vector3<T> &local_b) {
  auto a = tractor::internal::CollisionShapeSupport<T>(
      pose_a, (CollisionShape<T> *)shape_a);
  auto b = tractor::internal::CollisionShapeSupport<T>(
      pose_b, (CollisionShape<T> *)shape_b);
  tractor::internal::CollisionResult r;
  tractor::internal::doCollisionQuery(a, b, r);
  point_a = Vector3<T>(T(r.ax), T(r.ay), T(r.az));
  point_b = Vector3<T>(T(r.bx), T(r.by), T(r.bz));
  axis = normalized(Vector3<T>(T(r.nx), T(r.ny), T(r.nz)));
  local_a = pose_a.inverse() * point_a;
  local_b = pose_b.inverse() * point_b;
}

template <class T, size_t S>
void collision_axes(const Pose<Batch<T, S>> &pose_a,
                    const Pose<Batch<T, S>> &pose_b, const uint64_t &shape_a,
                    const uint64_t &shape_b, Vector3<Batch<T, S>> &point_a,
                    Vector3<Batch<T, S>> &point_b, Vector3<Batch<T, S>> &axis,
                    Vector3<Batch<T, S>> &local_a,
                    Vector3<Batch<T, S>> &local_b) {
  auto insertBatch = [](const Vector3<T> &v, size_t i,
                        Vector3<Batch<T, S>> &b) {
    b.x()[i] = v.x();
    b.y()[i] = v.y();
    b.z()[i] = v.z();
  };
  for (size_t i = 0; i < S; i++) {
    Vector3<T> point_a_i, point_b_i, axis_i, local_a_i, local_b_i;
    collision_axes(indexBatch(pose_a, i), indexBatch(pose_b, i), shape_a,
                   shape_b, point_a_i, point_b_i, axis_i, local_a_i, local_b_i);
    insertBatch(point_a_i, i, point_a);
    insertBatch(point_b_i, i, point_b);
    insertBatch(axis_i, i, axis);
    insertBatch(local_a_i, i, local_a);
    insertBatch(local_b_i, i, local_b);
  }
}

TRACTOR_OP(collision_axes,
           (const Pose<T> &pose_a, const Pose<T> &pose_b,
            const uint64_t &shape_a, const uint64_t &shape_b,
            Vector3<T> &point_a, Vector3<T> &point_b, Vector3<T> &axis,
            Vector3<T> &local_a, Vector3<T> &local_b),
           {
             collision_axes(pose_a, pose_b, shape_a, shape_b, point_a, point_b,
                            axis, local_a, local_b);
           })
TRACTOR_D(prepare, collision_axes,
          (const Pose<T> &pose_a, const Pose<T> &pose_b,
           const uint64_t &shape_a, const uint64_t &shape_b,
           const Vector3<T> &point_a, const Vector3<T> &point_b,
           const Vector3<T> &axis, const Vector3<T> &local_a,
           const Vector3<T> &local_b),
          {})
TRACTOR_D(forward, collision_axes,
          (const Twist<T> &pose_a, const Twist<T> &pose_b,
           const uint64_t &shape_a, const uint64_t &shape_b,
           Vector3<T> &point_a, Vector3<T> &point_b, Vector3<T> &axis,
           Vector3<T> &local_a, Vector3<T> &local_b),
          {
            point_a.setZero();
            point_b.setZero();
            axis.setZero();
            local_a.setZero();
            local_b.setZero();
          })
TRACTOR_D(reverse, collision_axes,
          (Twist<T> & pose_a, Twist<T> &pose_b, uint64_t &shape_a,
           uint64_t &shape_b, const Vector3<T> &point_a,
           const Vector3<T> &point_b, const Vector3<T> &axis,
           const Vector3<T> &local_a, const Vector3<T> &local_b),
          {
            pose_a.setZero();
            pose_b.setZero();
            shape_a = 0;
            shape_b = 0;
          })

// ------------------------------------------

} // namespace tractor
