// (c) 2022 Philipp Ruppel

#pragma once

#include "robot.h"

#include <tractor/geometry/convert.h>
#include <tractor/geometry/pose.h>
#include <tractor/geometry/vector3.h>

namespace tractor {

template <class T>
void collision_axes(const Pose<T> &pose_a, const Pose<T> &pose_b,
                    const uint64_t &shape_a, const uint64_t &shape_b,
                    Vector3<T> &point_a, Vector3<T> &point_b, Vector3<T> &axis,
                    Vector3<T> &local_a, Vector3<T> &local_b) {

  CollisionRequest req;
  req.pose_a = toEigenIsometry3d(pose_a);
  req.shape_a = (const CollisionShape *)shape_a;
  req.pose_b = toEigenIsometry3d(pose_b);
  req.shape_b = (const CollisionShape *)shape_b;

  CollisionResponse res;
  ((const CollisionShape *)shape_a)->engine()->collide(req, res);

  point_a = toVector3<T>(res.point_a);
  point_b = toVector3<T>(res.point_b);
  axis = toVector3<T>(res.normal);
  local_a = pose_a.inverse() * point_a;
  local_b = pose_b.inverse() * point_b;
  // local_a = point_a;
  // local_b = point_b;
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

template <class Geometry> struct CollisionResult {
  typename Geometry::Vector3 point_a;
  typename Geometry::Vector3 point_b;
  typename Geometry::Vector3 normal;
  typename Geometry::Scalar distance;
};

template <class Geometry>
CollisionResult<Geometry>
collide(const typename Geometry::Pose &pose_a,
        const std::shared_ptr<const CollisionShape> &shape_a,
        const typename Geometry::Pose &pose_b,
        const std::shared_ptr<const CollisionShape> &shape_b) {
  if (auto *rec = Recorder::instance()) {
    rec->reference(shape_a);
    rec->reference(shape_b);
  }
  typename Geometry::Vector3 global_a, global_b, axis, local_a, local_b;
  collision_axes(pose_a, pose_b, (uint64_t)shape_a.get(),
                 (uint64_t)shape_b.get(), global_a, global_b, axis, local_a,
                 local_b);
  CollisionResult<Geometry> ret;
  ret.point_a = pose_a * local_a;
  ret.point_b = pose_b * local_b;
  ret.normal = axis;
  ret.distance = dot(ret.point_a - ret.point_b, axis);
  return ret;
}

template <class Geometry>
std::vector<CollisionResult<Geometry>>
collide(const typename Geometry::Pose &pose_a,
        const std::shared_ptr<const CollisionLink> &link_a,
        const typename Geometry::Pose &pose_b,
        const std::shared_ptr<const CollisionLink> &link_b) {
  std::vector<CollisionResult<Geometry>> ret;
  for (auto &shape_a : link_a->shapes()) {
    for (auto &shape_b : link_b->shapes()) {
      ret.push_back(collide<Geometry>(pose_a, shape_a, pose_b, shape_b));
    }
  }
  return ret;
}

} // namespace tractor
