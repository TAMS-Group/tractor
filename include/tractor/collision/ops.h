// 2022-2024 Philipp Ruppel

#pragma once

#include "robot.h"

#include <tractor/geometry/pose.h>
#include <tractor/geometry/twist.h>
#include <tractor/geometry/vector3.h>

namespace tractor {

template <class T>
void collision_axes(const Pose<T> &pose_a, const Pose<T> &pose_b,
                    const uint64_t &shape_a, const uint64_t &shape_b,
                    const uint64_t &guess, Vector3<T> &point_a,
                    Vector3<T> &point_b, Vector3<T> &axis, Vector3<T> &local_a,
                    Vector3<T> &local_b) {
  CollisionRequest req;
  req.pose_a = Pose3d(pose_a);
  req.shape_a = (const CollisionShape *)shape_a;
  req.pose_b = Pose3d(pose_b);
  req.shape_b = (const CollisionShape *)shape_b;
  req.guess = (const Vec3d *)guess;

  CollisionResponse res;
  ((const CollisionShape *)shape_a)->engine()->collide(req, res);

  point_a = Vector3<T>(res.point_a);
  point_b = Vector3<T>(res.point_b);
  axis = Vector3<T>(res.normal);
  local_a = pose_a.inverse() * point_a;
  local_b = pose_b.inverse() * point_b;

  *(Vec3d *)guess = res.guess;
}

template <class T, size_t S>
void collision_axes(const Pose<Batch<T, S>> &pose_a,
                    const Pose<Batch<T, S>> &pose_b, const uint64_t &shape_a,
                    const uint64_t &shape_b, const uint64_t &guess,
                    Vector3<Batch<T, S>> &point_a,
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
                   shape_b, guess, point_a_i, point_b_i, axis_i, local_a_i,
                   local_b_i);
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
            const uint64_t &guess, Vector3<T> &point_a, Vector3<T> &point_b,
            Vector3<T> &axis, Vector3<T> &local_a, Vector3<T> &local_b),
           {
             collision_axes(pose_a, pose_b, shape_a, shape_b, guess, point_a,
                            point_b, axis, local_a, local_b);
           })
TRACTOR_D(prepare, collision_axes,
          (const Pose<T> &pose_a, const Pose<T> &pose_b,
           const uint64_t &shape_a, const uint64_t &shape_b,
           const uint64_t &guess, const Vector3<T> &point_a,
           const Vector3<T> &point_b, const Vector3<T> &axis,
           const Vector3<T> &local_a, const Vector3<T> &local_b),
          {})
TRACTOR_D(forward, collision_axes,
          (const Twist<T> &pose_a, const Twist<T> &pose_b,
           const uint64_t &shape_a, const uint64_t &shape_b,
           const uint64_t &guess, Vector3<T> &point_a, Vector3<T> &point_b,
           Vector3<T> &axis, Vector3<T> &local_a, Vector3<T> &local_b),
          {
            point_a.setZero();
            point_b.setZero();
            axis.setZero();
            local_a.setZero();
            local_b.setZero();
          })
TRACTOR_D(reverse, collision_axes,
          (Twist<T> & pose_a, Twist<T> &pose_b, uint64_t &shape_a,
           uint64_t &shape_b, uint64_t &guess, const Vector3<T> &point_a,
           const Vector3<T> &point_b, const Vector3<T> &axis,
           const Vector3<T> &local_a, const Vector3<T> &local_b),
          {
            pose_a.setZero();
            pose_b.setZero();
            shape_a = 0;
            shape_b = 0;
            guess = 0;
          })

template <class Geometry>
struct CollisionResult {
  typename Geometry::Vector3 point_a;
  typename Geometry::Vector3 point_b;
  typename Geometry::Vector3 normal;
  typename Geometry::Scalar distance;
};

template <class Geometry>
CollisionResult<Geometry> collide(
    const typename Geometry::Pose &pose_a,
    const std::shared_ptr<const CollisionShape> &shape_a,
    const typename Geometry::Pose &pose_b,
    const std::shared_ptr<const CollisionShape> &shape_b) {
  Vec3d *guess = nullptr;
  if (auto *rec = Recorder::instance()) {
    rec->reference(shape_a);
    rec->reference(shape_b);
    auto guess_instance = std::make_shared<Vec3d>(normalized(Vec3d(1, 2, 3)));
    rec->reference(guess_instance);
    guess = guess_instance.get();
  }
  typename Geometry::Vector3 global_a, global_b, axis, local_a, local_b;
  collision_axes(pose_a, pose_b, (uint64_t)shape_a.get(),
                 (uint64_t)shape_b.get(), (uint64_t)guess, global_a, global_b,
                 axis, local_a, local_b);
  CollisionResult<Geometry> ret;
  ret.point_a = pose_a * local_a;
  ret.point_b = pose_b * local_b;
  ret.normal = axis;
  ret.distance = dot(ret.point_a - ret.point_b, axis);
  // if (guess) {
  //   auto a = value(axis);
  //   guess->x() = a.x();
  //   guess->y() = a.y();
  //   guess->z() = a.z();
  // }
  return ret;
}

// -------------------------------------------------------------

template <class T>
void continuous_collision_axes(  //
    const Pose<T> &pose_a_0,     //
    const Pose<T> &pose_a_1,     //
    const Pose<T> &pose_b_0,     //
    const Pose<T> &pose_b_1,     //
    const uint64_t &shape_a,     //
    const uint64_t &shape_b,     //
    Vector3<T> &axis,            //
    Vector3<T> &local_a_0,       //
    Vector3<T> &local_a_1,       //
    Vector3<T> &local_b_0,       //
    Vector3<T> &local_b_1        //
) {
  ContinuousCollisionRequest req;
  req.pose_a_0 = Pose3d(pose_a_0);
  req.pose_a_1 = Pose3d(pose_a_1);
  req.shape_a = (const CollisionShape *)shape_a;
  req.pose_b_0 = Pose3d(pose_b_0);
  req.pose_b_1 = Pose3d(pose_b_1);
  req.shape_b = (const CollisionShape *)shape_b;

  ContinuousCollisionResponse res;
  ((const CollisionShape *)shape_a)->engine()->collide(req, res);

  axis = Vector3<T>(res.normal);
  local_a_0 = pose_a_0.inverse() * Vector3<T>(res.point_a_0);
  local_a_1 = pose_a_1.inverse() * Vector3<T>(res.point_a_1);
  local_b_0 = pose_b_0.inverse() * Vector3<T>(res.point_b_0);
  local_b_1 = pose_b_1.inverse() * Vector3<T>(res.point_b_1);
}

template <class T, size_t S>
void continuous_collision_axes(         //
    const Pose<Batch<T, S>> &pose_a_0,  //
    const Pose<Batch<T, S>> &pose_a_1,  //
    const Pose<Batch<T, S>> &pose_b_0,  //
    const Pose<Batch<T, S>> &pose_b_1,  //
    const uint64_t &shape_a,            //
    const uint64_t &shape_b,            //
    Vector3<Batch<T, S>> &axis,         //
    Vector3<Batch<T, S>> &local_a_0,    //
    Vector3<Batch<T, S>> &local_a_1,    //
    Vector3<Batch<T, S>> &local_b_0,    //
    Vector3<Batch<T, S>> &local_b_1     //
) {
  auto insertBatch = [](const Vector3<T> &v, size_t i,
                        Vector3<Batch<T, S>> &b) {
    b.x()[i] = v.x();
    b.y()[i] = v.y();
    b.z()[i] = v.z();
  };
  for (size_t i = 0; i < S; i++) {
    Vector3<T> _axis_i;
    Vector3<T> _local_a_0;
    Vector3<T> _local_a_1;
    Vector3<T> _local_b_0;
    Vector3<T> _local_b_1;
    continuous_collision_axes(    //
        indexBatch(pose_a_0, i),  //
        indexBatch(pose_a_1, i),  //
        indexBatch(pose_b_0, i),  //
        indexBatch(pose_b_1, i),  //
        shape_a,                  //
        shape_b,                  //
        _axis_i,                  //
        _local_a_0,               //
        _local_a_1,               //
        _local_b_0,               //
        _local_b_1                //
    );
    insertBatch(_axis_i, i, axis);
    insertBatch(_local_a_0, i, local_a_0);
    insertBatch(_local_a_1, i, local_a_1);
    insertBatch(_local_b_0, i, local_b_0);
    insertBatch(_local_b_1, i, local_b_1);
  }
}

TRACTOR_OP(continuous_collision_axes,
           (                             //
               const Pose<T> &pose_a_0,  //
               const Pose<T> &pose_a_1,  //
               const Pose<T> &pose_b_0,  //
               const Pose<T> &pose_b_1,  //
               const uint64_t &shape_a,  //
               const uint64_t &shape_b,  //
               Vector3<T> &axis,         //
               Vector3<T> &local_a_0,    //
               Vector3<T> &local_a_1,    //
               Vector3<T> &local_b_0,    //
               Vector3<T> &local_b_1     //
               ),
           {
             continuous_collision_axes(  //
                 pose_a_0,               //
                 pose_a_1,               //
                 pose_b_0,               //
                 pose_b_1,               //
                 shape_a,                //
                 shape_b,                //
                 axis,                   //
                 local_a_0,              //
                 local_a_1,              //
                 local_b_0,              //
                 local_b_1               //
             );
           })
TRACTOR_D(prepare, continuous_collision_axes,
          (                                 //
              const Pose<T> &pose_a_0,      //
              const Pose<T> &pose_a_1,      //
              const Pose<T> &pose_b_0,      //
              const Pose<T> &pose_b_1,      //
              const uint64_t &shape_a,      //
              const uint64_t &shape_b,      //
              const Vector3<T> &axis,       //
              const Vector3<T> &local_a_0,  //
              const Vector3<T> &local_a_1,  //
              const Vector3<T> &local_b_0,  //
              const Vector3<T> &local_b_1   //
              ),
          {})
TRACTOR_D(forward, continuous_collision_axes,
          (                              //
              const Twist<T> &pose_a_0,  //
              const Twist<T> &pose_a_1,  //
              const Twist<T> &pose_b_0,  //
              const Twist<T> &pose_b_1,  //
              const uint64_t &shape_a,   //
              const uint64_t &shape_b,   //
              Vector3<T> &axis,          //
              Vector3<T> &local_a_0,     //
              Vector3<T> &local_a_1,     //
              Vector3<T> &local_b_0,     //
              Vector3<T> &local_b_1      //
              ),
          {
            axis.setZero();
            local_a_0.setZero();
            local_a_1.setZero();
            local_b_0.setZero();
            local_b_1.setZero();
          })
TRACTOR_D(reverse, continuous_collision_axes,
          (                                 //
              Twist<T> & pose_a_0,          //
              Twist<T> &pose_a_1,           //
              Twist<T> &pose_b_0,           //
              Twist<T> &pose_b_1,           //
              uint64_t &shape_a,            //
              uint64_t &shape_b,            //
              const Vector3<T> &axis,       //
              const Vector3<T> &local_a_0,  //
              const Vector3<T> &local_a_1,  //
              const Vector3<T> &local_b_0,  //
              const Vector3<T> &local_b_1   //
              ),
          {
            pose_a_0.setZero();
            pose_a_1.setZero();
            pose_b_0.setZero();
            pose_b_1.setZero();
            shape_a = 0;
            shape_b = 0;
          })

template <class Geometry>
struct ContinuousCollisionResult {
  typename Geometry::Vector3 point_a_0;
  typename Geometry::Vector3 point_a_1;
  typename Geometry::Vector3 point_b_0;
  typename Geometry::Vector3 point_b_1;
  typename Geometry::Vector3 normal;
};

template <class Geometry>
ContinuousCollisionResult<Geometry> collide(               //
    const typename Geometry::Pose &pose_a_0,               //
    const typename Geometry::Pose &pose_a_1,               //
    const std::shared_ptr<const CollisionShape> &shape_a,  //
    const typename Geometry::Pose &pose_b_0,               //
    const typename Geometry::Pose &pose_b_1,               //
    const std::shared_ptr<const CollisionShape> &shape_b   //
) {
  if (auto *rec = Recorder::instance()) {
    rec->reference(shape_a);
    rec->reference(shape_b);
  }
  typename Geometry::Vector3 axis;
  typename Geometry::Vector3 local_a_0;
  typename Geometry::Vector3 local_a_1;
  typename Geometry::Vector3 local_b_0;
  typename Geometry::Vector3 local_b_1;
  continuous_collision_axes(    //
      pose_a_0,                 //
      pose_a_1,                 //
      pose_b_0,                 //
      pose_b_1,                 //
      (uint64_t)shape_a.get(),  //
      (uint64_t)shape_b.get(),  //
      axis,                     //
      local_a_0,                //
      local_a_1,                //
      local_b_0,                //
      local_b_1                 //
  );
  ContinuousCollisionResult<Geometry> ret;
  ret.point_a_0 = pose_a_0 * local_a_0;
  ret.point_a_1 = pose_a_1 * local_a_1;
  ret.point_b_0 = pose_b_0 * local_b_0;
  ret.point_b_1 = pose_b_1 * local_b_1;
  ret.normal = axis;
  return ret;
}

// -------------------------------------------------------------

template <class T>
void link_collision_axes(const Pose<T> &pose_a, const Pose<T> &pose_b,
                         const uint64_t &link_a, const uint64_t &link_b,
                         Vector3<T> &point_a, Vector3<T> &point_b,
                         Vector3<T> &axis, Vector3<T> &local_a,
                         Vector3<T> &local_b) {
  CollisionResponse best_res;
  bool first_res = true;

  for (auto &shape_a : ((const CollisionLink *)link_a)->shapes()) {
    for (auto &shape_b : ((const CollisionLink *)link_b)->shapes()) {
      CollisionRequest req;
      req.pose_a = Pose3d(pose_a);
      req.shape_a = shape_a.get();
      req.pose_b = Pose3d(pose_b);
      req.shape_b = shape_b.get();

      CollisionResponse res;
      shape_a->engine()->collide(req, res);

      if (first_res || res.distance < best_res.distance) {
        best_res = res;
        first_res = false;
      }
    }
  }

  point_a = Vector3<T>(best_res.point_a);
  point_b = Vector3<T>(best_res.point_b);
  axis = Vector3<T>(best_res.normal);
  local_a = pose_a.inverse() * point_a;
  local_b = pose_b.inverse() * point_b;
}

template <class T, size_t S>
void link_collision_axes(const Pose<Batch<T, S>> &pose_a,
                         const Pose<Batch<T, S>> &pose_b,
                         const uint64_t &link_a, const uint64_t &link_b,
                         Vector3<Batch<T, S>> &point_a,
                         Vector3<Batch<T, S>> &point_b,
                         Vector3<Batch<T, S>> &axis,
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
    link_collision_axes(indexBatch(pose_a, i), indexBatch(pose_b, i), link_a,
                        link_b, point_a_i, point_b_i, axis_i, local_a_i,
                        local_b_i);
    insertBatch(point_a_i, i, point_a);
    insertBatch(point_b_i, i, point_b);
    insertBatch(axis_i, i, axis);
    insertBatch(local_a_i, i, local_a);
    insertBatch(local_b_i, i, local_b);
  }
}

TRACTOR_OP(link_collision_axes,
           (const Pose<T> &pose_a, const Pose<T> &pose_b,
            const uint64_t &link_a, const uint64_t &link_b, Vector3<T> &point_a,
            Vector3<T> &point_b, Vector3<T> &axis, Vector3<T> &local_a,
            Vector3<T> &local_b),
           {
             link_collision_axes(pose_a, pose_b, link_a, link_b, point_a,
                                 point_b, axis, local_a, local_b);
           })
TRACTOR_D(prepare, link_collision_axes,
          (const Pose<T> &pose_a, const Pose<T> &pose_b, const uint64_t &link_a,
           const uint64_t &link_b, const Vector3<T> &point_a,
           const Vector3<T> &point_b, const Vector3<T> &axis,
           const Vector3<T> &local_a, const Vector3<T> &local_b),
          {})
TRACTOR_D(forward, link_collision_axes,
          (const Twist<T> &pose_a, const Twist<T> &pose_b,
           const uint64_t &link_a, const uint64_t &link_b, Vector3<T> &point_a,
           Vector3<T> &point_b, Vector3<T> &axis, Vector3<T> &local_a,
           Vector3<T> &local_b),
          {
            point_a.setZero();
            point_b.setZero();
            axis.setZero();
            local_a.setZero();
            local_b.setZero();
          })
TRACTOR_D(reverse, link_collision_axes,
          (Twist<T> & pose_a, Twist<T> &pose_b, uint64_t &link_a,
           uint64_t &link_b, const Vector3<T> &point_a,
           const Vector3<T> &point_b, const Vector3<T> &axis,
           const Vector3<T> &local_a, const Vector3<T> &local_b),
          {
            pose_a.setZero();
            pose_b.setZero();
            link_a = 0;
            link_b = 0;
          })

template <class Geometry>
CollisionResult<Geometry> collide(
    const typename Geometry::Pose &pose_a,
    const std::shared_ptr<const CollisionLink> &link_a,
    const typename Geometry::Pose &pose_b,
    const std::shared_ptr<const CollisionLink> &link_b) {
  if (auto *rec = Recorder::instance()) {
    rec->reference(link_a);
    rec->reference(link_b);
  }
  typename Geometry::Vector3 global_a, global_b, axis, local_a, local_b;
  link_collision_axes(pose_a, pose_b, (uint64_t)link_a.get(),
                      (uint64_t)link_b.get(), global_a, global_b, axis, local_a,
                      local_b);
  CollisionResult<Geometry> ret;
  ret.point_a = pose_a * local_a;
  ret.point_b = pose_b * local_b;
  ret.normal = axis;
  ret.distance = dot(ret.point_a - ret.point_b, axis);
  return ret;
}

// -------------------------------------------------------------

// , class X = decltype(decltype(value(std::declval<T>()))(
//                       std::declval<T>()))
template <class T>
static void collision_project(const Vector3<T> &point, const uint64_t &shape_id,
                              Vector3<T> &closest_point,
                              Vector3<T> &surface_normal) {
  auto *shape = (CollisionShape *)shape_id;
  Vec3d i_p = Vec3d(point.x(), point.y(), point.z());
  Vec3d o_p, o_n;
  shape->project(i_p, o_p, o_n);
  closest_point.x() = o_p.x();
  closest_point.y() = o_p.y();
  closest_point.z() = o_p.z();
  surface_normal.x() = o_n.x();
  surface_normal.y() = o_n.y();
  surface_normal.z() = o_n.z();
}

template <class T, size_t S>
static void collision_project(const Vector3<Batch<T, S>> &point,
                              const uint64_t &shape_id,
                              Vector3<Batch<T, S>> &closest_point,
                              Vector3<Batch<T, S>> &surface_normal) {
  auto *shape = (CollisionShape *)shape_id;
  for (size_t i = 0; i < S; i++) {
    Vec3d i_p = Vec3d(point.x()[i], point.y()[i], point.z()[i]);
    Vec3d o_p, o_n;
    shape->project(i_p, o_p, o_n);
    closest_point.x()[i] = o_p.x();
    closest_point.y()[i] = o_p.y();
    closest_point.z()[i] = o_p.z();
    surface_normal.x()[i] = o_n.x();
    surface_normal.y()[i] = o_n.y();
    surface_normal.z()[i] = o_n.z();
  }
}

TRACTOR_OP(collision_project,
           (const Vector3<T> &point, const uint64_t &shape_id,
            Vector3<T> &out_point, Vector3<T> &out_normal),
           { collision_project(point, shape_id, out_point, out_normal); })
TRACTOR_D(prepare, collision_project,
          (const Vector3<T> &point, const uint64_t &shape_id,
           const Vector3<T> &out_point, const Vector3<T> &out_normal),
          {})
TRACTOR_D(forward, collision_project,
          (const Vector3<T> &point, const uint64_t &shape_id,
           Vector3<T> &out_point, Vector3<T> &out_normal),
          {
            out_point.setZero();
            out_normal.setZero();
          })
TRACTOR_D(reverse, collision_project,
          (Vector3<T> & point, uint64_t &shape_id, const Vector3<T> &out_point,
           const Vector3<T> &out_normal),
          {
            point.setZero();
            shape_id = 0;
          })

template <class Geometry>
void project(const typename Geometry::Vector3 &point,
             const std::shared_ptr<const CollisionShape> &shape,
             typename Geometry::Vector3 &out_point,
             typename Geometry::Vector3 &out_normal) {
  collision_project(point, (uint64_t)shape.get(), out_point, out_normal);
}

}  // namespace tractor
