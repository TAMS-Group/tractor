// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/loader.h>

#include <tractor/core/error.h>
#include <tractor/core/log.h>

#include <geometric_shapes/mesh_operations.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include <LinearMath/btConvexHullComputer.h>
#include <LinearMath/btGeometryUtil.h>

namespace tractor {

void _loadCollisionLink(CollisionRobot *collision_robot,
                        const moveit::core::LinkModel *link_model,
                        const Eigen::Isometry3d &link_transform,
                        std::shared_ptr<CollisionLink> collision_link) {

  auto new_collision_link =
      std::make_shared<CollisionLink>(link_model->getName());
  collision_robot->addLink(new_collision_link);

  if (!collision_link) {
    collision_link = new_collision_link;
  }

  auto &shapes = link_model->getShapes();
  auto &origins = link_model->getCollisionOriginTransforms();
  for (size_t shape_index = 0; shape_index < shapes.size(); shape_index++) {
    auto &shape = shapes[shape_index];
    TRACTOR_DEBUG("link " << link_model->getName() << " shape "
                          << typeid(*shape).name());
    auto &shape_origin = origins[shape_index];
    Eigen::Affine3d shape_pose = Eigen::Affine3d(link_transform) * shape_origin;
    auto collision_shape = collision_robot->engine()->create(
        link_model->getName() + "_" + std::to_string(shape_index), shape_pose,
        shape.get());
    collision_link->addShape(collision_shape);
  }

  for (auto *child_joint : link_model->getChildJointModels()) {
    auto *child_link = child_joint->getChildLinkModel();
    if (child_joint->getType() == moveit::core::JointModel::FIXED) {
      _loadCollisionLink(
          collision_robot, child_link,
          Eigen::Isometry3d(
              (link_transform * child_link->getJointOriginTransform())
                  .matrix()),
          collision_link);
    } else {
      _loadCollisionLink(collision_robot, child_link,
                         Eigen::Isometry3d::Identity(), nullptr);
    }
  }
}

void loadCollisionRobot(const std::shared_ptr<const CollisionEngine> &engine,
                        const moveit::core::RobotModel &moveit_model,
                        CollisionRobot *collision_robot) {

  TRACTOR_ASSERT(collision_robot->links().empty());

  _loadCollisionLink(collision_robot,
                     moveit_model.getRootJoint()->getChildLinkModel(),
                     Eigen::Isometry3d::Identity(), nullptr);
}

} // namespace tractor
