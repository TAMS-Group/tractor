// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/loader.h>

#include <tractor/collision/base.h>
#include <tractor/collision/shape.h>

#include <tractor/core/error.h>
#include <tractor/core/log.h>

#include <geometric_shapes/mesh_operations.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

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

    const shapes::Mesh *mesh = dynamic_cast<const shapes::Mesh *>(shape.get());
    const shapes::Mesh *mesh_cleanup = nullptr;
    if (!mesh) {
      mesh_cleanup = mesh = shapes::createMeshFromShape(shape.get());
    }

    for (size_t i = 0; i < mesh->vertex_count; i++) {
      Eigen::Vector3d &v = ((Eigen::Vector3d *)mesh->vertices)[i];
      v = shape_pose * v;
    }

    auto collision_shape = collision_robot->engine()->createConvexMesh(
        link_model->getName() + "_" + std::to_string(shape_index), mesh);

    collision_link->addShape(
        std::dynamic_pointer_cast<const CollisionShape>(collision_shape));

    if (mesh_cleanup) {
      delete mesh_cleanup;
    }
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
