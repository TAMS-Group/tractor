// 2020-2024 Philipp Ruppel

#include <tractor/robot/robotinfo.h>

#include <eigen_conversions/eigen_kdl.h>
#include <kdl/rigidbodyinertia.hpp>
#include <kdl/rotationalinertia.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/robot_model/robot_model.h>
#include <tractor/geometry/eigen.h>

namespace kdl_parser {
KDL::RigidBodyInertia toKdl(urdf::InertialSharedPtr);
}

namespace tractor {

static void _importInertia(KDL::RigidBodyInertia &inertia,
                           const Eigen::Isometry3d &transform,
                           const moveit::core::RobotModel &robot_model,
                           const moveit::core::LinkModel *link_model) {
  if (auto urdf_link = robot_model.getURDF()->getLink(link_model->getName())) {
    if (auto urdf_inertial = urdf_link->inertial) {
      KDL::Frame kdl_frame;
      tf::transformEigenToKDL(transform, kdl_frame);
      KDL::RigidBodyInertia kdl_inertia = kdl_parser::toKdl(urdf_inertial);
      inertia = inertia + kdl_frame * kdl_inertia;
    }
  }

  for (auto *child_joint : link_model->getChildJointModels()) {
    auto *child_link = child_joint->getChildLinkModel();
    if (child_link &&
        child_joint->getType() == moveit::core::JointModel::FIXED) {
      _importInertia(
          inertia,
          Eigen::Isometry3d(
              (transform * child_link->getJointOriginTransform()).matrix()),
          robot_model, child_link);
    }
  }
}

RobotJointInfo::RobotJointInfo(const moveit::core::RobotModel &moveit_robot,
                               const moveit::core::JointModel &moveit_joint)
    : _index(moveit_joint.getJointIndex()),
      _name(moveit_joint.getName()),
      _first_variable_index(moveit_joint.getFirstVariableIndex()),
      _origin(convertEigenToPose<Geometry>(
          Eigen::Isometry3d(moveit_joint.getChildLinkModel()
                                ->getJointOriginTransform()
                                .matrix()))) {
  if (auto *moveit_parent = moveit_joint.getParentLinkModel()) {
    _parent_link_index = moveit_parent->getLinkIndex();
  }

  _child_link_index = moveit_joint.getChildLinkModel()->getLinkIndex();

  if (auto *mimic = moveit_joint.getMimic()) {
    _is_mimic = true;
    _mimic_index = mimic->getJointIndex();
    _mimic_factor = mimic->getMimicFactor();
    _mimic_offset = mimic->getMimicOffset();
  }

  auto &bounds = moveit_joint.getVariableBounds();
  if (!bounds.empty()) {
    _has_bounds = bounds.front().position_bounded_;
    _lower_bound = bounds.front().min_position_;
    _upper_bound = bounds.front().max_position_;
  }

  switch (moveit_joint.getType()) {
    case moveit::core::JointModel::FIXED:
      _type = JointType::Fixed;
      break;
    case moveit::core::JointModel::REVOLUTE: {
      _axis = Geometry::importVector3(
          dynamic_cast<const moveit::core::RevoluteJointModel &>(moveit_joint)
              .getAxis());
      _type = JointType::Revolute;
      break;
    }
    case moveit::core::JointModel::PRISMATIC: {
      _axis = Geometry::importVector3(
          dynamic_cast<const moveit::core::PrismaticJointModel &>(moveit_joint)
              .getAxis());
      _type = JointType::Prismatic;
      break;
    }
    case moveit::core::JointModel::PLANAR:
      _type = JointType::Planar;
      break;
    case moveit::core::JointModel::FLOATING:
      _type = JointType::Floating;
      break;
    default:
      throw std::runtime_error("unsupported joint type");
  }

  {
    KDL::RigidBodyInertia inertia = KDL::RigidBodyInertia::Zero();
    if (moveit_joint.getType() != moveit::core::JointModel::FIXED ||
        &moveit_joint == moveit_robot.getRootJoint()) {
      _importInertia(inertia, Eigen::Isometry3d::Identity(), moveit_robot,
                     moveit_joint.getChildLinkModel());
    }
    auto center = Eigen::Vector3d(inertia.getCOG().x(), inertia.getCOG().y(),
                                  inertia.getCOG().z());
    auto moment = Eigen::Matrix3d(
        Eigen::Map<const Eigen::Matrix3d>(inertia.getRotationalInertia().data));
    _inertia = Inertia<Geometry>(
        Geometry::importVector3(center),
        typename Geometry::Scalar(inertia.getMass()),
        typename Geometry::Scalar(
            (inertia.getMass() > 0.0) ? (1.0 / inertia.getMass()) : 0.0),
        Geometry::importMatrix3(moment),
        Geometry::importMatrix3(moment.inverse().eval()));
  }
}

RobotJointMap::RobotJointMap(const moveit::core::RobotModel &robot_model)
    : RobotIndexMap(robot_model.getJointModelNames()) {
  for (auto &joint : robot_model.getJointModels()) {
    _joint_infos.emplace_back(robot_model, *joint);
  }
  robot_model.getVariableDefaultPositions(_default_positions);
}

RobotInfo::RobotInfo(const moveit::core::RobotModel &robot_model)
    : _joints(robot_model),
      _links(robot_model.getLinkModelNames()),
      _variable_count(robot_model.getVariableCount()),
      _variables(robot_model.getVariableNames()) {}

}  // namespace tractor
