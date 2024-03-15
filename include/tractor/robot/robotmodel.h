// 2020-2024 Philipp Ruppel

#pragma once

#include "jointtypes.h"
#include "linkmodel.h"
#include "robotinfo.h"

#include <deque>
#include <memory>

namespace tractor {

template <class Geometry>
class JointState;
template <class Geometry>
class LinkState;

template <class Geometry>
class RobotModel {
  std::shared_ptr<const RobotInfo> _robot_info;
  std::vector<std::shared_ptr<const JointModelBase<Geometry>>> _joint_models;
  AlignedStdVector<JointVariant<JointStateBase<Geometry>>>
      _default_joint_states;
  AlignedStdVector<typename Geometry::Scalar> _default_positions;
  std::vector<std::shared_ptr<const LinkModel<Geometry>>> _link_models;

  struct JointInfo {
    ssize_t parent_link_index = -1;
    ssize_t child_link_index = -1;
  };
  std::vector<JointInfo> _joint_infos;

  template <class State>
  void addJoint(const RobotJointInfo &robot_joint_info,
                std::vector<std::shared_ptr<LinkModel<Geometry>>> &link_models,
                const std::shared_ptr<JointModelBase<Geometry>> &joint_model) {
    State joint_state;

    auto inertia = Inertia<Geometry>(
        Geometry::importVector3(robot_joint_info.inertia().center()),
        Geometry::importScalar(robot_joint_info.inertia().mass()),
        Geometry::importScalar(robot_joint_info.inertia().massInverse()),
        Geometry::importMatrix3(robot_joint_info.inertia().moment()),
        Geometry::importMatrix3(robot_joint_info.inertia().momentInverse()));

    auto origin = Geometry::pack(
        Geometry::importVector3(robot_joint_info.origin().position()),
        Geometry::importQuaternion(robot_joint_info.origin().orientation()));

    _joint_models.emplace_back(joint_model);

    joint_state.deserializePositions(_default_positions.data() +
                                     robot_joint_info.firstVariableIndex());
    _default_joint_states.emplace_back(joint_state);

    JointInfo joint_info;
    joint_info.parent_link_index = robot_joint_info.parentLinkIndex();
    joint_info.child_link_index = robot_joint_info.childLinkIndex();
    _joint_infos.emplace_back(joint_info);

    auto child_link = std::make_shared<LinkModel<Geometry>>(
        joint_model,
        _robot_info->links().name(robot_joint_info.childLinkIndex()), inertia);
    link_models.at(robot_joint_info.childLinkIndex()) = child_link;

    std::shared_ptr<LinkModel<Geometry>> parent_link;
    if (robot_joint_info.parentLinkIndex() >= 0) {
      parent_link = link_models.at(robot_joint_info.parentLinkIndex());
      parent_link->addChildJoint(joint_model);
    }

    joint_model->init(parent_link, robot_joint_info.name(), origin, child_link);
  }

 public:
  RobotModel() {}

  RobotModel(const moveit::core::RobotModel &moveit_robot)
      : RobotModel(std::make_shared<RobotInfo>(moveit_robot)) {}

  RobotModel(const std::shared_ptr<const RobotInfo> &robot_info)
      : _robot_info(robot_info) {
    _default_positions.clear();
    for (auto &p : _robot_info->joints().defaultPositions()) {
      _default_positions.push_back(typename Geometry::Value(p));
    }

    std::vector<std::shared_ptr<LinkModel<Geometry>>> link_models;
    link_models.resize(_robot_info->links().size());

    for (size_t joint_index = 0; joint_index < _robot_info->joints().size();
         joint_index++) {
      auto &joint_info = _robot_info->joints().info(joint_index);

      switch (joint_info.type()) {
        case JointType::Fixed: {
          addJoint<FixedJointState<Geometry>>(
              joint_info, link_models,
              std::make_shared<FixedJointModel<Geometry>>());
          break;
        }

        case JointType::Revolute: {
          auto joint_model = std::make_shared<RevoluteJointModel<Geometry>>();
          joint_model->axis() = Geometry::importVector3(joint_info.axis());
          if (joint_info.hasBounds()) {
            joint_model->limits() = JointLimits<Geometry>(
                typename Geometry::Value(joint_info.lowerBound()),
                typename Geometry::Value(joint_info.upperBound()));
          }
          addJoint<RevoluteJointState<Geometry>>(joint_info, link_models,
                                                 joint_model);
          break;
        }

        case JointType::Prismatic: {
          auto joint_model = std::make_shared<PrismaticJointModel<Geometry>>();
          joint_model->axis() = Geometry::importVector3(joint_info.axis());
          if (joint_info.hasBounds()) {
            joint_model->limits() = JointLimits<Geometry>(
                typename Geometry::Value(joint_info.lowerBound()),
                typename Geometry::Value(joint_info.upperBound()));
          }
          addJoint<PrismaticJointState<Geometry>>(joint_info, link_models,
                                                  joint_model);
          break;
        }

        case JointType::Planar: {
          addJoint<PlanarJointState<Geometry>>(
              joint_info, link_models,
              std::make_shared<PlanarJointModel<Geometry>>());
          break;
        }

        case JointType::Floating: {
          addJoint<FloatingJointState<Geometry>>(
              joint_info, link_models,
              std::make_shared<FloatingJointModel<Geometry>>());
          break;
        }

        default:
          throw std::runtime_error("joint type not yet implemented");
      }
    }

    for (auto &link_model : link_models) {
      TRACTOR_ASSERT(link_model != nullptr);
      _link_models.push_back(link_model);
    }
  }

  auto &joint(size_t i) const { return _joint_models.at(i); }
  auto &joint(const std::string &name) const {
    return _joint_models.at(_robot_info->joints().index(name));
  }

  auto &link(size_t i) const { return _link_models.at(i); }
  auto &link(const std::string &name) const {
    return _link_models.at(_robot_info->links().index(name));
  }

  auto &joints() const { return _joint_models; }

  auto &links() const { return _link_models; }

  auto &rootJoint() const { return joint(0); }

  void computeFK(const JointState<Geometry> &joint_state,
                 LinkState<Geometry> &link_state) const {
    for (size_t joint_index = 0; joint_index < _joint_infos.size();
         joint_index++) {
      auto &joint_info = _joint_infos[joint_index];

      if (joint_info.parent_link_index >= 0) {
        link_state.pose(joint_info.child_link_index) =
            joint_state.joint(joint_index)
                .compute(*_joint_models.at(joint_index),
                         link_state.pose(joint_info.parent_link_index) *
                             _joint_models.at(joint_index)->origin());
      } else {
        link_state.pose(joint_info.child_link_index) =
            joint_state.joint(joint_index)
                .compute(*_joint_models.at(joint_index),
                         _joint_models.at(joint_index)->origin());
      }
    }
  }

  const AlignedStdVector<JointVariant<JointStateBase<Geometry>>> &
  defaultJointStates() const {
    return _default_joint_states;
  }

  const std::shared_ptr<const RobotInfo> &info() const { return _robot_info; }

  size_t variableCount() const { return _robot_info->variableCount(); }
};

}  // namespace tractor
