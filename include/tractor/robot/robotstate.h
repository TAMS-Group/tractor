// 2020-2024 Philipp Ruppel

#pragma once

#include "jointstate.h"
#include "linkstate.h"
// #include "robotmodel.h"

namespace tractor {

template <class Geometry>
class RobotModel;

template <class Geometry>
class RobotState {
  std::shared_ptr<const RobotModel<Geometry>> _model;
  JointState<Geometry> _joints;
  LinkState<Geometry> _links;

 public:
  void init(const std::shared_ptr<const RobotModel<Geometry>> &robot_model) {
    _model = robot_model;
    _joints.init(robot_model);
    _links.init(robot_model);
  }
  RobotState() {}
  RobotState(const std::shared_ptr<const RobotModel<Geometry>> &robot_model) {
    init(robot_model);
  }
  auto &model() const { return _model; }
  auto &links() const { return _links; }
  auto &links() { return _links; }
  auto &joints() const { return _joints; }
  auto &joints() { return _joints; }
  template <class RobotState>
  void toMoveIt(RobotState &moveit_state) const {
    joints().toMoveIt(moveit_state);
  }
  template <class RobotState>
  void fromMoveIt(RobotState &moveit_state) {
    joints().fromMoveIt(moveit_state);
  }
};

}  // namespace tractor
