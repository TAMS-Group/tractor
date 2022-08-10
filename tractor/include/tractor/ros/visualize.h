// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/error.h>
#include <tractor/robot/robotmodel.h>
#include <tractor/robot/robotstate.h>
#include <tractor/ros/publish.h>

#include <moveit_msgs/DisplayRobotState.h>

namespace tractor {

void visualizePoints(const std::string &name, double scale,
                     const Eigen::Vector4d &color,
                     const std::vector<Eigen::Vector3d> &points);

template <class Geometry>
void visualize(const std::string &topic, const JointState<Geometry> &state) {

  moveit_msgs::DisplayRobotState display;
  display.state.joint_state.name = state.model()->info()->variables().names();

  AlignedStdVector<typename Geometry::Scalar> positions;
  state.serializePositions(positions);

  TRACTOR_ASSERT(display.state.joint_state.name.size() == positions.size());

  for (auto &p : positions) {
    display.state.joint_state.position.push_back(firstBatchElement(value(p)));
  }

  publish(topic, display);
}

} // namespace tractor
