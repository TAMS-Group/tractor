// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>
#include <tractor/core/error.h>
#include <tractor/robot/robotmodel.h>
#include <tractor/robot/robotstate.h>
#include <tractor/ros/publish.h>

#include <moveit_msgs/DisplayRobotState.h>
#include <moveit_msgs/DisplayTrajectory.h>

namespace tractor {

void clearVisualization();

void visualizePoints(const std::string &name, double scale,
                     const Eigen::Vector4d &color,
                     const std::vector<Eigen::Vector3d> &points);

void visualizeLines(const std::string &name, double scale,
                    const Eigen::Vector4d &color,
                    const std::vector<Eigen::Vector3d> &points);

void visualizePoints(const std::string &name, double scale,
                     const std::vector<Eigen::Vector4d> &colors,
                     const std::vector<Eigen::Vector3d> &points);

void visualizeMesh(const std::string &name, const Eigen::Vector4d &color,
                   const std::vector<Eigen::Vector3d> &vertices);

void visualizeMesh(const std::string &name,
                   const std::vector<Eigen::Vector4d> &colors,
                   const std::vector<Eigen::Vector3d> &points);

void visualizeLines(const std::string &name, double scale,
                    const std::vector<Eigen::Vector4d> &colors,
                    const std::vector<Eigen::Vector3d> &points);

void visualizeText(const std::string &name, double scale,
                   const Eigen::Vector4d &color,
                   const Eigen::Vector3d &position, const std::string &text);

template <class Geometry>
void visualize(const std::string &topic,
               const std::vector<JointState<Geometry>> &trajectory,
               const typename Geometry::Value &time_step) {
  if (trajectory.empty()) {
    TRACTOR_DEBUG("trajectory is empty, nothing to visualize");
    return;
  }

  auto robot_model = trajectory.front().model();
  for (auto &state : trajectory) {
    assert(robot_model == state.model());
  }

  trajectory_msgs::JointTrajectory joint_trajectory;
  joint_trajectory.joint_names = robot_model->info()->variables().names();
  for (size_t state_index = 0; state_index < trajectory.size(); state_index++) {
    auto &state = trajectory[state_index];
    AlignedStdVector<typename Geometry::Scalar> positions;
    state.serializePositions(positions);
    joint_trajectory.points.emplace_back();
    joint_trajectory.points.back().time_from_start =
        ros::Duration(state_index * time_step);
    for (auto &p : positions) {
      joint_trajectory.points.back().positions.push_back(p.value());
    }
  }

  moveit_msgs::DisplayTrajectory display_trajectory;
  display_trajectory.trajectory.emplace_back();
  display_trajectory.trajectory.back().joint_trajectory = joint_trajectory;

  publish(topic, display_trajectory);
}

template <class Geometry>
void visualize(const std::string &topic, const JointState<Geometry> &state) {
  TRACTOR_DEBUG("visualize joint states");

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

}  // namespace tractor
