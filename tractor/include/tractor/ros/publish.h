// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/factory.h>
#include <tractor/ros/message.h>
#include <tractor/robot/robotstate.h>

#include <ros/ros.h>

#include <sensor_msgs/JointState.h>

namespace tractor {

template <class Message>
void publish(const std::string &topic, const Message &message) {
  static ros::NodeHandle node_handle("~");

  static Factory::Key<std::string>::Value<ros::Publisher> factory(
      [&](const std::string &topic) {
        return node_handle.advertise<Message>(topic, 100);
      });

  factory.get(topic).publish(message);
}

void publish(const std::string &topic, const Message &message);

template <class Geometry>
void publish(const std::string &topic, const JointState<Geometry> &state) {
  sensor_msgs::JointState joint_state;
  joint_state.name = state.model()->info()->variables().names();

  AlignedStdVector<typename Geometry::Scalar> positions;
  state.serializePositions(positions);

  TRACTOR_ASSERT(joint_state.name.size() == positions.size());

  for (auto &p : positions) {
    joint_state.position.push_back(firstBatchElement(value(p)));
  }

  joint_state.header.stamp = ros::Time::now();

  publish(topic, joint_state);
}

}  // namespace tractor
