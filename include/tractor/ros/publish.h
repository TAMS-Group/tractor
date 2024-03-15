// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/factory.h>
#include <tractor/ros/message.h>
#include <tractor/robot/robotstate.h>

#include <ros/ros.h>

#include <sensor_msgs/JointState.h>

namespace tractor {

ros::Publisher advertise(const std::string &topic, const std::string &hash,
                         const std::string &name,
                         const std::string &definition);

template <class Message>
void publish(const std::string &topic, const Message &message) {
  ros::Publisher pub = advertise(topic, ros::message_traits::md5sum(message),
                                 ros::message_traits::datatype(message),
                                 ros::message_traits::definition(message));
  TRACTOR_DEBUG("publish message "
                << topic << " " << ros::message_traits::datatype(message) << " "
                << pub.getTopic() << " " << typeid(Message).name());
  pub.publish(message);
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
