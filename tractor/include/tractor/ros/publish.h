// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/factory.h>
#include <tractor/ros/message.h>

#include <ros/ros.h>

namespace tractor {

template <class Message>
void publish(const std::string &topic, const Message &message) {

  static ros::NodeHandle node_handle("~");

  static Factory::Key<std::string>::Value<ros::Publisher> factory(
      [&](const std::string &topic) {
        return node_handle.advertise<Message>(topic, 10);
      });

  factory.get(topic).publish(message);
}

void publish(const std::string &topic, const Message &message);

} // namespace tractor
