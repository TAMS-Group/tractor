// (c) 2020-2022 Philipp Ruppel

#include <tractor/ros/publish.h>

namespace tractor {

void publish(const std::string &topic, const Message &message) {

  static ros::NodeHandle node_handle("~");

  static Factory::Key<std::string>::Value<ros::Publisher> factory(
      [&](const std::string &topic) {
        ros::AdvertiseOptions advertise_options(
            topic, 10, message.type()->hash(), message.type()->name(),
            message.type()->definition());
        ros::Publisher publisher = node_handle.advertise(advertise_options);
        return publisher;
      });

  factory.get(topic).publish(message);
}

} // namespace tractor
