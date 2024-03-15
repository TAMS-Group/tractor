// 2020-2024 Philipp Ruppel

#include <tractor/ros/publish.h>

namespace tractor {

ros::Publisher advertise(const std::string &topic, const std::string &hash,
                         const std::string &name,
                         const std::string &definition) {
  static ros::NodeHandle node_handle("~");
  static Factory::Key<std::string, std::string, std::string,
                      std::string>::Value<ros::Publisher>
      factory([](const std::string &topic, const std::string &hash,
                 const std::string &name, const std::string &definition) {
        TRACTOR_INFO("advertise " << topic);
        ros::AdvertiseOptions advertise_options(topic, 100, hash, name,
                                                definition);
        ros::Publisher publisher = node_handle.advertise(advertise_options);
        ros::spinOnce();
        return publisher;
      });
  return factory.get(topic, hash, name, definition);
}

void publish(const std::string &topic, const Message &message) {
  advertise(topic, message.type()->hash(), message.type()->name(),
            message.type()->definition())
      .publish(message);
}

}  // namespace tractor
