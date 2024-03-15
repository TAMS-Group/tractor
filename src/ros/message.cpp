// 2020-2024 Philipp Ruppel

#include <tractor/ros/message.h>

#include <tractor/core/factory.h>

namespace tractor {

const std::shared_ptr<MessageType> &MessageType::instance(
    const std::string &name, const std::string &hash,
    const std::string &definition) {
  static Factory::Key<std::string,
                      std::string>::Value<std::shared_ptr<MessageType>>
      factory([&](const std::string &name, const std::string &hash) {
        return std::shared_ptr<MessageType>(
            new MessageType(name, hash, definition));
      });

  return factory.get(name, hash);
}

}  // namespace tractor
