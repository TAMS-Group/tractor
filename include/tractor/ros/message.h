// 2020-2024 Philipp Ruppel

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <ros/ros.h>

namespace tractor {

class MessageType {
  std::string _name, _hash, _definition;
  MessageType(const std::string &name, const std::string &hash,
              const std::string &definition)
      : _name(name), _hash(hash), _definition(definition) {}

 public:
  static const std::shared_ptr<MessageType> &instance(
      const std::string &name, const std::string &hash,
      const std::string &definition);
  const std::string &name() const { return _name; }
  const std::string &hash() const { return _hash; }
  const std::string &definition() const { return _definition; }
};

class Message {
  std::shared_ptr<const MessageType> _type;
  std::vector<uint8_t> _data;

 public:
  Message(const std::shared_ptr<const MessageType> &type, const void *data,
          size_t size)
      : _type(type),
        _data((const uint8_t *)data, (const uint8_t *)data + size) {}
  auto &type() const { return _type; }
  size_t size() const { return _data.size(); }
  const void *data() const { return _data.data(); }
};

}  // namespace tractor

namespace ros {
namespace message_traits {
template <>
struct MD5Sum<tractor::Message> {
  static const char *value() { return "*"; }
  static const char *value(const tractor::Message &message) {
    return message.type()->hash().c_str();
  }
};
template <>
struct DataType<tractor::Message> {
  static const char *value() { return "*"; }
  static const char *value(const tractor::Message &message) {
    return message.type()->name().c_str();
  }
};
template <>
struct Definition<tractor::Message> {
  static const char *value() { return "*"; }
  static const char *value(const tractor::Message &message) {
    return message.type()->definition().c_str();
  }
};
}  // namespace message_traits
namespace serialization {
template <>
struct Serializer<tractor::Message> {
  template <typename Stream>
  inline static void write(Stream &stream, const tractor::Message &m) {
    std::memcpy(stream.advance(m.size()), m.data(), m.size());
  }
  inline static uint32_t serializedLength(const tractor::Message &m) {
    return m.size();
  }
};
}  // namespace serialization
}  // namespace ros
