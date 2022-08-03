// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <functional>
#include <mutex>
#include <unordered_map>

namespace tractor {

template <class Key, class Value> class Factory {
  std::mutex _mutex;
  std::function<Value(const Key &)> _factory;
  std::unordered_map<Key, Value> _map;

public:
  Factory(const std::function<Value(const Key &)> &f) : _factory(f) {}
  const Value &operator[](const Key &key) {
    std::lock_guard<std::mutex> lock(_mutex);
    {
      auto it = _map.find(key);
      if (it != _map.end()) {
        return it->second;
      }
    }
    _map[key] = _factory(key);
    return _map[key];
  }
};

} // namespace tractor
