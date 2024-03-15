// 2020-2024 Philipp Ruppel

#pragma once

#include <boost/functional/hash.hpp>
#include <functional>
#include <mutex>
#include <unordered_map>

namespace tractor {

struct Factory {
  template <class... KeyTypes>
  struct Key {
    template <class ValueType>
    class Value {
      struct HashType {
        size_t operator()(const std::tuple<KeyTypes...> &key) const noexcept {
          return boost::hash_value(key);
        }
      };
      std::mutex _mutex;
      std::function<ValueType(const KeyTypes &...)> _factory;
      std::unordered_map<std::tuple<KeyTypes...>, ValueType, HashType> _map;

     public:
      Value(const std::function<ValueType(const KeyTypes &...)> &f)
          : _factory(f) {}
      const ValueType &get(const KeyTypes &...key_data) {
        auto key_tuple = std::make_tuple(key_data...);
        std::lock_guard<std::mutex> lock(_mutex);
        {
          auto it = _map.find(key_tuple);
          if (it != _map.end()) {
            return it->second;
          }
        }
        _map[key_tuple] = _factory(key_data...);
        return _map[key_tuple];
      }
    };
  };
};

}  // namespace tractor
