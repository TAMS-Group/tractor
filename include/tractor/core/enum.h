// 2020-2024 Philipp Ruppel

#pragma once

#include <random>
#include <string>
#include <unordered_map>

namespace tractor {
std::string enum_name_tolower(const std::string &s);
bool check_enum_key(const std::string &s);
}  // namespace tractor

#define TRACTOR_ENUM_KEYS(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, \
                          r, s, t, u, v, w, x, y, z...)                      \
  {                                                                          \
    #a, #b, #c, #d, #e, #f, #g, #h, #i, #j, #k, #l, #m, #n, #o, #p, #q, #r,  \
        #s, #t, #u, #v, #w, #x, #y, #z                                       \
  }

#define TRACTOR_ENUM(Name, ...)                                            \
  enum class Name { __VA_ARGS__ };                                         \
  static std::vector<std::pair<Name, std::string>> enumerateEach##Name() { \
    static std::vector<std::pair<Name, std::string>> ret = []() {          \
      std::vector<std::string> names = TRACTOR_ENUM_KEYS(                  \
          __VA_ARGS__, , , , , , , , , , , , , , , , , , , , , , , , , );  \
      std::vector<std::pair<Name, std::string>> ret;                       \
      for (size_t i = 0; i < names.size(); i++) {                          \
        if (tractor::check_enum_key(names[i])) {                           \
          ret.emplace_back((Name)i, names[i]);                             \
          ret.emplace_back((Name)i, tractor::enum_name_tolower(names[i])); \
        }                                                                  \
      }                                                                    \
      return ret;                                                          \
    }();                                                                   \
    return ret;                                                            \
  }                                                                        \
  static Name parse##Name(const std::string &str) {                        \
    auto all = enumerateEach##Name();                                      \
    for (auto &p : all) {                                                  \
      if (p.second == str) {                                               \
        return p.first;                                                    \
      }                                                                    \
    }                                                                      \
    throw std::runtime_error("enum value not found");                      \
  }
