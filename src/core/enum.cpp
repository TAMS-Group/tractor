// 2020-2024 Philipp Ruppel

#include <tractor/core/enum.h>

namespace tractor {

std::string enum_name_tolower(const std::string &s) {
  std::string ret;
  for (auto &c : s) {
    ret.push_back(std::tolower(c));
  }
  return ret;
}

bool check_enum_key(const std::string &s) {
  return !s.empty() && std::isalpha(s.front());
}

}  // namespace tractor
