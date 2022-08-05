// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <iostream>
#include <string>

namespace tractor {

void setLogVerbosity(int verbosity);
int getLogVerbosity();

} // namespace tractor

#define TRACTOR_DEBUG_STREAM(...)                                              \
  if (getLogVerbosity() >= 2) {                                                \
    std::cout << "debug " << __VA_ARGS__ << std::endl;                         \
  }

#define TRACTOR_INFO_STREAM(...)                                               \
  if (getLogVerbosity() >= 1) {                                                \
    std::cout << "info " << __VA_ARGS__ << std::endl;                          \
  }
