// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>

namespace tractor {}

#define TRACTOR_ASSERT_STR(a) #a

#define TRACTOR_ASSERT(x)                                                  \
  if (!(x)) {                                                              \
    throw std::runtime_error("assertion failure: " TRACTOR_ASSERT_STR(x)); \
  }
