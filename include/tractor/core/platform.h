// 2020-2024 Philipp Ruppel

#pragma once

namespace tractor {

int getTerminalWidth();

#define TRACTOR_FAST __attribute__((nothrow, hot, optimize("-O3")))

#define TRACTOR_SLOW __attribute__((nothrow, cold, optimize("-O0")))

}  // namespace tractor
