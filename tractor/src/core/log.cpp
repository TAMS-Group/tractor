// (c) 2020-2022 Philipp Ruppel

#include <tractor/core/log.h>

namespace tractor {

volatile int &refLogVerbosity() {
  static volatile int g_log_verbosity = []() {
    int v = 1;
    if (auto *s = getenv("TRACTOR_VERBOSITY")) {
      v = std::atoi(s);
    }
    return v;
  }();
  return g_log_verbosity;
}

void setLogVerbosity(int verbosity) { refLogVerbosity() = verbosity; }

int getLogVerbosity() { return refLogVerbosity(); }

} // namespace tractor
