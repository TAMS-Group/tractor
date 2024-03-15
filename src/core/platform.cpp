// 2020-2024 Philipp Ruppel
#include <tractor/core/platform.h>
#include <algorithm>
#include <cmath>
#include <sys/ioctl.h>

namespace tractor {

int getTerminalWidth() {
  struct winsize w;
  bool ok = (ioctl(0, TIOCGWINSZ, &w) == 0);
  if (!ok) {
    return 50;
  } else {
    return std::max(10, std::min(1000, (int)w.ws_col));
  }
}

}  // namespace tractor
