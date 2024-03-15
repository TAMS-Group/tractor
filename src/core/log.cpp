// 2020-2024 Philipp Ruppel

#include <tractor/core/log.h>

#define TRACTOR_LOG_RGB(r, g, b) "\033[38;2;" #r ";" #g ";" #b "m"
#define TRACTOR_LOG_RESET "\e[0m"
#define TRACTOR_LOG_NEWLINE "\n"

namespace tractor {

void logBeginLine(std::ostream &s, LogLevel level) {
  const char *color = TRACTOR_LOG_RGB(255, 255, 255);
  switch (level) {
    case LogLevel::Debug:
      s << "D/ ";
      break;
    case LogLevel::Info:
      s << "I/ ";
      break;
    case LogLevel::Warn:
      s << TRACTOR_LOG_RGB(255, 255, 0) "\e[1m";
      s << "W/ ";
      break;
    case LogLevel::Error:
      s << TRACTOR_LOG_RGB(255, 50, 0) "\e[1m";
      s << "E/ ";
      break;
    case LogLevel::Fatal:
      s << TRACTOR_LOG_RGB(255, 50, 0) "\e[1m";
      s << "F/ ";
      break;
    case LogLevel::Success:
      s << TRACTOR_LOG_RGB(50, 255, 0) "\e[1m";
      s << "S/ ";
      break;
  }
}

void logEndLine(std::ostream &s) {
  s << TRACTOR_LOG_RESET << TRACTOR_LOG_NEWLINE;
}

volatile int &refLogVerbosity() {
  static volatile int g_log_verbosity = []() {
    int v = (int)LogLevel::Info;
    if (auto *s = getenv("TRACTOR_VERBOSITY")) {
      v = std::atoi(s);
    }
    return v;
  }();
  return g_log_verbosity;
}

void setLogVerbosity(int verbosity) { refLogVerbosity() = verbosity; }

int getLogVerbosity() { return refLogVerbosity(); }

bool checkLogVerbosity(LogLevel verbosity) {
  return (int)verbosity <= refLogVerbosity();
}

}  // namespace tractor
