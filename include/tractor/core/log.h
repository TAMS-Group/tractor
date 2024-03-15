// 2020-2024 Philipp Ruppel

#pragma once

#include <iostream>
#include <string>

namespace tractor {

enum class LogLevel : int {
  Debug = 6,
  Info = 5,
  Success = 4,
  Warn = 3,
  Error = 2,
  Fatal = 1,
};

void setLogVerbosity(int verbosity);
int getLogVerbosity();
bool checkLogVerbosity(LogLevel verbosity);

template <class First>
void printList(std::ostream &s, const First &first) {
  s << first;
}

template <class First, class... Args>
void printList(std::ostream &s, const First &first, const Args &...args) {
  printList(s, first);
  s << " ";
  printList(s, args...);
}

void logBeginLine(std::ostream &s, LogLevel level);
void logEndLine(std::ostream &s);

}  // namespace tractor

#define TRACTOR_LOG_IMPL(level, ...)                            \
  if (checkLogVerbosity(tractor::LogLevel::level)) {            \
    tractor::logBeginLine(std::cout, tractor::LogLevel::level); \
    std::cout << __VA_ARGS__;                                   \
    tractor::logEndLine(std::cout);                             \
  }
#define TRACTOR_DEBUG(...) TRACTOR_LOG_IMPL(Debug, __VA_ARGS__)
#define TRACTOR_INFO(...) TRACTOR_LOG_IMPL(Info, __VA_ARGS__)
#define TRACTOR_SUCCESS(...) TRACTOR_LOG_IMPL(Success, __VA_ARGS__)
#define TRACTOR_WARN(...) TRACTOR_LOG_IMPL(Warn, __VA_ARGS__)
#define TRACTOR_ERROR(...) TRACTOR_LOG_IMPL(Error, __VA_ARGS__)
#define TRACTOR_FATAL(...) TRACTOR_LOG_IMPL(Fatal, __VA_ARGS__)
