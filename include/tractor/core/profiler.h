// 2020-2024 Philipp Ruppel

#pragma once

#include "log.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace tractor {

struct ProfilerData {
  uint64_t time = 0;
  uint64_t count = 0;
};

class ProfilerTrack {
  std::string _source;
  std::string _name;
  std::atomic<int64_t> _time;
  std::atomic<int64_t> _count;

 public:
  ProfilerTrack(const std::string &source, const std::string &name);
  ProfilerTrack(const ProfilerTrack &) = delete;
  ProfilerTrack &operator=(const ProfilerTrack &) = delete;
  ~ProfilerTrack();
  const std::string &name() const { return _name; }
  const std::string &source() const { return _source; }
  inline void add(int64_t time) {
    _time += time;
    _count++;
  }
  ProfilerData swap();
};

class ProfilerScope {
  ProfilerTrack &_track;
  std::chrono::steady_clock::time_point _start;

 public:
  inline ProfilerScope(ProfilerTrack &track)
      : _track(track), _start(std::chrono::steady_clock::now()) {}
  inline ~ProfilerScope() {
    uint64_t dt = std::chrono::duration_cast<std::chrono::nanoseconds, int64_t>(
                      std::chrono::steady_clock::now() - _start)
                      .count();
    if (dt > (1000000000 / 5)) {
      TRACTOR_DEBUG("profiler " << dt * (1.0 / 1000000000) << "s "
                                << _track.name() << " " << _track.source());
    }
    _track.add(dt);
  }
  ProfilerScope(const ProfilerScope &) = delete;
  ProfilerScope &operator=(const ProfilerScope &) = delete;
};

class Profiler {
  mutable std::mutex _mutex;
  mutable std::vector<std::weak_ptr<ProfilerTrack>> _tracks;

 public:
  static std::shared_ptr<Profiler> instance();
  std::shared_ptr<ProfilerTrack> track(
      const std::shared_ptr<ProfilerTrack> &track);
  std::vector<std::shared_ptr<ProfilerTrack>> tracks() const;
  std::vector<std::pair<std::shared_ptr<ProfilerTrack>, ProfilerData>> swap();
};

class ProfilerThread {
  std::thread _thread;
  bool _exit = false;
  std::mutex _mutex;
  std::condition_variable _condition;

 public:
  static void start(double interval = 10);
  ProfilerThread(double interval, const std::shared_ptr<Profiler> &profiler =
                                      Profiler::instance());
  ~ProfilerThread();
};

#define TRACTOR_PROFILER(...)                                                 \
  static std::shared_ptr<ProfilerTrack> profiler_track =                      \
      Profiler::instance()->track(                                            \
          std::make_shared<ProfilerTrack>(__PRETTY_FUNCTION__, __VA_ARGS__)); \
  ProfilerScope profiler_scope(*profiler_track);

}  // namespace tractor
