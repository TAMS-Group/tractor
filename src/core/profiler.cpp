// TAMSVIZ
// 2020-2024 Philipp Ruppel

#include <tractor/core/profiler.h>

#include <tractor/core/log.h>
#include <tractor/core/platform.h>

#include <algorithm>
#include <chrono>

#include <ros/ros.h>

namespace tractor {

void ProfilerThread::start(double interval) {
  static ProfilerThread instance(interval);
}

ProfilerData ProfilerTrack::swap() {
  ProfilerData ret;
  ret.time = _time.exchange(0);
  ret.count = _count.exchange(0);
  return ret;
}

std::shared_ptr<Profiler> Profiler::instance() {
  static std::shared_ptr<Profiler> instance = std::make_shared<Profiler>();
  return instance;
}

std::shared_ptr<ProfilerTrack> Profiler::track(
    const std::shared_ptr<ProfilerTrack> &track) {
  if (track) {
    std::unique_lock<std::mutex> lock(_mutex);
    _tracks.emplace_back(track);
  }
  return track;
}

std::vector<std::shared_ptr<ProfilerTrack>> Profiler::tracks() const {
  std::unique_lock<std::mutex> lock(_mutex);
  std::vector<std::shared_ptr<ProfilerTrack>> ret;
  for (auto it = _tracks.begin(); it != _tracks.end();) {
    if (auto ptr = it->lock()) {
      ret.push_back(ptr);
      it++;
    } else {
      it = _tracks.erase(it);
    }
  }
  return std::move(ret);
}

ProfilerTrack::ProfilerTrack(const std::string &source, const std::string &name)
    : _source(source), _name(name) {
  _time = 0;
  _count = 0;
}

ProfilerTrack::~ProfilerTrack() {}

std::vector<std::pair<std::shared_ptr<ProfilerTrack>, ProfilerData>>
Profiler::swap() {
  std::vector<std::shared_ptr<ProfilerTrack>> tracks = this->tracks();
  std::vector<std::pair<std::shared_ptr<ProfilerTrack>, ProfilerData>> data;
  for (auto &t : tracks) {
    data.emplace_back(t, t->swap());
  }
  return data;
}

ProfilerThread::ProfilerThread(double interval,
                               const std::shared_ptr<Profiler> &profiler) {
  _thread = std::thread([this, profiler, interval]() {
    auto timeout = std::chrono::steady_clock::now();
    while (true) {
      {
        std::unique_lock<std::mutex> lock(_mutex);
        while (true) {
          if (_exit) {
            return;
          }
          if (std::chrono::steady_clock::now() >= timeout) {
            break;
          }
          _condition.wait_until(lock, timeout);
        }
      }
      auto data = profiler->swap();
      std::sort(
          data.begin(), data.end(),
          [](const std::pair<std::shared_ptr<ProfilerTrack>, ProfilerData> &a,
             const std::pair<std::shared_ptr<ProfilerTrack>, ProfilerData> &b) {
            return a.second.time < b.second.time;
          });
      std::stringstream stream;
      stream << "profiler\n";
      stream << "       N  T/N[us]    T[s] - name - source\n";
      for (auto &row : data) {
        if (row.second.count > 0) {
          auto source = row.first->source();
          {
            auto i = source.find(" [with ");
            if (i != std::string::npos) {
              source.resize(i);
            }
          }
          double t = row.second.time * (1.0 / 1000000000.0);
          int i = row.second.count;
          char buf[getTerminalWidth()];
          snprintf(buf, sizeof(buf), "%8i %8i %7.3f - %s - %s", i,
                   (int)std::round(1000000 * t / i), t,
                   row.first->name().c_str(), source.c_str());
          stream << buf << "\n";
        }
      }
      TRACTOR_INFO(stream.str());
      timeout = std::max(
          timeout +
              std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                  std::chrono::duration<double>(interval)),
          std::chrono::steady_clock::now());
    }
  });
}

ProfilerThread::~ProfilerThread() {
  {
    std::unique_lock<std::mutex> lock(_mutex);
    _exit = true;
    _condition.notify_all();
  }
  _thread.join();
}

}  // namespace tractor
