// TAMSVIZ
// (c) 2020-2022 Philipp Ruppel

#include <tractor/core/profiler.h>

#include <tractor/core/log.h>
#include <tractor/core/platform.h>

#include <algorithm>
#include <chrono>

#include <ros/ros.h>

namespace tractor {

void ProfilerThread::start() { static ProfilerThread instance; }

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

std::shared_ptr<ProfilerTrack>
Profiler::track(const std::shared_ptr<ProfilerTrack> &track) {
  if (track) {
    // TRACTOR_DEBUG_STREAM("adding profiler track " << track->name() << " "
    //                                               << track->source());
    {
      std::unique_lock<std::mutex> lock(_mutex);
      _tracks.emplace_back(track);
    }
    // TRACTOR_DEBUG_STREAM("profiler track added, total number "
    //                      << _tracks.size());
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

ProfilerThread::ProfilerThread(const std::shared_ptr<Profiler> &profiler) {
  _thread = std::thread([this, profiler]() {
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
      TRACTOR_DEBUG_STREAM("profiler swap");
      auto data = profiler->swap();
      TRACTOR_DEBUG_STREAM("start printing profiler information");
      std::sort(
          data.begin(), data.end(),
          [](const std::pair<std::shared_ptr<ProfilerTrack>, ProfilerData> &a,
             const std::pair<std::shared_ptr<ProfilerTrack>, ProfilerData> &b) {
            return a.second.time < b.second.time;
          });
      std::stringstream stream;
      stream << "profiler\n";
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
          snprintf(buf, sizeof(buf), "%8i - %.3f - %s - %s", i, t,
                   row.first->name().c_str(), source.c_str());
          stream << buf << "\n";
        }
      }
      TRACTOR_INFO_STREAM(stream.str());
      timeout = std::max(timeout + std::chrono::seconds(2),
                         std::chrono::steady_clock::now());
      TRACTOR_DEBUG_STREAM("finished printing profiler information");
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

// ProfilerThread g_profiler_thread;

} // namespace tractor
