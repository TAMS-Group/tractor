// (c) 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/ros/visualize.h>

#include <ros/ros.h>

#include <thread>
#include <mutex>
#include <condition_variable>
#include <chrono>

#include <pybind11/embed.h>

#include <visualization_msgs/MarkerArray.h>

namespace tractor {

static void pythonizeUtils(py::module m) {
  struct VizMarker {
    std::string ns;
    int id = 0;
    int type = 0;
    int action = 0;
    Eigen::Vector4f color = Eigen::Vector4f::Zero();
    Eigen::MatrixXd points;
    Eigen::MatrixXf colors;
    Eigen::Vector3d scale = Eigen::Vector3d::Zero();
  };

  typedef std::vector<std::vector<VizMarker>> VizData;

  struct VizLooper {
    std::mutex mutex;
    std::condition_variable condition;
    bool has_new_data = false;
    VizData new_data;
    bool ok = true;
    std::thread thread;
    VizLooper(double time_step, bool sync) {
      thread = std::thread([this, time_step, sync]() {
        VizData data;
        std::chrono::steady_clock::time_point tlast =
            std::chrono::steady_clock::now();
        size_t index = 0;
        auto tstep =
            std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(time_step));
        while (true) {
          bool timeout = false;

          {
            std::unique_lock<std::mutex> lock(mutex);
            while (true) {
              if (!ok) return;
              auto tnext = tlast + tstep;
              timeout = (std::chrono::steady_clock::now() >= tnext);
              if (timeout) {
                tlast = tlast + tstep;
              }
              if (sync) {
                if (timeout) {
                  if (index >= data.size()) {
                    index = 0;
                    data = new_data;
                    has_new_data = false;
                  }
                  break;
                }
              } else {
                if (has_new_data || timeout) {
                  if (has_new_data) {
                    has_new_data = false;
                    data = new_data;
                  }
                  index++;
                  if (index >= data.size()) {
                    index = 0;
                  }
                  break;
                }
              }
              condition.wait_until(lock, tnext);
            }
          }

          TRACTOR_DEBUG("update viz loop");

          visualization_msgs::MarkerArray marker_array;

          if (data.size()) {
            for (auto& dat : data.at(index)) {
              visualization_msgs::Marker marker;

              marker = visualization_msgs::Marker();
              marker.ns = dat.ns;
              marker.id = dat.id;
              marker.type = dat.type;
              marker.action = dat.action;
              marker.color.r = dat.color.x();
              marker.color.g = dat.color.y();
              marker.color.b = dat.color.z();
              marker.color.a = dat.color.w();
              marker.scale.x = dat.scale.x();
              marker.scale.y = dat.scale.y();
              marker.scale.z = dat.scale.z();

              if (dat.colors.rows() > 0) {
                TRACTOR_ASSERT(dat.colors.cols() == 4);
                for (size_t row = 0; row < dat.colors.rows(); row++) {
                  marker.colors.emplace_back();
                  marker.colors.back().r = dat.colors(row, 0);
                  marker.colors.back().g = dat.colors(row, 1);
                  marker.colors.back().b = dat.colors(row, 2);
                  marker.colors.back().a = dat.colors(row, 3);
                }
              }

              if (dat.points.rows() > 0) {
                TRACTOR_ASSERT(dat.points.cols() == 3);
                for (size_t row = 0; row < dat.points.rows(); row++) {
                  marker.points.emplace_back();
                  marker.points.back().x = dat.points(row, 0);
                  marker.points.back().y = dat.points(row, 1);
                  marker.points.back().z = dat.points(row, 2);
                }
              }

              marker_array.markers.push_back(marker);
            }
          }

          tractor::publish("/tractor/visualization", marker_array);

          if (timeout) {
            index++;
          }
        }
      });
    }
    ~VizLooper() {
      TRACTOR_DEBUG("stop viz looper");
      {
        std::unique_lock<std::mutex> lock(mutex);
        ok = false;
        condition.notify_all();
      }
      thread.join();
      TRACTOR_DEBUG("viz looper stopped");
    }
    void update(const std::vector<std::vector<VizMarker>>& data) {
      TRACTOR_DEBUG("update viz looper");
      {
        std::unique_lock<std::mutex> lock(mutex);
        new_data = data;
        has_new_data = true;
        condition.notify_all();
      }
    }
  };

  py::class_<VizMarker>(m, "VizMarker")
      .def(py::init<>())
      .def_readwrite("action", &VizMarker::action)
      .def_readwrite("ns", &VizMarker::ns)
      .def_readwrite("scale", &VizMarker::scale)
      .def_readwrite("id", &VizMarker::id)
      .def_readwrite("type", &VizMarker::type)
      .def_readwrite("color", &VizMarker::color)
      .def_readwrite("colors", &VizMarker::colors)
      .def_readwrite("points", &VizMarker::points);

  py::class_<VizLooper>(m, "VizLooper")
      .def(py::init<double, bool>())
      .def("update", &VizLooper::update);

  //   struct RateThread {
  //     std::thread thread;
  //     std::mutex mutex;
  //     std::condition_variable condition;
  //     bool ok = true;
  //     RateThread(double rate, const std::function<void()>& fun) {
  //       thread = std::thread([this, rate, fun]() {
  //         while (ok && ros::ok()) {
  //           {
  //             std::unique_lock<std::mutex> lock(mutex);
  //             if (!ok) {
  //               break;
  //             }
  //           }
  //           fun();
  //           ros::Duration(1.0 / rate).sleep();
  //         }
  //       });
  //     }
  //     ~RateThread() {
  //       {
  //         std::unique_lock<std::mutex> lock(mutex);
  //         ok = false;
  //         condition.notify_all();
  //       }
  //       thread.join();
  //     }
  //   };

  //   struct RateThread {
  //     std::thread thread;
  //     std::mutex mutex;
  //     volatile bool ok = true;
  //     RateThread(double rate, const std::function<void()>& fun) {
  //       thread = std::thread([this, rate, fun]() {
  //         // py::scoped_interpreter guard{};
  //         TRACTOR_ASSERT(fun);
  //         TRACTOR_ASSERT(rate > 0);
  //         TRACTOR_INFO("entering rate thread");
  //         while (ros::ok()) {
  //           {
  //             // py::gil_scoped_release release;
  //             std::unique_lock<std::mutex>(mutex);
  //             if (!ok) {
  //               break;
  //             }
  //           }

  //           TRACTOR_INFO("acquire gil");
  //           {
  //             // py::gil_scoped_acquire acquire;
  //             TRACTOR_INFO("start fun");
  //             fun();
  //             TRACTOR_INFO("fun ready");
  //           }
  //           TRACTOR_INFO("gil released");

  //           {
  //             // py::gil_scoped_release release;
  //             ros::Duration(1.0 / rate).sleep();
  //           }
  //         }
  //         TRACTOR_INFO("exiting rate thread");
  //         // py::finalize_interpreter();
  //       });
  //     }
  //     ~RateThread() {
  //       TRACTOR_INFO("stopping rate thread");
  //       py::gil_scoped_release release;
  //       {
  //         std::unique_lock<std::mutex>(mutex);
  //         ok = false;
  //       }
  //       TRACTOR_INFO("joining rate thread");
  //       thread.join();
  //       TRACTOR_INFO("rate thread stopped");
  //     }
  //   };

  //   py::class_<RateThread>(m, "RateThread")
  //       .def(py::init<double, std::function<void()>>());
}

TRACTOR_PYTHON_GLOBAL(pythonizeUtils);

}  // namespace tractor
