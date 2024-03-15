// 2020-2024 Philipp Ruppel

#include <tractor/ros/interact.h>

#include <tractor/core/factory.h>
#include <tractor/core/var.h>
#include <tractor/geometry/eigen.h>

#include <eigen_conversions/eigen_msg.h>
#include <interactive_markers/interactive_marker_server.h>
#include <moveit/robot_interaction/interactive_marker_helpers.h>

namespace tractor {

class InteractivePoseMarker {
  struct Data {
    mutable std::mutex _mutex;
    Eigen::Affine3d _pose = Eigen::Affine3d::Identity();
    bool _flag = true;
  };
  std::shared_ptr<Data> _data = std::make_shared<Data>();
  std::string _name;
  Eigen::Affine3d _start_pose = Eigen::Affine3d::Identity();

 public:
  InteractivePoseMarker(const InteractivePoseMarker &) = delete;
  InteractivePoseMarker &operator=(const InteractivePoseMarker &) = delete;

  InteractivePoseMarker(
      interactive_markers::InteractiveMarkerServer &interactive_marker_server,
      const std::string &root, const Eigen::Affine3d &pose,
      const std::string &name, double scale = 0.2)
      : _name(name) {
    TRACTOR_DEBUG("creating interactive pose marker " << name);

    _data->_pose = pose;
    _start_pose = pose;

    visualization_msgs::InteractiveMarker interactive_marker;
    interactive_marker.header.frame_id = root;
    interactive_marker.header.stamp = ros::Time::now();
    interactive_marker.name = name;
    interactive_marker.scale = scale;

    robot_interaction::add6DOFControl(interactive_marker);

    tf::poseEigenToMsg(pose, interactive_marker.pose);

    std_msgs::ColorRGBA marker_color;
    marker_color.r = 1;
    marker_color.g = 1;
    marker_color.b = 0;
    marker_color.a = 1;
    robot_interaction::addViewPlaneControl(interactive_marker, 0.33 * scale,
                                           marker_color, true, false);

    auto data = _data;
    interactive_marker_server.insert(
        interactive_marker,
        [data, name](const visualization_msgs::InteractiveMarkerFeedbackConstPtr
                         &feedback) {
          if (feedback->marker_name == name &&
              feedback->event_type ==
                  visualization_msgs::InteractiveMarkerFeedback::POSE_UPDATE) {
            Eigen::Affine3d pose;
            tf::poseMsgToEigen(feedback->pose, pose);
            std::lock_guard<std::mutex> lock(data->_mutex);
            if (!data->_pose.isApprox(pose)) {
              data->_pose = pose;
              data->_flag = true;
            }
          }
        });

    interactive_marker_server.applyChanges();
  }

  bool poll() const {
    std::lock_guard<std::mutex> lock(_data->_mutex);
    bool ret = _data->_flag;
    _data->_flag = false;
    return ret;
  }

  Eigen::Affine3d pose() const {
    std::lock_guard<std::mutex> lock(_data->_mutex);
    return _data->_pose;
  }

  const std::string &name() const { return _name; }

  const Eigen::Affine3d &initialPose() const { return _start_pose; }
};

class InteractivePositionMarker {
  struct Data {
    mutable std::mutex _mutex;
    Eigen::Vector3d _position = Eigen::Vector3d::Zero();
    bool _flag = true;
  };
  std::shared_ptr<Data> _data = std::make_shared<Data>();
  std::string _name;
  Eigen::Vector3d _initial_position = Eigen::Vector3d::Zero();

 public:
  InteractivePositionMarker(const InteractivePositionMarker &) = delete;
  InteractivePositionMarker &operator=(const InteractivePositionMarker &) =
      delete;
  InteractivePositionMarker(
      interactive_markers::InteractiveMarkerServer &interactive_marker_server,
      const std::string &root, const Eigen::Vector3d &position,
      const std::string &name, double scale)
      : _name(name) {
    TRACTOR_DEBUG("creating interactive position marker " << name);

    _data->_position = position;
    _initial_position = position;

    visualization_msgs::InteractiveMarker interactive_marker;
    interactive_marker.header.frame_id = root;
    interactive_marker.header.stamp = ros::Time::now();
    interactive_marker.name = name;
    interactive_marker.scale = scale;

    robot_interaction::addPositionControl(interactive_marker);

    tf::pointEigenToMsg(position, interactive_marker.pose.position);
    interactive_marker.pose.orientation.w = 1;

    std_msgs::ColorRGBA marker_color;
    marker_color.r = 1;
    marker_color.g = 1;
    marker_color.b = 0;
    marker_color.a = 1;
    robot_interaction::addViewPlaneControl(interactive_marker, 0.33 * scale,
                                           marker_color, true, false);

    auto data = _data;
    interactive_marker_server.insert(
        interactive_marker,
        [data, name](const visualization_msgs::InteractiveMarkerFeedbackConstPtr
                         &feedback) {
          TRACTOR_DEBUG("marker callback");
          if (feedback->marker_name == name &&
              feedback->event_type ==
                  visualization_msgs::InteractiveMarkerFeedback::POSE_UPDATE) {
            TRACTOR_DEBUG("point update " << feedback->marker_name);
            std::lock_guard<std::mutex> lock(data->_mutex);
            Eigen::Vector3d pos = Eigen::Vector3d::Zero();
            tf::pointMsgToEigen(feedback->pose.position, pos);
            if (!data->_position.isApprox(pos)) {
              data->_position = pos;
              data->_flag = true;
            }
          }
        });

    interactive_marker_server.applyChanges();
  }

  bool poll() const {
    std::lock_guard<std::mutex> lock(_data->_mutex);
    bool ret = _data->_flag;
    _data->_flag = false;
    return ret;
  }

  Eigen::Vector3d position() const {
    std::lock_guard<std::mutex> lock(_data->_mutex);
    return _data->_position;
  }

  const std::string &name() const { return _name; }

  const Eigen::Vector3d &initialPosition() const { return _initial_position; }
};

const std::shared_ptr<interactive_markers::InteractiveMarkerServer> &
markerServerInstance() {
  static auto instance = []() {
    TRACTOR_DEBUG("creating interactive marker server");
    return std::make_shared<interactive_markers::InteractiveMarkerServer>(
        "/interactive_markers", "", true);
  }();
  return instance;
}

template <class T>
bool interact(const std::string &frame, const std::string &name, Pose<T> &pose,
              const T &size) {
  static Factory::Key<std::string>::Value<
      std::shared_ptr<InteractivePoseMarker>>
      factory([&](const std::string &name) {
        return std::make_shared<InteractivePoseMarker>(
            *markerServerInstance(), frame, toEigenIsometry3d(value(pose)),
            name, size);
      });
  auto marker = factory.get(name);
  bool changed = marker->poll();
  if (changed) {
    pose = convertEigenToPose<GeometryFast<T>>(marker->pose());
  }
  return changed;
}

template bool interact(const std::string &frame, const std::string &name,
                       Pose<double> &position, const double &size);

template bool interact(const std::string &frame, const std::string &name,
                       Pose<float> &position, const float &size);

template <class T>
bool interact(const std::string &frame, const std::string &name,
              Vector3<T> &position, const T &size) {
  static Factory::Key<std::string>::Value<
      std::shared_ptr<InteractivePositionMarker>>
      factory([&](const std::string &name) {
        return std::make_shared<InteractivePositionMarker>(
            *markerServerInstance(), frame,
            Eigen::Vector3d(position.x(), position.y(), position.z()), name,
            size);
      });
  auto marker = factory.get(name);
  bool changed = marker->poll();
  if (changed) {
    auto p = marker->position();
    position.x() = p.x();
    position.y() = p.y();
    position.z() = p.z();
  }
  return changed;
}

template bool interact(const std::string &frame, const std::string &name,
                       Vector3<double> &position, const double &size);

template bool interact(const std::string &frame, const std::string &name,
                       Vector3<float> &position, const float &size);

}  // namespace tractor
