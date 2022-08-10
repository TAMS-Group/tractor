// (c) 2020-2022 Philipp Ruppel

#include <tractor/ros/visualize.h>

#include <tractor/ros/publish.h>

#include <visualization_msgs/MarkerArray.h>

namespace tractor {

void visualizePoints(const std::string &name, double scale,
                     const Eigen::Vector4d &color,
                     const std::vector<Eigen::Vector3d> &points) {

  visualization_msgs::Marker marker;

  marker = visualization_msgs::Marker();
  marker.ns = name;
  marker.color.r = color.x();
  marker.color.g = color.y();
  marker.color.b = color.z();
  marker.color.a = color.w();
  marker.scale.x = scale;
  marker.type = visualization_msgs::Marker::POINTS;

  for (auto &p : points) {
    marker.points.emplace_back();
    marker.points.back().x = p.x();
    marker.points.back().y = p.y();
    marker.points.back().z = p.z();
  }

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish("/tractor/visualization", marker_array);
}

} // namespace tractor
