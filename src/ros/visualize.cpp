// 2020-2024 Philipp Ruppel

#include <tractor/ros/visualize.h>

#include <tractor/ros/publish.h>

#include <visualization_msgs/MarkerArray.h>

namespace tractor {

static const char *g_visualization_topic = "/tractor/visualization";

void clearVisualization() {
  visualization_msgs::Marker marker;
  marker.action = visualization_msgs::Marker::DELETEALL;

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish(g_visualization_topic, marker_array);
}

void visualizeText(const std::string &name, double scale,
                   const Eigen::Vector4d &color,
                   const Eigen::Vector3d &position, const std::string &text) {
  visualization_msgs::Marker marker;

  marker = visualization_msgs::Marker();
  marker.ns = name;
  marker.color.r = color.x();
  marker.color.g = color.y();
  marker.color.b = color.z();
  marker.color.a = color.w();
  marker.scale.x = scale;
  marker.scale.y = scale;
  marker.scale.z = scale;
  marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  marker.pose.orientation.w = 1;
  marker.pose.position.x = position.x();
  marker.pose.position.y = position.y();
  marker.pose.position.z = position.z();
  marker.text = text;

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish(g_visualization_topic, marker_array);
}

void visualizePoints(const std::string &name, double scale,
                     const std::vector<Eigen::Vector4d> &colors,
                     const std::vector<Eigen::Vector3d> &points) {
  TRACTOR_ASSERT(colors.size() == points.size());

  visualization_msgs::Marker marker;

  marker = visualization_msgs::Marker();
  marker.ns = name;
  marker.color.r = 0;
  marker.color.g = 0;
  marker.color.b = 0;
  marker.color.a = 0;
  marker.scale.x = scale;
  marker.type = visualization_msgs::Marker::POINTS;

  for (auto &c : colors) {
    marker.colors.emplace_back();
    marker.colors.back().r = c.x();
    marker.colors.back().g = c.y();
    marker.colors.back().b = c.z();
    marker.colors.back().a = c.w();
  }

  for (auto &p : points) {
    marker.points.emplace_back();
    marker.points.back().x = p.x();
    marker.points.back().y = p.y();
    marker.points.back().z = p.z();
  }

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish(g_visualization_topic, marker_array);
}

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

  publish(g_visualization_topic, marker_array);
}

void visualizeLines(const std::string &name, double scale,
                    const std::vector<Eigen::Vector4d> &colors,
                    const std::vector<Eigen::Vector3d> &points) {
  TRACTOR_ASSERT(colors.size() == points.size());

  visualization_msgs::Marker marker;

  marker = visualization_msgs::Marker();
  marker.ns = name;
  marker.color.r = 1;
  marker.color.g = 1;
  marker.color.b = 1;
  marker.color.a = 1;
  marker.scale.x = scale;
  marker.type = visualization_msgs::Marker::LINE_LIST;

  for (auto &c : colors) {
    marker.colors.emplace_back();
    marker.colors.back().r = c.x();
    marker.colors.back().g = c.y();
    marker.colors.back().b = c.z();
    marker.colors.back().a = c.w();
  }

  for (auto &p : points) {
    marker.points.emplace_back();
    marker.points.back().x = p.x();
    marker.points.back().y = p.y();
    marker.points.back().z = p.z();
  }

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish(g_visualization_topic, marker_array);
}

void visualizeLines(const std::string &name, double scale,
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
  marker.type = visualization_msgs::Marker::LINE_LIST;

  for (auto &p : points) {
    marker.points.emplace_back();
    marker.points.back().x = p.x();
    marker.points.back().y = p.y();
    marker.points.back().z = p.z();
  }

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish(g_visualization_topic, marker_array);
}

void visualizeMesh(const std::string &name, const Eigen::Vector4d &color,
                   const std::vector<Eigen::Vector3d> &vertices) {
  visualization_msgs::Marker marker;

  marker = visualization_msgs::Marker();
  marker.ns = name;
  marker.color.r = color.x();
  marker.color.g = color.y();
  marker.color.b = color.z();
  marker.color.a = color.w();
  marker.scale.x = 1;
  marker.scale.y = 1;
  marker.scale.z = 1;
  marker.type = visualization_msgs::Marker::TRIANGLE_LIST;

  for (auto &p : vertices) {
    marker.points.emplace_back();
    marker.points.back().x = p.x();
    marker.points.back().y = p.y();
    marker.points.back().z = p.z();
  }

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish(g_visualization_topic, marker_array);
}

void visualizeMesh(const std::string &name,
                   const std::vector<Eigen::Vector4d> &colors,
                   const std::vector<Eigen::Vector3d> &points) {
  TRACTOR_ASSERT(colors.size() == points.size());

  visualization_msgs::Marker marker;

  marker = visualization_msgs::Marker();
  marker.ns = name;
  marker.color.r = 0;
  marker.color.g = 0;
  marker.color.b = 0;
  marker.color.a = 0;
  marker.scale.x = 1;
  marker.scale.y = 1;
  marker.scale.z = 1;
  marker.type = visualization_msgs::Marker::TRIANGLE_LIST;

  marker.colors.resize(colors.size());
  for (size_t i = 0; i < colors.size(); i++) {
    auto &c = colors[i];
    auto &m = marker.colors[i];
    m.r = c.x();
    m.g = c.y();
    m.b = c.z();
    m.a = c.w();
  }

  marker.points.resize(points.size());
  for (size_t i = 0; i < points.size(); i++) {
    auto &p = points[i];
    auto &m = marker.points[i];
    m.x = p.x();
    m.y = p.y();
    m.z = p.z();
  }

  visualization_msgs::MarkerArray marker_array;
  marker_array.markers.push_back(marker);

  publish(g_visualization_topic, marker_array);
}

}  // namespace tractor
