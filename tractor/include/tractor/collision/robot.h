// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/collision/link.h>

#include <unordered_map>

namespace moveit {
namespace core {
class RobotModel;
class LinkModel;
class RobotState;
} // namespace core
} // namespace moveit

namespace tractor {

class CollisionRobotBase {
protected:
  void _load(const moveit::core::RobotModel &robot_model,
             bool merge_fixed_links);

public:
  virtual std::shared_ptr<CollisionLinkBase>
  createLink(const std::string &name) = 0;
};

template <class Scalar> class CollisionRobot : public CollisionRobotBase {
  std::vector<std::shared_ptr<const CollisionLink<Scalar>>> _links;
  std::unordered_map<std::string, std::shared_ptr<CollisionLink<Scalar>>>
      _link_map;

public:
  CollisionRobot() {}
  CollisionRobot(const moveit::core::RobotModel &robot_model,
                 bool merge_fixed_links = true) {
    _load(robot_model, merge_fixed_links);
  }
  const auto &links() const { return _links; }
  const std::shared_ptr<CollisionLink<Scalar>> &link(const std::string &name);
  const auto &link(size_t i) { return _links.at(i); }
  virtual std::shared_ptr<CollisionLinkBase>
  createLink(const std::string &name) override;
};

} // namespace tractor
