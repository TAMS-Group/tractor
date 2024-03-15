// 2022-2024 Philipp Ruppel

#pragma once

#include "engine.h"
#include "link.h"

namespace tractor {

class CollisionRobot {
  std::shared_ptr<const CollisionEngine> _engine;
  std::vector<std::shared_ptr<const CollisionLink>> _links;
  std::unordered_map<std::string, std::shared_ptr<const CollisionLink>>
      _link_map;

 public:
  CollisionRobot(const CollisionRobot &) = delete;
  CollisionRobot &operator=(const CollisionRobot &) = delete;
  CollisionRobot(const std::shared_ptr<const CollisionEngine> &engine)
      : _engine(engine) {}
  const std::shared_ptr<const CollisionEngine> &engine() const {
    return _engine;
  }
  void addLink(const std::shared_ptr<const CollisionLink> &link) {
    if (_link_map[link->name()]) {
      throw std::runtime_error(
          "collision link with the same name already exists " + link->name());
    }
    _links.push_back(link);
    _link_map[link->name()] = link;
  }
  const std::vector<std::shared_ptr<const CollisionLink>> &links() const {
    return _links;
  }
  const std::shared_ptr<const CollisionLink> &link(
      const std::string &name) const {
    return _link_map.at(name);
  }
};

}  // namespace tractor
