// 2022-2024 Philipp Ruppel

#pragma once

#include "shape.h"

namespace tractor {

class CollisionLink {
  std::string _name;
  std::vector<std::shared_ptr<const CollisionShape>> _shapes;

 public:
  CollisionLink();
  CollisionLink(const std::string &name);
  const std::string &name() const;
  const std::vector<std::shared_ptr<const CollisionShape>> &shapes() const;
  void addShape(const std::shared_ptr<const CollisionShape> &shape);
};

}  // namespace tractor
