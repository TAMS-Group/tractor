// 2020-2024 Philipp Ruppel

#include <tractor/collision/link.h>

#include <tractor/core/log.h>

namespace tractor {

CollisionLink::CollisionLink() {}

CollisionLink::CollisionLink(const std::string &name) : _name(name) {}

const std::string &CollisionLink::name() const { return _name; }

const std::vector<std::shared_ptr<const CollisionShape>> &
CollisionLink::shapes() const {
  return _shapes;
}

void CollisionLink::addShape(
    const std::shared_ptr<const CollisionShape> &shape) {
  TRACTOR_DEBUG("collision link add shape");
  _shapes.push_back(shape);
}

}  // namespace tractor
