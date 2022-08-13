// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/link.h>

#include <tractor/core/log.h>

namespace tractor {

CollisionLink::CollisionLink() {}

CollisionLink::CollisionLink(const std::string &name) : _name(name) {}

// CollisionLink::CollisionLink(
//     const std::string &name,
//     const std::vector<std::shared_ptr<const CollisionShape>> &shapes)
//     : _name(name), _shapes(shapes) {}

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

} // namespace tractor
