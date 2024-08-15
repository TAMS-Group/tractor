// (c) 2022 Philipp Ruppel

#pragma once

#include "engine.h"
#include "robot.h"

namespace moveit {
namespace core {
class RobotModel;
}
}  // namespace moveit

namespace tractor {

struct CollisionLoaderOptions {
  bool use_primitives = true;
  bool use_convex_decomposition = true;
  bool merge_fixed_links = true;
};

void loadCollisionRobot(const std::shared_ptr<const CollisionEngine> &engine,
                        const moveit::core::RobotModel &moveit_model,
                        CollisionRobot *collision_robot,
                        const CollisionLoaderOptions &options);

static std::shared_ptr<CollisionRobot> loadCollisionRobot(
    const std::shared_ptr<const CollisionEngine> &engine,
    const moveit::core::RobotModel &moveit_model,
    const CollisionLoaderOptions &options = CollisionLoaderOptions()) {
  auto ret = std::make_shared<CollisionRobot>(engine);
  loadCollisionRobot(engine, moveit_model, ret.get(), options);
  return ret;
}

}  // namespace tractor
