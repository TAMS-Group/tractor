// 2022-2024 Philipp Ruppel

#pragma once

#include "engine.h"
#include "robot.h"

namespace moveit {
namespace core {
class RobotModel;
}
}  // namespace moveit

namespace tractor {

void loadCollisionRobot(const std::shared_ptr<const CollisionEngine> &engine,
                        const moveit::core::RobotModel &moveit_model,
                        CollisionRobot *collision_robot);

static std::shared_ptr<CollisionRobot> loadCollisionRobot(
    const std::shared_ptr<const CollisionEngine> &engine,
    const moveit::core::RobotModel &moveit_model) {
  auto ret = std::make_shared<CollisionRobot>(engine);
  loadCollisionRobot(engine, moveit_model, ret.get());
  return ret;
}

}  // namespace tractor
