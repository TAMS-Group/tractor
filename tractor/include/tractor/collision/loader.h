// (c) 2022 Philipp Ruppel

#pragma once

#include "engine.h"
#include "robot.h"

namespace moveit {
namespace core {
class RobotModel;
}
} // namespace moveit

namespace tractor {

void loadCollisionRobot(const std::shared_ptr<const CollisionEngine> &engine,
                        const moveit::core::RobotModel &moveit_model,
                        CollisionRobot *collision_robot);

}
