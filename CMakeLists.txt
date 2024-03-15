cmake_minimum_required(VERSION 3.16)
project(tractor)

set(OPT_FLAGS -O3)
#set(OPT_FLAGS -O0)
#set(ASAN_FLAGS -fsanitize=address)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ASAN_FLAGS} -ftemplate-backtrace-limit=0 -Wfatal-errors -ffast-math -g -Wno-register -std=c++17 ${OPT_FLAGS} -mavx -U__AVX__ -fopenmp")

message(STATUS "compiler options: ${CMAKE_CXX_FLAGS}")

find_package(Bullet REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    moveit_core
    moveit_ros_planning
    pybind11_catkin
    eigen_conversions
    interactive_markers
    moveit_ros_robot_interaction
    kdl_parser
)

set(LIBNAME libtractor)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${LIBNAME}
    CATKIN_DEPENDS
        roscpp
        moveit_core
        moveit_ros_planning
        pybind11_catkin
        eigen_conversions
        interactive_markers
        moveit_ros_robot_interaction
        kdl_parser
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${BULLET_INCLUDE_DIRS}
    include
)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${LIBNAME}
    src/collision/base.cpp
    src/collision/bullet.cpp
    src/collision/engine.cpp
    src/collision/link.cpp
    src/collision/loader.cpp
    src/collision/ops.cpp
    src/collision/robot.cpp
    src/collision/shape.cpp
    src/core/allocator.cpp
    src/core/any.cpp
    src/core/array.cpp
    src/core/batch.cpp
    src/core/buffer.cpp
    src/core/constraints.cpp
    src/core/eigen.cpp
    src/core/engine.cpp
    src/core/enum.cpp
    src/core/factory.cpp
    src/core/gradients.cpp
    src/core/lambda.cpp
    src/core/linesearch.cpp
    src/core/list.cpp
    src/core/log.cpp
    src/core/matrix.cpp
    src/core/operator.cpp
    src/core/ops.cpp
    src/core/platform.cpp
    src/core/profiler.cpp
    src/core/program.cpp
    src/core/recorder.cpp
    src/core/simplify.cpp
    src/core/solver.cpp
    src/core/sparsity.cpp
    src/core/tensor.cpp
    src/core/type.cpp
    src/core/var.cpp
    src/core/verify.cpp
    src/dynamics/inertia.cpp
    src/dynamics/rigidbody.cpp
    src/engines/base.cpp
    src/engines/jit.cpp
    src/engines/loop.cpp
    src/engines/parallel.cpp
    src/engines/simple.cpp
    src/geometry/eigen.cpp
    src/geometry/eigen_ops.cpp
    src/geometry/eigenmat.cpp
    src/geometry/eigenquat.cpp
    src/geometry/fast.cpp
    src/geometry/matrix3.cpp
    src/geometry/matrix3_ops.cpp
    src/geometry/ops.cpp
    src/geometry/plane.cpp
    src/geometry/pose.cpp
    src/geometry/pose_ops.cpp
    src/geometry/quaternion.cpp
    src/geometry/quaternion_ops.cpp
    src/geometry/twist.cpp
    src/geometry/twist_ops.cpp
    src/geometry/types.cpp
    src/geometry/vector3.cpp
    src/geometry/vector3_ops.cpp
    src/neural/layer.cpp
    src/neural/network.cpp
    src/neural/ops.cpp
    src/robot/goals.cpp
    src/robot/jointstate.cpp
    src/robot/jointtypes.cpp
    src/robot/linkmodel.cpp
    src/robot/linkstate.cpp
    src/python/collision.cpp
    src/python/common.cpp
    src/python/dynamics.cpp
    src/python/engine.cpp
    src/python/geometry.cpp
    src/python/module.cpp
    src/python/neural.cpp
    src/python/operators.cpp
    src/python/program.cpp
    src/python/robot.cpp
    src/python/ros.cpp
    src/python/solvers.cpp
    src/python/tensor.cpp
    src/robot/robot.cpp
    src/robot/robotinfo.cpp
    src/robot/robotmodel.cpp
    src/robot/robotstate.cpp
    src/robot/trajectory.cpp
    src/ros/interact.cpp
    src/ros/message.cpp
    src/ros/publish.cpp
    src/ros/visualize.cpp
    src/solvers/base.cpp
    src/solvers/ec.cpp
    src/solvers/gd.cpp
    src/solvers/ip.cpp
    src/solvers/pgd.cpp
    src/solvers/ps.cpp
    src/solvers/spsq.cpp
    src/solvers/spsqp.cpp
    src/solvers/sq.cpp
    src/tensor/ops.cpp
    src/tensor/matmul.cpp
)

add_dependencies(
  ${LIBNAME}
  ${${LIBNAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
)

target_link_libraries(
  ${LIBNAME}
  ${catkin_LIBRARIES}
  ${BULLET_LIBRARIES}
  ${ASAN_FLAGS}
  HACD-float64
)

install(DIRECTORY include/tractor DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(TARGETS ${LIBNAME} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

pybind_add_module(tractor MODULE
  src/python/loader.cpp
)
target_link_libraries(tractor PRIVATE libtractor ${catkin_LIBRARIES})
set_target_properties(tractor PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION})
