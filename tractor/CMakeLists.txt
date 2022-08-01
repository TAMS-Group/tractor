project(tractor)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -fopenmp -O3 -mavx -U__AVX__")

find_package(Bullet REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    moveit_core
    pybind11_catkin
)

set(LIBNAME libtractor)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES ${LIBNAME}
    CATKIN_DEPENDS
        roscpp
        moveit_core
        pybind11_catkin
)

include_directories(
    ${catkin_INCLUDE_DIRS}
    ${BULLET_INCLUDE_DIRS}
    include
)

#catkin_python_setup()

#set(CMAKE_STATIC_LIBRARY_PREFIX "")
set(CMAKE_SHARED_LIBRARY_PREFIX "")

add_library(${LIBNAME}
    src/collision/link.cpp
    src/collision/query.cpp
    src/collision/ops.cpp
    src/collision/robot.cpp
    src/collision/shape.cpp
    src/collision/types.cpp
    src/core/allocator.cpp
    src/core/any.cpp
    src/core/array.cpp
    src/core/batch.cpp
    src/core/buffer.cpp
    src/core/constraints.cpp
    src/core/eigen.cpp
    src/core/engine.cpp
    src/core/gradients.cpp
    src/core/linesearch.cpp
    src/core/list.cpp
    src/core/matrix.cpp
    src/core/operator.cpp
    src/core/ops.cpp
    src/core/profiler.cpp
    src/core/program.cpp
    src/core/recorder.cpp
    src/core/solver.cpp
    src/core/type.cpp
    src/core/var.cpp
    src/engines/base.cpp
    src/engines/jit.cpp
    src/engines/loop.cpp
    src/engines/parallel.cpp
    src/engines/simple.cpp
    src/geometry/eigen.cpp
    src/geometry/fast.cpp
    src/geometry/ops.cpp
    src/geometry/plane.cpp
    src/geometry/pose.cpp
    src/geometry/quaternion.cpp
    src/geometry/twist.cpp
    src/geometry/types.cpp
    src/geometry/vector3.cpp
    src/neural/layer.cpp
    src/neural/network.cpp
    src/neural/ops.cpp
    src/neural/synapse.cpp
    src/robot/goals.cpp
    src/robot/jointstate.cpp
    src/robot/jointtypes.cpp
    src/robot/linkstate.cpp
    src/robot/robot.cpp
    src/robot/robotinfo.cpp
    src/robot/robotmodel.cpp
    src/robot/robotstate.cpp
    src/robot/trajectory.cpp
    src/solvers/base.cpp
    src/solvers/ec.cpp
    src/solvers/gd.cpp
    src/solvers/ip.cpp
    src/solvers/pgd.cpp
    src/solvers/ps.cpp
    src/solvers/sq.cpp
    src/tensor/newtensor.cpp
    src/tensor/tensor.cpp
    src/tensor/ops.cpp
)
add_dependencies(${LIBNAME} ${${LIBNAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${LIBNAME} ${catkin_LIBRARIES} ${BULLET_LIBRARIES})
install(DIRECTORY include/tractor DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION})
install(TARGETS ${LIBNAME} LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})

pybind_add_module(tractor
  src/python/python.cpp
)
target_link_libraries(tractor PRIVATE libtractor ${catkin_LIBRARIES})
set_target_properties(tractor PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/${CATKIN_GLOBAL_PYTHON_DESTINATION})
