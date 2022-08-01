project(dexopt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register -g -fopenmp -std=c++17 -O3 -mavx -U__AVX__")

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tractor
  moveit_ros_planning
  tf
  tf_conversions
  interactive_markers
  moveit_ros_robot_interaction
  kdl_parser
  eigen_conversions
  moveit_core
  moveit_ros_planning_interface
  tams_hand_synergies
)

catkin_package(
  CATKIN_DEPENDS
  roscpp
  tractor
  moveit_ros_planning
  tf
  tf_conversions
  interactive_markers
  moveit_ros_robot_interaction
  kdl_parser
  eigen_conversions
  moveit_core
  moveit_ros_planning_interface
  tams_hand_synergies
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(dexopt src/ops.cpp src/dexopt.cpp)
add_dependencies(dexopt ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(dexopt ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${OpenMP_LIBS})
set_target_properties(dexopt PROPERTIES OUTPUT_NAME dexopt PREFIX "")
