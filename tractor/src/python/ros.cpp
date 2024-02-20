// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/ros/interact.h>
#include <tractor/ros/message.h>
#include <tractor/ros/publish.h>
#include <tractor/ros/visualize.h>

#include <ros/ros.h>

#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf2_ros/transform_listener.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/collision_detection/collision_matrix.h>
#include <moveit/planning_scene_monitor/current_state_monitor.h>

namespace tractor {

template <class Scalar>
static void pythonizeROSTyped(py::module main_module, py::module type_module) {
  main_module.def("interact",
                  [](const std::string &frame, const std::string &name,
                     Var<Vector3<Scalar>> &point, double size) {
                    return interact(frame, name, point, Scalar(size));
                  });

  main_module.def("interact",
                  [](const std::string &frame, const std::string &name,
                     Var<Pose<Scalar>> &pose, double size) {
                    return interact(frame, name, pose, Scalar(size));
                  });
}

TRACTOR_PYTHON_TYPED(pythonizeROSTyped);

static void pythonizeROS(py::module m) {
  m.def("visualize_points",
        py::overload_cast<const std::string &, double, const Eigen::Vector4d &,
                          const std::vector<Eigen::Vector3d> &>(
            &visualizePoints));

  m.def("visualize_points",
        py::overload_cast<const std::string &, double,
                          const std::vector<Eigen::Vector4d> &,
                          const std::vector<Eigen::Vector3d> &>(
            &visualizePoints));

  m.def(
      "visualize_lines",
      py::overload_cast<const std::string &, double, const Eigen::Vector4d &,
                        const std::vector<Eigen::Vector3d> &>(&visualizeLines));

  m.def(
      "visualize_lines",
      py::overload_cast<const std::string &, double,
                        const std::vector<Eigen::Vector4d> &,
                        const std::vector<Eigen::Vector3d> &>(&visualizeLines));

  m.def("visualize_text", &visualizeText);

  // m.def(
  //     "visualize_mesh",
  //     py::overload_cast<const std::string &, const Eigen::Vector4d &,
  //                       const std::vector<Eigen::Vector3d>
  //                       &>(&visualizeMesh));

  // m.def(
  //     "visualize_mesh",
  //     py::overload_cast<const std::string &,
  //                       const std::vector<Eigen::Vector4d> &,
  //                       const std::vector<Eigen::Vector3d>
  //                       &>(&visualizeMesh));

  m.def("visualize_mesh",
        [](const std::string &name, const py::array_t<float> &colors,
           const py::array_t<float> &vertices) {
          TRACTOR_ASSERT(colors.ndim() == 1 || colors.ndim() == 2);
          TRACTOR_ASSERT(vertices.ndim() == 2);
          TRACTOR_ASSERT(vertices.shape(1) == 3);

          size_t count = vertices.shape(0);

          auto vertex_data = vertices.unchecked<2>();

          std::vector<Eigen::Vector3d> vertex_vector(count);

          for (size_t i = 0; i < count; i++) {
            vertex_vector[i].x() = vertex_data(i, 0);
            vertex_vector[i].y() = vertex_data(i, 1);
            vertex_vector[i].z() = vertex_data(i, 2);
          }

          if (colors.ndim() == 1) {
            TRACTOR_ASSERT(colors.shape(0) == 4);
            auto color_data = colors.unchecked<1>();
            Eigen::Vector4d color_vector;
            color_vector.x() = color_data(0);
            color_vector.y() = color_data(1);
            color_vector.z() = color_data(2);
            color_vector.w() = color_data(3);
            visualizeMesh(name, color_vector, vertex_vector);
          }

          if (colors.ndim() == 2) {
            TRACTOR_ASSERT(colors.shape(0) == vertices.shape(0));
            TRACTOR_ASSERT(colors.shape(1) == 4);
            auto color_data = colors.unchecked<2>();
            std::vector<Eigen::Vector4d> color_vector(count);
            for (size_t i = 0; i < count; i++) {
              color_vector[i].x() = color_data(i, 0);
              color_vector[i].y() = color_data(i, 1);
              color_vector[i].z() = color_data(i, 2);
              color_vector[i].w() = color_data(i, 3);
            }
            visualizeMesh(name, color_vector, vertex_vector);
          }
        });

  m.def("clear_visualization", &clearVisualization);

  m.def("ros_ok", []() { return ros::ok(); });

  auto init_ros = [](const std::string &name, bool sigint_handler = false) {
    TRACTOR_DEBUG("init_ros " << name);
    auto args =
        py::module::import("sys").attr("argv").cast<std::vector<std::string>>();
    std::vector<char *> argv;
    for (auto &a : args) {
      TRACTOR_DEBUG("arg " << a);
      argv.push_back((char *)a.c_str());
    }
    int argc = args.size();
    int flags = ros::init_options::NoRosout;
    if (!sigint_handler) {
      flags |= ros::init_options::NoSigintHandler;
    }
    ros::init(argc, argv.data(), name, flags);
    static ros::NodeHandle node_handle("~");
    static ros::AsyncSpinner spinner = []() {
      ros::AsyncSpinner spinner(0);
      spinner.start();
      return spinner;
    }();
    clearVisualization();
  };
  m.def("init_ros", init_ros);
  m.def("init_ros", [init_ros](const std::string &name) { init_ros(name); });

  m.def("ros_wait_for_shutdown", []() { ros::waitForShutdown(); });

  m.def("publish", [](const std::string &topic, const py::object &message) {
    auto bytes_io = py::module::import("io").attr("BytesIO")();
    message.attr("serialize")(bytes_io);
    std::string serialized_data =
        bytes_io.attr("getvalue")().cast<std::string>();
    std::string type = message.attr("_type").cast<std::string>();
    std::string hash = message.attr("_md5sum").cast<std::string>();
    std::string definition = message.attr("_full_text").cast<std::string>();
    // TRACTOR_DEBUG("publish " << topic << " " << type << " " << hash << " "
    //                          << definition << " " << serialized_data);
    publish(topic, Message(MessageType::instance(type, hash, definition),
                           serialized_data.data(), serialized_data.size()));
  });

  m.def("advertise", [](const std::string &topic, const py::object &type) {
    std::string name = type.attr("_type").cast<std::string>();
    std::string hash = type.attr("_md5sum").cast<std::string>();
    std::string definition = type.attr("_full_text").cast<std::string>();
    // TRACTOR_DEBUG("advertise " << topic << " type:" << name << " hash:" <<
    // hash
    //                            << " def:" << definition);
    advertise(topic, hash, name, definition);
  });

  // py::class_<moveit::planning_interface::PlanningSceneInterface>(
  //     m, "PlanningSceneInterface")
  //     .def(py::init([](const RobotModel<Geometry> &robot_model) {
  //       auto *ret = new CollisionRobot(engine);
  //       loadCollisionRobot(
  //           engine,
  //           *((const PyRobotModel<Geometry> *)&robot_model)->moveit_model,
  //           ret);
  //       return ret;
  //     }))
  //     .def_property_readonly("links", &CollisionRobot::links)
  //     .def("link", &CollisionRobot::link);

  // struct PyPlanningScene {
  //   planning_scene::PlanningScenePtr planning_scene;
  // };
  // py::class_<PyPlanningScene>(m, "PlanningScene")
  //     .def(py::init(
  //         [](const std::shared_ptr<RobotModel<Geometry>> &robot_model) {
  //   auto moveit_robot =
  //       ((const PyRobotModel<Geometry> *)robot_model.get())->moveit_model;
  //   PyPlanningScene ret;
  //   ret.robot_model = robot_model;
  //   ret.planning_scene =
  //       std::make_shared<planning_scene::PlanningScene>(moveit_robot);
  //   return ret;
  //         }))
  //     .def("check_allowed_collision_matrix",
  //          [](const PyPlanningScene *scene, const std::string &a,
  //             const std::string &b) {
  //   auto allowed = collision_detection::AllowedCollision::NEVER;
  //   scene->planning_scene->getAllowedCollisionMatrix().getAllowedCollision(
  //       a, b, allowed);
  //   return (allowed != collision_detection::AllowedCollision::ALWAYS);
  //          })
  //     // .def("get_current_state",
  //     //      [](const PyPlanningScene *scene, JointState<Geometry>
  //     *out_state)
  //     //      {
  //     // out_state->fromMoveIt(scene->planning_scene->getCurrentState());
  //     //      })
  //     // .def("get_current_state", [](const PyPlanningScene *scene) {
  //     //   JointState<Geometry> ret(scene->robot_model);
  //     //   ret.fromMoveIt(scene->planning_scene->getCurrentState());
  //     //   return ret;
  //     // })
  //     ;
}
TRACTOR_PYTHON_GLOBAL(pythonizeROS);

template <class Geometry>
static void pythonizeROSTwist(py::module main_module, py::module type_module) {
  struct PyCurrentStateMonitor {
    std::shared_ptr<tf2_ros::Buffer> tf_buffer =
        std::make_shared<tf2_ros::Buffer>();
    std::shared_ptr<tf2_ros::TransformListener> tf_listener;
    std::shared_ptr<RobotModel<Geometry>> robot_model;
    planning_scene_monitor::CurrentStateMonitor state_monitor;
    PyCurrentStateMonitor(
        const std::shared_ptr<RobotModel<Geometry>> &robot_model)
        : tf_listener(std::make_shared<tf2_ros::TransformListener>(*tf_buffer)),
          robot_model(robot_model),
          state_monitor(
              ((const PyRobotModel<Geometry> *)robot_model.get())->moveit_model,
              tf_buffer) {
      state_monitor.startStateMonitor("joint_states");
      if (!state_monitor.isActive()) {
        throw std::runtime_error("failed to start state monitor");
      }
    }
  };
  py::class_<PyCurrentStateMonitor, std::shared_ptr<PyCurrentStateMonitor>>(
      type_module, "CurrentStateMonitor")
      .def(py::init<std::shared_ptr<RobotModel<Geometry>>>())
      // .def(py::init(
      //     [](const std::shared_ptr<RobotModel<Geometry>> &robot_model) {
      //       auto moveit_robot =
      //           ((const PyRobotModel<Geometry> *)robot_model.get())
      //               ->moveit_model;
      //       auto ret =
      //  std::make_shared<PyCurrentStateMonitor>(moveit_robot);
      //       ret->startStateMonitor();
      //       return ret;
      //     }))
      // .def("start_state_monitor",
      //      [](PyCurrentStateMonitor *thiz) {
      //        thiz->state_monitor.startStateMonitor("/joint_states");
      //      })
      .def("get_current_state",
           [](PyCurrentStateMonitor *thiz) {
             std::vector<std::string> missing;
             if (!thiz->state_monitor.haveCompleteState(missing)) {
               for (auto &m : missing) {
                 TRACTOR_FATAL("joint state missing " << m);
               }
               throw std::runtime_error("robot state incomplete");
             }
             auto state = thiz->state_monitor.getCurrentState();
             if (!state) {
               throw std::runtime_error("failed to get current state");
             }
             JointState<Geometry> ret(thiz->robot_model);
             ret.fromMoveIt(*state);
             return ret;
           })
      .def("wait_for_complete_state",
           [](PyCurrentStateMonitor *thiz, double timeout) {
             bool ok = thiz->state_monitor.waitForCompleteState(timeout);
             if (!ok) {
               throw std::runtime_error("wait for robot state failed");
             }
           });

  struct PyPlanningScene {
    std::shared_ptr<RobotModel<Geometry>> robot_model;
    planning_scene::PlanningScenePtr planning_scene;
  };
  py::class_<PyPlanningScene>(type_module, "PlanningScene")
      .def(py::init(
          [](const std::shared_ptr<RobotModel<Geometry>> &robot_model) {
            auto moveit_robot =
                ((const PyRobotModel<Geometry> *)robot_model.get())
                    ->moveit_model;
            PyPlanningScene ret;
            ret.robot_model = robot_model;
            ret.planning_scene =
                std::make_shared<planning_scene::PlanningScene>(moveit_robot);
            return ret;
          }))
      .def("check_allowed_collision_matrix", [](const PyPlanningScene *scene,
                                                const std::string &a,
                                                const std::string &b) {
        auto allowed = collision_detection::AllowedCollision::NEVER;
        scene->planning_scene->getAllowedCollisionMatrix().getAllowedCollision(
            a, b, allowed);
        return (allowed != collision_detection::AllowedCollision::ALWAYS);
      });
}
TRACTOR_PYTHON_TWIST(pythonizeROSTwist);

}  // namespace tractor
