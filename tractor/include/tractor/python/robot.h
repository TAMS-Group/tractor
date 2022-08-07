// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/geometry/fast.h>
#include <tractor/robot/robot.h>

#include <moveit/robot_model_loader/robot_model_loader.h>

namespace tractor {

template <class Scalar>
static void pythonizeRobot(py::module &main_module, py::module &type_module) {

  typedef GeometryFast<Var<Scalar>> Geometry;

  // static auto getParam = [](const std::string &name) {
  //   return py::module::import("rospy")
  //       .attr("get_param")(name)
  //       .cast<std::string>();
  // };

  static Factory::Key<std::string>::Value<moveit::core::RobotModelConstPtr>
      robot_model_factory([](const std::string &robot_description) {
        TRACTOR_DEBUG("loading robot model " << robot_description);
        // robot_model_loader::RobotModelLoader::Options options;
        // options.load_kinematics_solvers_ = false;
        // options.urdf_string_ = getParam(robot_description);
        // options.srdf_string_ = getParam(robot_description + "_semantic");
        // robot_model_loader::RobotModelLoader loader(options);
        robot_model_loader::RobotModelLoader loader(robot_description);
        moveit::core::RobotModelConstPtr ret = loader.getModel();
        if (!ret) {
          throw std::runtime_error("failed to load robot model: " +
                                   robot_description);
        }
        return ret;
      });

  py::class_<LinkState<Geometry>>(type_module, "LinkStates")
      .def(py::init<const RobotModel<Geometry> &>())
      .def("pose", [](LinkState<Geometry> &_this,
                      const std::string &name) { return _this.pose(name); })
      .def("pose", [](LinkState<Geometry> &_this, size_t index) {
        return _this.pose(index);
      });

  py::class_<JointState<Geometry>>(type_module, "JointStates")
      .def(py::init<const RobotModel<Geometry> &>())
      .def("serialize",
           [](JointState<Geometry> &_this) {
             AlignedStdVector<Var<Scalar>> positions;
             _this.serializePositions(positions);
             return positions;
           })
      .def("deserialize", [](JointState<Geometry> &_this,
                             const AlignedStdVector<Var<Scalar>> &positions) {
        _this.deserializePositions(positions);
      });

  // main_module.def("variable", [](JointState<Geometry> &joint_states) {
  //   for (auto &joint : joint_states) {
  //     if (auto *rec = Recorder::instance()) {
  //       rec->reference(joint.pointer());
  //     }
  //     joint->makeVariables();
  //   }
  // });

  py::class_<RobotState<Geometry>>(type_module, "RobotState")
      .def(py::init<const RobotModel<Geometry> &>())
      .def_property_readonly(
          "joints", [](RobotState<Geometry> &_this) { return &_this.joints(); },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "links", [](RobotState<Geometry> &_this) { return &_this.links(); },
          py::return_value_policy::reference_internal);

  py::class_<RobotModel<Geometry>, std::shared_ptr<RobotModel<Geometry>>>(
      type_module, "RobotModel")
      .def(py::init([](const std::string &robot_description) {
        return std::make_shared<RobotModel<Geometry>>(
            *robot_model_factory.get(robot_description));
      }))
      .def(py::init([]() {
        return std::make_shared<RobotModel<Geometry>>(
            *robot_model_factory.get("/robot_description"));
      }))
      .def("compute_kinematics",
           [](const RobotModel<Geometry> &robot_model,
              const JointState<Geometry> &joint_state,
              LinkState<Geometry> &link_state) {
             robot_model.computeFK(joint_state, link_state);
           })
      .def("compute_kinematics",
           [](const RobotModel<Geometry> &robot_model,
              RobotState<Geometry> &robot_state) {
             robot_model.computeFK(robot_state.joints(), robot_state.links());
           })
      .def("variable_name",
           [](const RobotModel<Geometry> &robot_model, size_t i) {
             return robot_model.info()->variables().name(i);
           })
      .def(
          "variable_index",
          [](const RobotModel<Geometry> &robot_model, const std::string &name) {
            return robot_model.info()->variables().index(name);
          })
      .def_property_readonly("variable_names",
                             [](const RobotModel<Geometry> &robot_model) {
                               return robot_model.info()->variables().names();
                             })
      .def_property_readonly("variable_count",
                             [](const RobotModel<Geometry> &robot_model) {
                               return robot_model.info()->variables().size();
                             })
      .def("link_name",
           [](const RobotModel<Geometry> &robot_model, size_t i) {
             return robot_model.info()->links().name(i);
           })
      .def(
          "link_index",
          [](const RobotModel<Geometry> &robot_model, const std::string &name) {
            return robot_model.info()->links().index(name);
          })
      .def_property_readonly("link_names",
                             [](const RobotModel<Geometry> &robot_model) {
                               return robot_model.info()->links().names();
                             })
      .def_property_readonly("link_count",
                             [](const RobotModel<Geometry> &robot_model) {
                               return robot_model.info()->links().size();
                             })
      .def("joint_name",
           [](const RobotModel<Geometry> &robot_model, size_t i) {
             return robot_model.info()->joints().name(i);
           })
      .def(
          "joint_index",
          [](const RobotModel<Geometry> &robot_model, const std::string &name) {
            return robot_model.info()->joints().index(name);
          })
      .def_property_readonly("joint_names",
                             [](const RobotModel<Geometry> &robot_model) {
                               return robot_model.info()->joints().names();
                             })
      .def_property_readonly("joint_count",
                             [](const RobotModel<Geometry> &robot_model) {
                               return robot_model.info()->joints().size();
                             })
      .def_property_readonly("variable_count",
                             &RobotModel<Geometry>::variableCount);
}

} // namespace tractor
