// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/factory.h>
#include <tractor/geometry/fast.h>
#include <tractor/robot/robot.h>
#include <tractor/ros/visualize.h>

#include <moveit/robot_model_loader/robot_model_loader.h>

namespace tractor {

template <class Scalar>
static void pythonizeRobot(py::module &main_module, py::module &type_module) {

  typedef GeometryFast<Var<Scalar>> Geometry;

  struct PyRobotModel {
    std::shared_ptr<const RobotModel<Geometry>> tractor_model;
    moveit::core::RobotModelConstPtr moveit_model;
    PyRobotModel(const std::string &robot_description) {
      static Factory::Key<std::string>::Value<moveit::core::RobotModelConstPtr>
          robot_model_factory([](const std::string &robot_description) {
            TRACTOR_DEBUG("loading robot model " << robot_description);
            robot_model_loader::RobotModelLoader loader(robot_description,
                                                        false);
            moveit::core::RobotModelConstPtr ret = loader.getModel();
            if (!ret) {
              throw std::runtime_error("failed to load robot model: " +
                                       robot_description);
            }
            return ret;
          });
      moveit_model = robot_model_factory.get(robot_description);
      tractor_model = std::make_shared<RobotModel<Geometry>>(*moveit_model);
    }
    PyRobotModel() : PyRobotModel("/robot_description") {}
  };

  // strict PyRobotState {
  //   std::shared_ptr<const RobotModel<Geometry>> model;
  //   RobotState<Geometry> state;
  //   PyRobotState(const PyRobotModel &model) : model(model), state(*model) {}
  // };

  struct PyJointStates {
    std::shared_ptr<const RobotModel<Geometry>> model;
    JointState<Geometry> state;
    PyJointStates(const std::shared_ptr<const PyRobotModel> &model)
        : model(model), state(*model) {}
  };

  struct PyLinkStates {
    std::shared_ptr<const RobotModel<Geometry>> model;
    LinkState<Geometry> state;
    PyLinkStates(const std::shared_ptr<const PyRobotModel> &model)
        : model(model), state(*model) {}
  };

  main_module.def("visualize", [](const std::string &topic,
                                  const RobotModel<Geometry> &robot_model,
                                  const JointState<Geometry> &joint_state) {
    visualize(topic, robot_model, joint_state);
  });

  py::class_<PyLinkStates>(type_module, "LinkStates")
      .def(py::init<const PyRobotModel &>())
      .def("pose",
           [](PyLinkStates &_this, const std::string &name) {
             return _this.state.pose(name);
           })
      .def("pose", [](PyLinkStates &_this, size_t index) {
        return _this.state.pose(index);
      });

  py::class_<PyJointStates>(type_module, "JointStates")
      .def(py::init<const PyRobotModel &>())
      .def("serialize",
           [](PyJointStates &_this) {
             AlignedStdVector<Var<Scalar>> positions;
             _this.state.serializePositions(positions);
             return positions;
           })
      .def("deserialize", [](PyJointStates &_this,
                             const AlignedStdVector<Var<Scalar>> &positions) {
        _this.state.deserializePositions(positions);
      });

  // main_module.def("variable", [](JointState<Geometry> &joint_states) {
  //   for (auto &joint : joint_states) {
  //     if (auto *rec = Recorder::instance()) {
  //       rec->reference(joint.pointer());
  //     }
  //     joint->makeVariables();
  //   }
  // });

  // py::class_<PyRobotState>(type_module, "RobotState")
  //     .def(py::init<const PyRobotModel &>())
  //     .def_property_readonly(
  //         "joints", [](PyRobotState &_this) { return &_this.joints(); },
  //         py::return_value_policy::reference_internal)
  //     .def_property_readonly(
  //         "links", [](PyRobotState &_this) { return &_this.links(); },
  //         py::return_value_policy::reference_internal);

  py::class_<PyRobotModel>(type_module, "RobotModel")
      .def(py::init<>())
      .def(py::init<std::string>())
      .def("compute_kinematics",
           [](const PyRobotModel &py_model,
              const JointState<Geometry> &joint_state,
              LinkState<Geometry> &link_state) {
             py_model.tractor_model->computeFK(joint_state, link_state);
           })
      .def("compute_kinematics",
           [](const PyRobotModel &py_model, RobotState<Geometry> &robot_state) {
             py_model.tractor_model->computeFK(robot_state.joints(),
                                               robot_state.links());
           })
      .def("variable_name",
           [](const PyRobotModel &py_model, size_t i) {
             return py_model.tractor_model->info()->variables().name(i);
           })
      .def("variable_index",
           [](const PyRobotModel &py_model, const std::string &name) {
             return py_model.tractor_model->info()->variables().index(name);
           })
      .def_property_readonly(
          "variable_names",
          [](const PyRobotModel &py_model) {
            return py_model.tractor_model->info()->variables().names();
          })
      .def_property_readonly(
          "variable_count",
          [](const PyRobotModel &py_model) {
            return py_model.tractor_model->info()->variables().size();
          })
      .def("link_name",
           [](const PyRobotModel &py_model, size_t i) {
             return py_model.tractor_model->info()->links().name(i);
           })
      .def("link_index",
           [](const PyRobotModel &py_model, const std::string &name) {
             return py_model.tractor_model->info()->links().index(name);
           })
      .def_property_readonly(
          "link_names",
          [](const PyRobotModel &py_model) {
            return py_model.tractor_model->info()->links().names();
          })
      .def_property_readonly(
          "link_count",
          [](const PyRobotModel &py_model) {
            return py_model.tractor_model->info()->links().size();
          })
      .def("joint_name",
           [](const PyRobotModel &py_model, size_t i) {
             return py_model.tractor_model->info()->joints().name(i);
           })
      .def("joint_index",
           [](const PyRobotModel &py_model, const std::string &name) {
             return py_model.tractor_model->info()->joints().index(name);
           })
      .def_property_readonly(
          "joint_names",
          [](const PyRobotModel &py_model) {
            return py_model.tractor_model->info()->joints().names();
          })
      .def_property_readonly("joint_count", [](const PyRobotModel &py_model) {
        return py_model.tractor_model->info()->joints().size();
      });
}

TRACTOR_PYTHON_TYPED(pythonizeRobot);

} // namespace tractor
