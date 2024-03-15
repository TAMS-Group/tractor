// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/factory.h>
#include <tractor/geometry/fast.h>
#include <tractor/robot/robot.h>
#include <tractor/ros/visualize.h>

#include <moveit/robot_model_loader/robot_model_loader.h>

namespace tractor {

template <class Geometry>
static void pythonizeRobot(py::module main_module, py::module type_module) {
  main_module.def("visualize", [](const std::string &topic,
                                  const JointState<Geometry> &joint_state) {
    visualize<Geometry>(topic, joint_state);
  });

  main_module.def("publish", [](const std::string &topic,
                                const JointState<Geometry> &joint_state) {
    publish<Geometry>(topic, joint_state);
  });

  main_module.def(
      "visualize_joint_states",
      [](const std::string &topic, const JointState<Geometry> &joint_state) {
        visualize<Geometry>(topic, joint_state);
      });

  main_module.def(
      "visualize_joint_states",
      [](const std::string &topic, const JointState<Geometry> &joint_state) {
        publish<Geometry>(topic, joint_state);
      });

  main_module.def("visualize",
                  [](const std::string &topic,
                     const std::vector<JointState<Geometry>> &trajectory) {
                    visualize(topic, trajectory, typename Geometry::Value(0.1));
                  });

  main_module.def("visualize",
                  [](const std::string &topic,
                     const std::vector<JointState<Geometry>> &trajectory,
                     const typename Geometry::Value &time_step) {
                    visualize(topic, trajectory, time_step);
                  });

  static Factory::Key<std::string>::Value<std::shared_ptr<RobotModel<Geometry>>>
      robot_model_factory([](const std::string &robot_description) {
        TRACTOR_DEBUG("loading robot model " << robot_description);
        auto loader = std::make_shared<robot_model_loader::RobotModelLoader>(
            robot_description);
        auto model = loader->getModel();
        if (!model) {
          throw std::runtime_error("failed to load robot model: " +
                                   robot_description);
        }
        return std::static_pointer_cast<RobotModel<Geometry>>(
            std::shared_ptr<PyRobotModel<Geometry>>(
                new PyRobotModel<Geometry>(model, nullptr)));
      });

  py::class_<JointLimits<Geometry>>(type_module, "JointLimits")
      .def_property_readonly("lower",
                             [](const JointLimits<Geometry> &_this) {
                               return typename Geometry::Scalar(_this.lower());
                             })
      .def_property_readonly("upper", [](const JointLimits<Geometry> &_this) {
        return typename Geometry::Scalar(_this.upper());
      });

  ptr_class<LinkModel<Geometry>>(type_module, "LinkModel")
      .def_property_readonly("name", &LinkModel<Geometry>::name)
      .def_property_readonly("inertia", &LinkModel<Geometry>::inertia)
      .def_property_readonly("parent_joint", &LinkModel<Geometry>::parentJoint)
      .def_property_readonly("child_joints", &LinkModel<Geometry>::childJoints);

  ptr_class<JointModelBase<Geometry>>(type_module, "JointModel")
      .def_property_readonly("parent_link",
                             &JointModelBase<Geometry>::parentLink)
      .def_property_readonly("child_link", &JointModelBase<Geometry>::childLink)
      .def_property_readonly("name", &JointModelBase<Geometry>::name)
      .def_property_readonly("origin", &JointModelBase<Geometry>::origin);
  ptr_class<JointStateBase<Geometry>>(type_module, "JointState");

  ptr_class<FixedJointModel<Geometry>, JointModelBase<Geometry>>(
      type_module, "FixedJointModel");
  ptr_class<FixedJointState<Geometry>, JointStateBase<Geometry>>(
      type_module, "FixedJointState");

  ptr_class<PlanarJointModel<Geometry>, JointModelBase<Geometry>>(
      type_module, "PlanarJointModel");
  ptr_class<PlanarJointState<Geometry>, JointStateBase<Geometry>>(
      type_module, "PlanarJointState");

  ptr_class<FloatingJointModel<Geometry>, JointModelBase<Geometry>>(
      type_module, "FloatingJointModel");
  ptr_class<FloatingJointState<Geometry>, JointStateBase<Geometry>>(
      type_module, "FloatingJointState")
      .def_property(
          "pose",
          [](FloatingJointState<Geometry> &_this) { return &_this.pose(); },
          [](FloatingJointState<Geometry> &_this,
             const typename Geometry::Pose &pose) { _this.pose() = pose; },
          py::return_value_policy::reference_internal);

  ptr_class<ScalarJointModelBase<Geometry>, JointModelBase<Geometry>>(
      type_module, "ScalarJointModelBase")
      .def_property_readonly(
          "limits", [](const ScalarJointModelBase<Geometry> &_this) {
            if (_this.limits()) {
              return std::optional<JointLimits<Geometry>>(_this.limits());
            } else {
              return std::optional<JointLimits<Geometry>>();
            }
          });
  ptr_class<ScalarJointStateBase<Geometry>, JointStateBase<Geometry>>(
      type_module, "ScalarJointStateBase")
      .def_property(
          "position",
          [](ScalarJointStateBase<Geometry> &_this) {
            return &_this.position();
          },
          [](ScalarJointStateBase<Geometry> &_this,
             const typename Geometry::Scalar &position) {
            _this.position() = position;
          },
          py::return_value_policy::reference_internal);

  ptr_class<RevoluteJointModel<Geometry>, ScalarJointModelBase<Geometry>>(
      type_module, "RevoluteJointModel");
  ptr_class<RevoluteJointState<Geometry>, ScalarJointStateBase<Geometry>>(
      type_module, "RevoluteJointState");

  ptr_class<PrismaticJointModel<Geometry>, ScalarJointModelBase<Geometry>>(
      type_module, "PrismaticJointModel");
  ptr_class<PrismaticJointState<Geometry>, ScalarJointStateBase<Geometry>>(
      type_module, "PrismaticJointState");

  py::class_<LinkState<Geometry>>(type_module, "LinkStates")
      .def(py::init<const std::shared_ptr<const RobotModel<Geometry>> &>())
      .def("link_pose",
           [](LinkState<Geometry> &_this, const std::string &name) {
             return _this.pose(name);
           })
      .def("link_pose", [](LinkState<Geometry> &_this, size_t index) {
        return _this.pose(index);
      });

  py::class_<JointState<Geometry>>(type_module, "JointStates")
      .def(py::init<const std::shared_ptr<const RobotModel<Geometry>> &>())
      .def("__copy__",
           [](const JointState<Geometry> &self) {
             return std::make_unique<JointState<Geometry>>(self);
           })
      .def("serialize",
           [](JointState<Geometry> &_this) {
             AlignedStdVector<typename Geometry::Scalar> positions;
             _this.serializePositions(positions);
             return positions;
           })
      .def("deserialize",
           [](JointState<Geometry> &_this,
              const AlignedStdVector<typename Geometry::Scalar> &positions) {
             _this.deserializePositions(positions);
           })
      .def(
          "joint_state",
          [](JointState<Geometry> &_this, size_t i) { return &_this.joint(i); },
          py::return_value_policy::reference_internal)
      .def(
          "joint_state",
          [](JointState<Geometry> &_this, const std::string &name) {
            return &_this.joint(name);
          },
          py::return_value_policy::reference_internal)
      .def("update_mimic_joints", &JointState<Geometry>::updateMimicJoints);

  main_module.def("variable", [](JointState<Geometry> &joint_states) {
    for (size_t i = 0; i < joint_states.model()->info()->joints().count();
         i++) {
      auto &joint_model = joint_states.model()->joint(i);
      auto &joint_state = joint_states.pointer(i);
      if (auto *rec = Recorder::instance()) {
        rec->reference(joint_state);
      }
      joint_state->makeVariables(*joint_model,
                                 JointVariableOptions<Geometry>());
    }
  });

  py::class_<RobotState<Geometry>>(type_module, "RobotState")
      .def(py::init<const std::shared_ptr<const RobotModel<Geometry>> &>())
      .def_property_readonly(
          "joint_states",
          [](RobotState<Geometry> &_this) { return &_this.joints(); },
          py::return_value_policy::reference_internal)
      .def_property_readonly(
          "link_states",
          [](RobotState<Geometry> &_this) { return &_this.links(); },
          py::return_value_policy::reference_internal);

  py::class_<RobotModel<Geometry>, std::shared_ptr<RobotModel<Geometry>>>(
      type_module, "RobotModel")
      .def(py::init([](const std::string &robot_description) {
        return robot_model_factory.get(robot_description);
      }))
      .def(py::init(
          []() { return robot_model_factory.get("/robot_description"); }))
      .def(py::init([](const std::string &urdf, const std::string &srdf) {
        rdf_loader::RDFLoader loader(urdf, srdf);
        return std::static_pointer_cast<RobotModel<Geometry>>(
            std::make_shared<PyRobotModel<Geometry>>(
                std::make_shared<moveit::core::RobotModel>(loader.getURDF(),
                                                           loader.getSRDF()),
                nullptr));
      }))
      .def("forward_kinematics",
           [](const RobotModel<Geometry> &robot_model,
              const JointState<Geometry> &joint_state,
              LinkState<Geometry> &link_state) {
             robot_model.computeFK(joint_state, link_state);
           })
      .def("forward_kinematics",
           [](const std::shared_ptr<RobotModel<Geometry>> &robot_model,
              const JointState<Geometry> &joint_state) {
             LinkState<Geometry> link_state(robot_model);
             robot_model->computeFK(joint_state, link_state);
             return link_state;
           })
      .def("forward_kinematics",
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
                             &RobotModel<Geometry>::variableCount)
      .def_property_readonly("root_joint", &RobotModel<Geometry>::rootJoint)

      .def("joint", [](const RobotModel<Geometry> &_this,
                       size_t index) { return _this.joint(index); })
      .def("joint", [](const RobotModel<Geometry> &_this,
                       const std::string &name) { return _this.joint(name); })

      .def("link", [](const RobotModel<Geometry> &_this,
                      size_t index) { return _this.link(index); })
      .def("link", [](const RobotModel<Geometry> &_this,
                      const std::string &name) { return _this.link(name); })

      .def_property_readonly("joints", &RobotModel<Geometry>::joints)
      .def_property_readonly("links", &RobotModel<Geometry>::links)

      .def("is_mimic_joint",
           [](const RobotModel<Geometry> &_this, size_t index) {
             return _this.info()->joints().info(index).isMimicJoint();
           })
      .def("is_mimic_joint",
           [](const RobotModel<Geometry> &_this, const std::string &name) {
             return _this.info()->joints().info(name).isMimicJoint();
           });
}

TRACTOR_PYTHON_GEOMETRY(pythonizeRobot);

}  // namespace tractor
