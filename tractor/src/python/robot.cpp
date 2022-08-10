// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/collision/bullet.h>
#include <tractor/collision/loader.h>
#include <tractor/collision/ops.h>
#include <tractor/collision/robot.h>
#include <tractor/core/factory.h>
#include <tractor/geometry/fast.h>
#include <tractor/robot/robot.h>
#include <tractor/ros/visualize.h>

#include <moveit/robot_model_loader/robot_model_loader.h>

namespace tractor {

static void pythonizeRobotGlobal(py::module &main_module) {

  py::class_<CollisionShape, std::shared_ptr<CollisionShape>>(main_module,
                                                              "CollisionShape");

  py::class_<CollisionLink, std::shared_ptr<CollisionLink>>(main_module,
                                                            "CollisionLink")
      .def_property_readonly("shapes", &CollisionLink::shapes)
      .def_property_readonly("name", &CollisionLink::name);
}

TRACTOR_PYTHON_GLOBAL(pythonizeRobotGlobal);

template <class Scalar>
static void pythonizeRobot(py::module &main_module, py::module &type_module) {

  typedef GeometryFast<Var<Scalar>> Geometry;

  main_module.def("visualize", [](const std::string &topic,
                                  const JointState<Geometry> &joint_state) {
    visualize(topic, joint_state);
  });

  static Factory::Key<std::string>::Value<moveit::core::RobotModelConstPtr>
      robot_model_factory([](const std::string &robot_description) {
        TRACTOR_DEBUG("loading robot model " << robot_description);
        robot_model_loader::RobotModelLoader loader(robot_description);
        moveit::core::RobotModelConstPtr ret = loader.getModel();
        if (!ret) {
          throw std::runtime_error("failed to load robot model: " +
                                   robot_description);
        }
        return ret;
      });

  py::class_<JointLimits<Geometry>>(type_module, "JointLimits")
      .def_property_readonly("lower",
                             [](const JointLimits<Geometry> &_this) {
                               return typename Geometry::Scalar(_this.lower());
                             })
      .def_property_readonly("upper", [](const JointLimits<Geometry> &_this) {
        return typename Geometry::Scalar(_this.upper());
      });

  ptr_class<JointModelBase<Geometry>>(type_module, "JointModel")
      .def_property_readonly(
          "origin",
          [](const JointModelBase<Geometry> &_this) { return _this.origin(); })
      .def_property_readonly("inertia",
                             [](const JointModelBase<Geometry> &_this) {
                               return _this.inertia();
                             });
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
      .def("serialize",
           [](JointState<Geometry> &_this) {
             AlignedStdVector<Var<Scalar>> positions;
             _this.serializePositions(positions);
             return positions;
           })
      .def("deserialize",
           [](JointState<Geometry> &_this,
              const AlignedStdVector<Var<Scalar>> &positions) {
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
          py::return_value_policy::reference_internal);

  main_module.def("variable", [](JointState<Geometry> &joint_states) {
    for (size_t i = 0; i < joint_states.model()->info()->joints().count();
         i++) {
      auto &joint_model = joint_states.model()->joint(i);
      auto &joint_state = joint_states.pointer(i);
      if (auto *rec = Recorder::instance()) {
        rec->reference(joint_state);
      }
      joint_state->makeVariables(joint_model, JointVariableOptions<Geometry>());
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

  struct PyRobotModel : RobotModel<Geometry> {
    moveit::core::RobotModelConstPtr moveit_model;
    PyRobotModel(const moveit::core::RobotModelConstPtr &m)
        : RobotModel<Geometry>(*m), moveit_model(m) {
      TRACTOR_DEBUG("robot model created");
    }
    ~PyRobotModel() { TRACTOR_DEBUG("robot model destroyed"); }
  }; // namespace tractor
  py::class_<RobotModel<Geometry>, std::shared_ptr<RobotModel<Geometry>>>(
      type_module, "RobotModel")
      .def(py::init([](const std::string &robot_description) {
        return std::static_pointer_cast<RobotModel<Geometry>>(
            std::make_shared<PyRobotModel>(
                robot_model_factory.get(robot_description)));
      }))
      .def(py::init([]() {
        return std::static_pointer_cast<RobotModel<Geometry>>(
            std::make_shared<PyRobotModel>(
                robot_model_factory.get("/robot_description")));
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
      .def(
          "joint_model",
          [](RobotModel<Geometry> &_this, size_t i) { return &_this.joint(i); },
          py::return_value_policy::reference_internal)
      .def(
          "joint_model",
          [](RobotModel<Geometry> &_this, const std::string &name) {
            return &_this.joint(name);
          },
          py::return_value_policy::reference_internal);

  static auto engine = std::make_shared<BulletCollisionEngine>();

  struct CollisionRobot : tractor::CollisionRobot {
    CollisionRobot(const std::shared_ptr<CollisionEngine> &r)
        : tractor::CollisionRobot(r) {}
  };

  py::class_<CollisionRobot>(type_module, "CollisionRobot")
      .def(py::init([](const RobotModel<Geometry> &robot_model) {
        auto *ret = new CollisionRobot(engine);
        loadCollisionRobot(
            engine, *((const PyRobotModel *)&robot_model)->moveit_model, ret);
        return ret;
      }))
      .def_property_readonly("links", &CollisionRobot::links)
      .def("link", &CollisionRobot::link);

  py::class_<CollisionResult<Geometry>>(type_module, "CollisionResult")
      .def_readonly("point_a", &CollisionResult<Geometry>::point_a)
      .def_readonly("point_b", &CollisionResult<Geometry>::point_b)
      .def_readonly("normal", &CollisionResult<Geometry>::normal)
      .def_readonly("distance", &CollisionResult<Geometry>::distance);

  main_module.def("collide",
                  [](const typename Geometry::Pose &pose_a,
                     const std::shared_ptr<CollisionShape> &shape_a,
                     const typename Geometry::Pose &pose_b,
                     const std::shared_ptr<CollisionShape> &shape_b) {
                    return collide<Geometry>(pose_a, shape_a, pose_b, shape_b);
                  });

  main_module.def("collide", [](const typename Geometry::Pose &pose_a,
                                const std::shared_ptr<CollisionLink> &link_a,
                                const typename Geometry::Pose &pose_b,
                                const std::shared_ptr<CollisionLink> &link_b) {
    return collide<Geometry>(pose_a, link_a, pose_b, link_b);
  });
}

TRACTOR_PYTHON_TYPED(pythonizeRobot);

} // namespace tractor
