// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/collision/bullet.h>
#include <tractor/collision/loader.h>
#include <tractor/collision/ops.h>
#include <tractor/collision/robot.h>
#include <tractor/geometry/fast.h>
#include <tractor/robot/robot.h>

namespace tractor {

static void pythonizeCollisionGlobal(py::module main_module) {

  py::class_<CollisionShape, std::shared_ptr<CollisionShape>>(main_module,
                                                              "CollisionShape")
      .def("sample", [](const CollisionShape &_this, size_t n) {
        Eigen::MatrixXd ret(n, 6);
        for (size_t i = 0; i < n; i++) {
          Eigen::Vector3d pos, norm;
          _this.sample(pos, norm);
          ret.row(i).head(3) = pos;
          ret.row(i).tail(3) = norm;
        }
        return ret;
      });

  py::class_<CollisionLink, std::shared_ptr<CollisionLink>>(main_module,
                                                            "CollisionLink")
      .def_property_readonly("shapes", &CollisionLink::shapes)
      .def_property_readonly("name", &CollisionLink::name);
}

TRACTOR_PYTHON_GLOBAL(pythonizeCollisionGlobal);

template <class Geometry>
static void pythonizeCollisionTwist(py::module main_module,
                                    py::module type_module) {

  static auto engine = std::make_shared<BulletCollisionEngine>();

  struct CollisionRobot : tractor::CollisionRobot {
    CollisionRobot(const std::shared_ptr<CollisionEngine> &r)
        : tractor::CollisionRobot(r) {}
  };

  py::class_<CollisionRobot>(type_module, "CollisionRobot")
      .def(py::init([](const RobotModel<Geometry> &robot_model) {
        auto *ret = new CollisionRobot(engine);
        loadCollisionRobot(
            engine,
            *((const PyRobotModel<Geometry> *)&robot_model)->moveit_model, ret);
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

  struct PySurfacePoint {
    typename Geometry::Vector3 point = Geometry::Vector3Zero();
    typename Geometry::Vector3 normal = Geometry::Vector3Zero();
  };

  py::class_<PySurfacePoint>(type_module, "SurfacePoint")
      .def_readonly("point", &PySurfacePoint::point)
      .def_readonly("normal", &PySurfacePoint::normal);

  main_module.def("project", [](const typename Geometry::Vector3 &in_point,
                                const std::shared_ptr<CollisionShape> &shape) {
    PySurfacePoint ret;
    project<Geometry>(in_point, shape, ret.point, ret.normal);
    return ret;
  });
}

TRACTOR_PYTHON_TWIST(pythonizeCollisionTwist);

} // namespace tractor
