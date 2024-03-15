// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/dynamics/rigidbody.h>
#include <tractor/geometry/fast.h>

namespace tractor {

template <class Geometry>
static void pythonizeDynamics(py::module main_module, py::module type_module) {
  py::class_<Inertia<Geometry>>(type_module, "Inertia")
      .def_property_readonly("center", &Inertia<Geometry>::center)
      .def_property_readonly("mass", &Inertia<Geometry>::mass)
      .def_property_readonly("mass_inverse", &Inertia<Geometry>::massInverse)
      .def_property_readonly("moment", &Inertia<Geometry>::moment)
      .def_property_readonly("moment_inverse",
                             &Inertia<Geometry>::momentInverse);

  py::class_<RigidBody<Geometry>>(type_module, "RigidBody")
      .def(py::init<const typename Geometry::Pose &,
                    const Inertia<Geometry> &>())

      .def(py::init<const typename Geometry::Pose &, const Inertia<Geometry> &,
                    const typename Geometry::Twist &>())

      .def_property_readonly("inertia", &RigidBody<Geometry>::inertia)

      .def_property_readonly("pose", &RigidBody<Geometry>::pose)
      .def_property_readonly("position", &RigidBody<Geometry>::position)
      .def_property_readonly("orientation", &RigidBody<Geometry>::orientation)

      .def_property_readonly("local_velocity",
                             &RigidBody<Geometry>::localVelocity)
      .def_property_readonly("global_velocity",
                             &RigidBody<Geometry>::globalVelocity)
      .def_property_readonly("global_linear_velocity",
                             &RigidBody<Geometry>::globalLinearVelocity)
      .def_property_readonly("global_angular_velocity",
                             &RigidBody<Geometry>::globalAngularVelocity)

      .def("apply_damping", &RigidBody<Geometry>::applyDamping)
      .def("apply_acceleration", &RigidBody<Geometry>::applyAcceleration)
      .def("apply_force", py::overload_cast<const typename Geometry::Vector3 &>(
                              &RigidBody<Geometry>::applyForce))
      .def("apply_force", py::overload_cast<const typename Geometry::Vector3 &,
                                            const typename Geometry::Vector3 &>(
                              &RigidBody<Geometry>::applyForce))
      .def("apply_torque", &RigidBody<Geometry>::applyTorque)
      .def("apply_wrench", &RigidBody<Geometry>::applyWrench)

      .def("integrate", &RigidBody<Geometry>::integrate)
      .def("integrate", [](RigidBody<Geometry> &_this, double dt) {
        _this.integrate(typename Geometry::Value(dt));
      });
}

TRACTOR_PYTHON_GEOMETRY(pythonizeDynamics);

}  // namespace tractor
