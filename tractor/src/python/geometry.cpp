// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/constraints.h>
#include <tractor/core/error.h>
#include <tractor/core/ops.h>
#include <tractor/core/var.h>
#include <tractor/geometry/fast.h>

namespace tractor {

template <class Geometry>
static void pythonizeGeometry(py::module main_module, py::module type_module) {

  // -------------------------------------------------------------
  // Twist
  pythonizeType<typename Geometry::Twist>(main_module, type_module, "Twist")
      .def_property_readonly_static(
          "zero", [](const py::object &) { return Geometry::TwistZero(); })
      .def(py::init([]() { return Geometry::TwistZero(); }))
      .def(py::init([](const typename Geometry::Vector3 &translation,
                       const typename Geometry::Vector3 &rotation) {
        return Geometry::twist(translation, rotation);
      }))
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * typename Geometry::Scalar())
      .def(typename Geometry::Scalar() * py::self)
      .def(-py::self);
  type_module.def("translation",
                  py::overload_cast<const typename Geometry::Twist &>(
                      &Geometry::translation));
  type_module.def(
      "rotation",
      py::overload_cast<const typename Geometry::Twist &>(&Geometry::rotation));
  type_module.def("translation_twist",
                  py::overload_cast<const typename Geometry::Vector3 &>(
                      &Geometry::translationTwist));

  // -------------------------------------------------------------
  // Pose
  pythonizeType<typename Geometry::Pose>(main_module, type_module, "Pose")
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::PoseIdentity(); })
      .def(py::init([]() { return Geometry::PoseIdentity(); }))
      .def(py::init([](const typename Geometry::Vector3 &translation,
                       const typename Geometry::Orientation &orientation) {
        return Geometry::translationPose(translation) *
               Geometry::orientationPose(orientation);
      }))
      .def(py::self * py::self)
      .def(py::self * typename Geometry::Vector3());
  type_module.def("inverse", py::overload_cast<const typename Geometry::Pose &>(
                                 &Geometry::inverse));
  type_module.def("translation",
                  py::overload_cast<const typename Geometry::Pose &>(
                      &Geometry::translation));
  type_module.def(
      "position",
      py::overload_cast<const typename Geometry::Pose &>(&Geometry::position));
  type_module.def("orientation", &Geometry::orientation);
  // type_module.def(
  //     "residual",
  //     py::overload_cast<const typename Geometry::Pose &,
  //                       const typename Geometry::Pose
  //                       &>(&Geometry::residual));
  type_module.def("angle_axis_pose",
                  py::overload_cast<const typename Geometry::Scalar &,
                                    const typename Geometry::Vector3 &>(
                      &Geometry::angleAxisPose));
  type_module.def("angle_axis_pose",
                  py::overload_cast<const typename Geometry::Pose &,
                                    const typename Geometry::Scalar &,
                                    const typename Geometry::Vector3 &>(
                      &Geometry::angleAxisPose));
  type_module.def("translation_pose",
                  py::overload_cast<const typename Geometry::Vector3 &>(
                      &Geometry::translationPose));
  type_module.def("translation_pose",
                  py::overload_cast<const typename Geometry::Pose &,
                                    const typename Geometry::Vector3 &>(
                      &Geometry::translationPose));

  // -------------------------------------------------------------
  // Orientation
  pythonizeType<typename Geometry::Orientation>(main_module, type_module,
                                                "Orientation")
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::OrientationIdentity(); })
      .def(py::init([]() { return Geometry::OrientationIdentity(); }))
      .def(py::init([](const typename Geometry::Scalar &x,
                       const typename Geometry::Scalar &y,
                       const typename Geometry::Scalar &z,
                       const typename Geometry::Scalar &w) {
        return Geometry::pack(x, y, z, w);
      }))
      .def(py::init([](const typename Geometry::Value &x,
                       const typename Geometry::Value &y,
                       const typename Geometry::Value &z,
                       const typename Geometry::Value &w) {
        return Geometry::pack(x, y, z, w);
      }))
      .def(py::init([](const std::array<typename Geometry::Value, 4> &array) {
        return Geometry::pack(array[0], array[1], array[2], array[3]);
      }))
      .def(py::init([](const std::array<typename Geometry::Scalar, 4> &array) {
        return Geometry::pack(array[0], array[1], array[2], array[3]);
      }))
      .def_property_readonly("value",
                             [](const typename Geometry::Orientation &_this) {
                               typename Geometry::Scalar x, y, z, w;
                               Geometry::unpack(_this, x, y, z, w);
                               py::array_t<typename Geometry::Value> r(4);
                               r.mutable_at(0) = value(x);
                               r.mutable_at(1) = value(y);
                               r.mutable_at(2) = value(z);
                               r.mutable_at(3) = value(w);
                               return r;
                             })
      .def(py::self * py::self)
      .def(py::self * typename Geometry::Vector3());
  type_module.def("inverse",
                  py::overload_cast<const typename Geometry::Orientation &>(
                      &Geometry::inverse));
  type_module.def(
      "pack",
      py::overload_cast<
          const typename Geometry::Scalar &, const typename Geometry::Scalar &,
          const typename Geometry::Scalar &, const typename Geometry::Scalar &>(
          &Geometry::pack));
  // type_module.def("residual",
  //                 py::overload_cast<const typename Geometry::Orientation &,
  //                                   const typename Geometry::Orientation &>(
  //                     &Geometry::residual));
  type_module.def("angle_axis_orientation",
                  py::overload_cast<const typename Geometry::Scalar &,
                                    const typename Geometry::Vector3 &>(
                      &Geometry::angleAxisOrientation));

  // -------------------------------------------------------------
  // Matrix3
  pythonizeType<typename Geometry::Matrix3>(main_module, type_module, "Matrix3")
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::Matrix3Identity(); })
      .def_property_readonly_static(
          "zero", [](const py::object &) { return Geometry::Matrix3Zero(); })
      .def(py::init([]() { return Geometry::Matrix3Zero(); }))
      .def(py::init(
          [](const Eigen::Matrix<typename Geometry::Value, 3, 3> &value) {
            return Geometry::import(value);
          }))
      .def(py::self * typename Geometry::Vector3())
      .def(py::self + py::self)
      .def(-py::self)
      .def_property_readonly("value",
                             [](const typename Geometry::Matrix3 &_this) {
                               py::array_t<typename Geometry::Value> r({3, 3});
                               for (size_t i = 0; i < 3; i++) {
                                 for (size_t j = 0; j < 3; j++) {
                                   r.mutable_at(i, j) =
                                       value(value(_this)(i, j));
                                 }
                               }
                               return r;
                             })
      .def(-py::self);

  // -------------------------------------------------------------
  // Vector3
  pythonizeType<typename Geometry::Vector3>(main_module, type_module, "Vector3")
      .def_property_readonly_static(
          "zero", [](const py::object &) { return Geometry::Vector3Zero(); })
      .def(py::init([](const std::array<typename Geometry::Value, 3> &array) {
        return Geometry::pack(array[0], array[1], array[2]);
      }))
      .def(py::init([](const std::array<typename Geometry::Scalar, 3> &array) {
        return Geometry::pack(array[0], array[1], array[2]);
      }))
      .def(py::init([](const typename Geometry::Value &x,
                       const typename Geometry::Value &y,
                       const typename Geometry::Value &z) {
        return Geometry::pack(x, y, z);
      }))
      .def(py::init([](const typename Geometry::Scalar &x,
                       const typename Geometry::Scalar &y,
                       const typename Geometry::Scalar &z) {
        return Geometry::pack(x, y, z);
      }))
      .def_property_readonly("value",
                             [](const typename Geometry::Vector3 &_this) {
                               typename Geometry::Scalar x, y, z;
                               Geometry::unpack(_this, x, y, z);
                               py::array_t<typename Geometry::Value> r(3);
                               r.mutable_at(0) = value(x);
                               r.mutable_at(1) = value(y);
                               r.mutable_at(2) = value(z);
                               return r;
                             })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * typename Geometry::Scalar())
      .def(typename Geometry::Scalar() * py::self)
      .def(-py::self);
  type_module.def("pack", py::overload_cast<const typename Geometry::Scalar &,
                                            const typename Geometry::Scalar &,
                                            const typename Geometry::Scalar &>(
                              &Geometry::pack));
}

static int _tractor_python_geometry = []() {
  PythonRegistry::instance()->add([](py::module mod_main) {
    auto mod_geometry = mod_main.def_submodule("geometry");
    {
      auto mod_vector = mod_geometry.def_submodule("block");
      pythonizeGeometry<GeometryFast<Var<float>>>(
          mod_main, mod_vector.def_submodule("float"));
      pythonizeGeometry<GeometryFast<Var<double>>>(
          mod_main, mod_vector.def_submodule("double"));
    }
    {
      auto mod_vector = mod_geometry.def_submodule("scalar");
      pythonizeGeometry<GeometryScalar<Var<float>>>(
          mod_main, mod_vector.def_submodule("float"));
      pythonizeGeometry<GeometryScalar<Var<double>>>(
          mod_main, mod_vector.def_submodule("double"));
    }
  });
  return 0;
}();

} // namespace tractor
