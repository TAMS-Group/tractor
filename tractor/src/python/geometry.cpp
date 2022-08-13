// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/constraints.h>
#include <tractor/core/error.h>
#include <tractor/core/ops.h>
#include <tractor/core/var.h>
#include <tractor/geometry/fast.h>

namespace tractor {

template <class Geometry>
static void pythonizeGeometry(py::module mod_main, py::module mod_type) {

  // -------------------------------------------------------------
  typedef typename Geometry::Value Value;
  typedef typename Geometry::Scalar Scalar;
  typedef typename Geometry::Vector3 Vector3;
  typedef typename Geometry::Matrix3 Matrix3;
  typedef typename Geometry::Pose Pose;
  typedef typename Geometry::Orientation Orientation;

  // -------------------------------------------------------------
  // Twist
  pythonizeType<typename Geometry::Twist>(mod_main, mod_type, "Twist")
      .def_property_readonly_static(
          "zero", [](const py::object &) { return Geometry::TwistZero(); })
      .def(py::init([]() { return Geometry::TwistZero(); }))
      .def(py::init([](const Vector3 &translation, const Vector3 &rotation) {
        return Geometry::twist(translation, rotation);
      }))
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * Scalar())
      .def(Scalar() * py::self)
      .def(-py::self);
  mod_type.def("translation",
               py::overload_cast<const typename Geometry::Twist &>(
                   &Geometry::translation));
  mod_type.def("rotation", py::overload_cast<const typename Geometry::Twist &>(
                               &Geometry::rotation));
  mod_type.def("translation_twist",
               py::overload_cast<const Vector3 &>(&Geometry::translationTwist));

  // -------------------------------------------------------------
  // Pose
  pythonizeType<Pose>(mod_main, mod_type, "Pose")
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::PoseIdentity(); })
      .def(py::init([]() { return Geometry::PoseIdentity(); }))
      .def(py::init(
          [](const Vector3 &translation, const Orientation &orientation) {
            return Geometry::translationPose(translation) *
                   Geometry::orientationPose(orientation);
          }))
      .def(py::self * py::self)
      .def(py::self * Vector3());
  mod_type.def("inverse", py::overload_cast<const Pose &>(&Geometry::inverse));
  mod_type.def("translation",
               py::overload_cast<const Pose &>(&Geometry::translation));
  mod_type.def("position",
               py::overload_cast<const Pose &>(&Geometry::position));
  mod_type.def("orientation", &Geometry::orientation);
  // mod_type.def(
  //     "residual",
  //     py::overload_cast<const Pose &,
  //                       const Pose
  //                       &>(&Geometry::residual));
  mod_type.def("angle_axis_pose",
               py::overload_cast<const Scalar &, const Vector3 &>(
                   &Geometry::angleAxisPose));
  mod_type.def("angle_axis_pose",
               py::overload_cast<const Pose &, const Scalar &, const Vector3 &>(
                   &Geometry::angleAxisPose));
  mod_type.def("translation_pose",
               py::overload_cast<const Vector3 &>(&Geometry::translationPose));
  mod_type.def("translation_pose",
               py::overload_cast<const Pose &, const Vector3 &>(
                   &Geometry::translationPose));

  // -------------------------------------------------------------
  // Orientation
  pythonizeType<Orientation>(mod_main, mod_type, "Orientation")
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::OrientationIdentity(); })
      .def(py::init([]() { return Geometry::OrientationIdentity(); }))
      .def(py::init([](const Scalar &x, const Scalar &y, const Scalar &z,
                       const Scalar &w) { return Geometry::pack(x, y, z, w); }))
      .def(py::init([](const Value &x, const Value &y, const Value &z,
                       const Value &w) { return Geometry::pack(x, y, z, w); }))
      .def(py::init([](const std::array<Value, 4> &array) {
        return Geometry::pack(array[0], array[1], array[2], array[3]);
      }))
      .def(py::init([](const std::array<Scalar, 4> &array) {
        return Geometry::pack(array[0], array[1], array[2], array[3]);
      }))
      .def_property_readonly("value",
                             [](const Orientation &_this) {
                               Scalar x, y, z, w;
                               Geometry::unpack(_this, x, y, z, w);
                               py::array_t<Value> r(4);
                               r.mutable_at(0) = value(x);
                               r.mutable_at(1) = value(y);
                               r.mutable_at(2) = value(z);
                               r.mutable_at(3) = value(w);
                               return r;
                             })
      .def(py::self * py::self)
      .def(py::self * Vector3());
  mod_type.def("inverse",
               py::overload_cast<const Orientation &>(&Geometry::inverse));
  mod_type.def("pack",
               py::overload_cast<const Scalar &, const Scalar &, const Scalar &,
                                 const Scalar &>(&Geometry::pack));
  // mod_type.def("residual",
  //                 py::overload_cast<const Orientation &,
  //                                   const Orientation &>(
  //                     &Geometry::residual));
  mod_type.def("angle_axis_orientation",
               py::overload_cast<const Scalar &, const Vector3 &>(
                   &Geometry::angleAxisOrientation));

  // -------------------------------------------------------------
  // Matrix3
  pythonizeType<Matrix3>(mod_main, mod_type, "Matrix3")
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::Matrix3Identity(); })
      .def_property_readonly_static(
          "zero", [](const py::object &) { return Geometry::Matrix3Zero(); })
      .def(py::init([]() { return Geometry::Matrix3Zero(); }))
      .def(py::init([](const Eigen::Matrix<Value, 3, 3> &value) {
        return Geometry::import(value);
      }))
      .def(py::self * Vector3())
      .def(py::self + py::self)
      .def(-py::self)
      .def_property_readonly("value",
                             [](const Matrix3 &_this) {
                               py::array_t<Value> r({3, 3});
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
  pythonizeType<Vector3>(mod_main, mod_type, "Vector3")
      .def_property_readonly_static(
          "zero", [](const py::object &) { return Geometry::Vector3Zero(); })
      .def(py::init([](const std::array<Value, 3> &array) {
        return Geometry::pack(array[0], array[1], array[2]);
      }))
      .def(py::init([](const std::array<Scalar, 3> &array) {
        return Geometry::pack(array[0], array[1], array[2]);
      }))
      .def(py::init([](const Value &x, const Value &y, const Value &z) {
        return Geometry::pack(x, y, z);
      }))
      .def(py::init([](const Scalar &x, const Scalar &y, const Scalar &z) {
        return Geometry::pack(x, y, z);
      }))
      .def_property_readonly("value",
                             [](const Vector3 &_this) {
                               Scalar x, y, z;
                               Geometry::unpack(_this, x, y, z);
                               py::array_t<Value> r(3);
                               r.mutable_at(0) = value(x);
                               r.mutable_at(1) = value(y);
                               r.mutable_at(2) = value(z);
                               return r;
                             })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * Scalar())
      .def(Scalar() * py::self)
      .def(-py::self);
  mod_type.def("cross", py::overload_cast<const Vector3 &, const Vector3 &>(
                            &Geometry::cross));
  mod_type.def("dot", py::overload_cast<const Vector3 &, const Vector3 &>(
                          &Geometry::dot));
  mod_type.def("norm", py::overload_cast<const Vector3 &>(&Geometry::norm));
  mod_type.def("squaredNorm",
               py::overload_cast<const Vector3 &>(&Geometry::squaredNorm));
  mod_type.def("normalized",
               py::overload_cast<const Vector3 &>(&Geometry::normalized));
  mod_type.def(
      "pack", py::overload_cast<const Scalar &, const Scalar &, const Scalar &>(
                  &Geometry::pack));
  mod_type.def("unpack", [](const Vector3 &v) {
    std::array<Scalar, 3> ret;
    Geometry::unpack(v, ret[0], ret[1], ret[2]);
    return ret;
  });

  // -------------------------------------------------------------
}

TRACTOR_PYTHON_GEOMETRY(pythonizeGeometry);

// static int _tractor_python_geometry = []() {
//   PythonRegistry::instance()->add([](py::module mod_main) {
//     auto mod_geometry = mod_main.def_submodule("geometry");
//     {
//       auto mod_mode = mod_geometry.def_submodule("block");
//       pythonizeGeometry<GeometryFast<Var<float>>>(mod_main, mod_mode,
//       "float"); pythonizeGeometry<GeometryFast<Var<double>>>(mod_main,
//       mod_mode,
//                                                    "double");
//     }
//     {
//       auto mod_mode = mod_geometry.def_submodule("scalar");
//       pythonizeGeometry<GeometryScalar<Var<float>>>(mod_main, mod_mode,
//                                                     "float");
//       pythonizeGeometry<GeometryScalar<Var<double>>>(mod_main, mod_mode,
//                                                      "double");
//     }
//   });
//   return 0;
// }();

} // namespace tractor
