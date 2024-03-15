// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/constraints.h>
#include <tractor/core/error.h>
#include <tractor/core/ops.h>
#include <tractor/core/var.h>
#include <tractor/geometry/eigen.h>
#include <tractor/geometry/eigen_ops.h>
#include <tractor/geometry/fast.h>

namespace tractor {

template <class Type, size_t ArraySize>
struct PythonArrayHelper {
  typedef std::array<Type, ArraySize> ImportType;
  typedef std::array<Type, ArraySize> InternalType;
  typedef py::array_t<Type> ExportType;
  static ExportType toPython(const InternalType &a) {
    py::array_t<Type> x({ArraySize});
    for (size_t i = 0; i < ArraySize; i++) {
      x.mutable_at(i) = a[i];
    }
    return x;
  }
  static InternalType fromPython(const ImportType &a) { return a; }
};

template <class ElementType, size_t BatchSize, size_t ArraySize>
struct PythonArrayHelper<Batch<ElementType, BatchSize>, ArraySize> {
  typedef std::array<std::array<ElementType, BatchSize>, ArraySize> ImportType;
  typedef std::array<Batch<ElementType, BatchSize>, ArraySize> InternalType;
  typedef py::array_t<ElementType> ExportType;
  static ExportType toPython(const InternalType &a) {
    py::array_t<ElementType> x({ArraySize, BatchSize});
    for (size_t ia = 0; ia < ArraySize; ia++) {
      for (size_t ib = 0; ib < BatchSize; ib++) {
        x.mutable_at(ia, ib) = a[ia][ib];
      }
    }
    return x;
  }
  static InternalType fromPython(const ImportType &a) {
    InternalType x;
    for (size_t ia = 0; ia < ArraySize; ia++) {
      for (size_t ib = 0; ib < BatchSize; ib++) {
        x[ia][ib] = a[ia][ib];
      }
    }
    return x;
  }
};

template <class Geometry>
static void pythonizeGeometry(py::module mod_main, py::module mod_type) {
  typedef typename Geometry::Value Value;
  typedef typename Geometry::Scalar Scalar;
  typedef typename Geometry::Vector3 Vector3;
  typedef typename Geometry::Matrix3 Matrix3;
  typedef typename Geometry::Pose Pose;
  typedef typename Geometry::Orientation Orientation;
  typedef typename Geometry::Twist Twist;

  pythonizeType<Twist>(mod_main, mod_type, "Twist")
      .def(py::init([]() { return Geometry::TwistZero(); }))
      .def_property_readonly_static("zero",
                                    [](const py::object &_this) {
                                      // std::cout << "twist zero" << _this
                                      //           << std::endl;
                                      return Geometry::TwistZero();
                                    })
      .def(py::init([](const Vector3 &translation, const Vector3 &rotation) {
        return Geometry::twist(translation, rotation);
      }))
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * Scalar())
      .def(Scalar() * py::self)
      .def(-py::self);
  mod_main.def("translation",
               py::overload_cast<const Twist &>(&Geometry::translation));
  mod_main.def("rotation",
               py::overload_cast<const Twist &>(&Geometry::rotation));
  mod_main.def("unpack", [](const Twist &twist) {
    Vector3 t, r;
    Geometry::unpack(twist, t, r);
    return std::make_tuple(t, r);
  });
  pythonizeType<Pose>(mod_main, mod_type, "Pose")
      .def(py::init([]() { return Geometry::PoseIdentity(); }))
      .def_static("angle_axis",
                  py::overload_cast<const Scalar &, const Vector3 &>(
                      &Geometry::angleAxisPose))
      .def_static(
          "angle_axis",
          py::overload_cast<const Pose &, const Scalar &, const Vector3 &>(
              &Geometry::angleAxisPose))
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::PoseIdentity(); })
      .def(py::init(
          [](const Vector3 &translation, const Orientation &orientation) {
            return Geometry::pack(translation, orientation);
          }))
      .def_property(
          "value",
          [](const Pose &pose) {
            return std::make_pair(std::array<Value, 3>({
                                      value(value(pose).position().x()),
                                      value(value(pose).position().y()),
                                      value(value(pose).position().z()),
                                  }),
                                  std::array<Value, 4>({
                                      value(value(pose).orientation().x()),
                                      value(value(pose).orientation().y()),
                                      value(value(pose).orientation().z()),
                                      value(value(pose).orientation().w()),
                                  }));
          },
          [](Pose &pose, const std::pair<std::array<Value, 3>,
                                         std::array<Value, 4>> &data) {
            value(value(pose).position().x()) = data.first[0];
            value(value(pose).position().y()) = data.first[1];
            value(value(pose).position().z()) = data.first[2];
            value(value(pose).orientation().x()) = data.second[0];
            value(value(pose).orientation().y()) = data.second[1];
            value(value(pose).orientation().z()) = data.second[2];
            value(value(pose).orientation().w()) = data.second[3];
          })
      .def(py::self * py::self)
      .def(py::self * Vector3())
      .def(py::self + Twist());
  mod_main.def("inverse", py::overload_cast<const Pose &>(&Geometry::inverse));
  mod_main.def("translation",
               py::overload_cast<const Pose &>(&Geometry::translation));
  mod_main.def("position",
               py::overload_cast<const Pose &>(&Geometry::position));
  mod_main.def("orientation", &Geometry::orientation);
  mod_main.def("residual", py::overload_cast<const Pose &, const Pose &>(
                               &Geometry::residual));
  mod_main.def("residual",
               py::overload_cast<const Pose &>(&Geometry::residual));
  mod_main.def("unpack", [](const Pose &twist) {
    Vector3 p;
    Orientation o;
    Geometry::unpack(twist, p, o);
    return std::make_tuple(p, o);
  });
  pythonizeType<Orientation>(mod_main, mod_type, "Orientation")
      .def(py::init([]() { return Geometry::OrientationIdentity(); }))
      .def_static("angle_axis",
                  py::overload_cast<const Scalar &, const Vector3 &>(
                      &Geometry::angleAxisOrientation))
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::OrientationIdentity(); })
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
      .def_property(
          "value",
          [](const Orientation &_this) {
            typename PythonArrayHelper<Value, 4>::InternalType r;
            r[0] = value(value(_this).x());
            r[1] = value(value(_this).y());
            r[2] = value(value(_this).z());
            r[3] = value(value(_this).w());
            return PythonArrayHelper<Value, 4>::toPython(r);
          },
          [](Orientation &_this,
             const typename PythonArrayHelper<Value, 4>::ImportType &py_array) {
            auto array = PythonArrayHelper<Value, 4>::fromPython(py_array);
            value(value(_this).x()) = array.at(0);
            value(value(_this).y()) = array.at(1);
            value(value(_this).z()) = array.at(2);
            value(value(_this).w()) = array.at(3);
          })
      .def(py::self * py::self)
      .def(py::self * Vector3())
      .def(py::self + Vector3());
  mod_main.def("inverse",
               py::overload_cast<const Orientation &>(&Geometry::inverse));
  mod_main.def("residual",
               py::overload_cast<const Orientation &, const Orientation &>(
                   &Geometry::residual));
  mod_main.def("residual",
               py::overload_cast<const Orientation &>(&Geometry::residual));
  mod_main.def("unpack", [](const Orientation &q) {
    std::array<Scalar, 4> ret;
    Geometry::unpack(q, ret[0], ret[1], ret[2], ret[3]);
    return ret;
  });

  pythonizeType<Matrix3>(mod_main, mod_type, "Matrix3")
      .def(py::init([]() { return Geometry::Matrix3Zero(); }))
      .def_property_readonly_static(
          "identity",
          [](const py::object &) { return Geometry::Matrix3Identity(); })
      .def_property_readonly_static(
          "zero", [](const py::object &) { return Geometry::Matrix3Zero(); })
      .def(py::init([](const std::array<std::array<Value, 3>, 3> &array) {
        Eigen::Matrix<Value, 3, 3> r;
        for (size_t i = 0; i < 3; i++) {
          for (size_t j = 0; j < 3; j++) {
            r(i, j) = array[i][j];
          }
        }
        return Geometry::importMatrix3(r);
      }))
      .def(py::self * Vector3())
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(-py::self)

      .def_property(
          "value",
          [](const Matrix3 &_this) {
            std::array<std::array<Value, 3>, 3> r;
            for (size_t i = 0; i < 3; i++) {
              for (size_t j = 0; j < 3; j++) {
                r[i][j] = value(value(_this)(i, j));
              }
            }
            return r;
          },
          [](Matrix3 &_this, const std::array<std::array<Value, 3>, 3> &array) {
            for (size_t i = 0; i < 3; i++) {
              for (size_t j = 0; j < 3; j++) {
                value(value(_this)(i, j)) = array[i][j];
              }
            }
          })
      .def(-py::self);
  mod_main.def("inverse", [](const Matrix3 &m) { return inverse(m); });

  pythonizeType<Vector3>(mod_main, mod_type, "Vector3")
      .def(py::init([]() { return Geometry::Vector3Zero(); }))
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
      .def_property(
          "value",
          [](const Vector3 &_this) {
            typename PythonArrayHelper<Value, 3>::InternalType r;
            r.at(0) = value(value(_this).x());
            r.at(1) = value(value(_this).y());
            r.at(2) = value(value(_this).z());
            return PythonArrayHelper<Value, 3>::toPython(r);
          },
          [](Vector3 &_this,
             const typename PythonArrayHelper<Value, 3>::ImportType &py_array) {
            auto array = PythonArrayHelper<Value, 3>::fromPython(py_array);
            value(value(_this).x()) = array.at(0);
            value(value(_this).y()) = array.at(1);
            value(value(_this).z()) = array.at(2);
          })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * Scalar())
      .def(Scalar() * py::self)
      .def(-py::self);
  mod_main.def("cross", py::overload_cast<const Vector3 &, const Vector3 &>(
                            &Geometry::cross));
  mod_main.def("dot", py::overload_cast<const Vector3 &, const Vector3 &>(
                          &Geometry::dot));
  mod_main.def("norm", py::overload_cast<const Vector3 &>(&Geometry::norm));
  mod_main.def("squaredNorm",
               py::overload_cast<const Vector3 &>(&Geometry::squaredNorm));
  mod_main.def("normalized",
               py::overload_cast<const Vector3 &>(&Geometry::normalized));
  mod_main.def("unpack", [](const Vector3 &v) {
    std::array<Scalar, 3> ret;
    Geometry::unpack(v, ret[0], ret[1], ret[2]);
    return ret;
  });
}
TRACTOR_PYTHON_GEOMETRY_BATCH(pythonizeGeometry);

}  // namespace tractor
