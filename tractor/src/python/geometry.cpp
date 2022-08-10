// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/constraints.h>
#include <tractor/core/error.h>
#include <tractor/core/ops.h>
#include <tractor/core/var.h>
#include <tractor/geometry/fast.h>

namespace tractor {

template <class Scalar>
static void pythonizeGeometry(py::module &main_module,
                              py::module &type_module) {

  typedef GeometryFast<Var<Scalar>> Geometry;

  pythonizeType<Var<Twist<Scalar>>>(main_module, type_module, "Twist")
      .def(py::self + py::self);

  pythonizeType<Var<Pose<Scalar>>>(main_module, type_module, "Pose")
      .def(py::init([]() { return Geometry::PoseIdentity(); }))
      .def(py::self * py::self)
      .def(py::self * typename Geometry::Vector3());

  pythonizeType<Var<Quaternion<Scalar>>>(main_module, type_module, "Quaternion")
      .def(py::init([]() { return Geometry::OrientationIdentity(); }))
      .def(py::init([](const Var<Scalar> &x, const Var<Scalar> &y,
                       const Var<Scalar> &z, const Var<Scalar> &w) {
        Var<Quaternion<Scalar>> ret;
        quat_pack(x, y, z, w, ret);
        return ret;
      }))
      .def(py::init([](const Scalar &x, const Scalar &y, const Scalar &z,
                       const Scalar &w) {
        return Var<Quaternion<Scalar>>(Quaternion<Scalar>(x, y, z, w));
      }))
      .def(py::self * py::self)
      .def(py::self * Var<Vector3<Scalar>>());

  pythonizeType<Var<Matrix3<Scalar>>>(main_module, type_module, "Matrix3");

  pythonizeType<Var<Vector3<Scalar>>>(main_module, type_module, "Vector3")
      .def(py::init([](const Eigen::Vector3d &v) {
        return Var<Vector3<Scalar>>(Vector3<Scalar>(v.x(), v.y(), v.z()));
      }))
      .def(py::init(
          [](const Var<Scalar> &x, const Var<Scalar> &y, const Var<Scalar> &z) {
            Var<Vector3<Scalar>> ret;
            vec3_pack(x, y, z, ret);
            return ret;
          }))
      .def(py::init([](const Scalar &x, const Scalar &y, const Scalar &z) {
        return Var<Vector3<Scalar>>(Vector3<Scalar>(x, y, z));
      }))
      .def(py::init([](const Var<Scalar> &v) {
        Var<Vector3<Scalar>> ret;
        vec3_pack(v, v, v, ret);
        return ret;
      }))
      .def(py::init([](const Scalar &v) {
        return Var<Vector3<Scalar>>(Vector3<Scalar>(v, v, v));
      }))
      .def_property(
          "value",
          [](const Var<Vector3<Scalar>> &_this) {
            auto v = value(_this);
            py::array_t<Scalar> r(3);
            r.mutable_at(0) = v.x();
            r.mutable_at(1) = v.y();
            r.mutable_at(2) = v.z();
            return r;
          },
          [](Var<Vector3<Scalar>> &_this, const py::array_t<Scalar> &array) {
            TRACTOR_ASSERT(array.ndim() == 1);
            TRACTOR_ASSERT(array.size() == 3);
            value(_this).x() = array.at(0);
            value(_this).y() = array.at(1);
            value(_this).z() = array.at(2);
          })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * Var<Scalar>())
      .def(Var<Scalar>() * py::self);
}

TRACTOR_PYTHON_TYPED(pythonizeGeometry);

} // namespace tractor
