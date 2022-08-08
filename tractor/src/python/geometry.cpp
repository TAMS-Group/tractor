// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/constraints.h>
#include <tractor/core/ops.h>
#include <tractor/core/var.h>
#include <tractor/geometry/fast.h>

namespace tractor {

template <class Scalar>
static void pythonizeGeometry(py::module &main_module,
                              py::module &type_module) {

  pythonizeType<Var<Twist<Scalar>>>(main_module, type_module, "Twist")
      .def(py::self + py::self);

  pythonizeType<Var<Pose<Scalar>>>(main_module, type_module, "Pose")
      .def(py::self * py::self);

  pythonizeType<Var<Quaternion<Scalar>>>(main_module, type_module, "Quaternion")
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

  pythonizeType<Var<Vector3<Scalar>>>(main_module, type_module, "Vector3")
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
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * Var<Scalar>())
      .def(Var<Scalar>() * py::self);
}

TRACTOR_PYTHON_TYPED(pythonizeGeometry);

} // namespace tractor
