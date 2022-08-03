// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/var.h>

namespace tractor {

template <class Scalar>
static void pythonizeScalar(py::module &main_module, py::module &type_module) {

  pythonizeType<Var<Scalar>>(main_module, type_module, "Scalar")
      .def(py::init<Scalar>())
      .def_property(
          "value", [](const Var<Scalar> &v) { return (Scalar)v.value(); },
          [](Var<Scalar> &v, const Scalar &p) { v.value() = p; })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self);
}

} // namespace tractor
