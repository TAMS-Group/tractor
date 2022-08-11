// (c) 2022 Philipp Ruppel

#pragma once

#include <tractor/core/any.h>
#include <tractor/core/constraints.h>
#include <tractor/core/operator.h>
#include <tractor/core/ops.h>
#include <tractor/core/type.h>
#include <tractor/core/var.h>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace tractor {

namespace py = pybind11;

template <class T, class... Args>
using ptr_class = py::class_<T, std::shared_ptr<T>, Args...>;
// typedef py::class_<T, std::shared_ptr<T>, Args...> ptr_class;

class PythonRegistry {
  std::vector<std::function<void(py::module &)>> _ff;

public:
  void add(const std::function<void(py::module &)> &f) { _ff.push_back(f); }
  void run(py::module &m);
  static const std::shared_ptr<PythonRegistry> &instance();
};

#define TRACTOR_PYTHON_STRINGIFY(name) #name

#define TRACTOR_PYTHON_GLOBAL(name)                                            \
  static int _tractor_python_global = []() {                                   \
    PythonRegistry::instance()->add([](py::module &m) { name(m); });           \
    return 0;                                                                  \
  }();

#define TRACTOR_PYTHON_TYPED(name)                                             \
  static int _tractor_python_typed = []() {                                    \
    auto reg = PythonRegistry::instance();                                     \
    reg->add([](py::module m) {                                                \
      auto t = m.attr("types_float").cast<py::module>();                       \
      name<float>(m, t);                                                       \
    });                                                                        \
    reg->add([](py::module m) {                                                \
      auto t = m.attr("types_double").cast<py::module>();                      \
      name<double>(m, t);                                                      \
    });                                                                        \
    return 0;                                                                  \
  }();

template <class Type>
static auto pythonizeTypeBase(py::module &main_module, py::module &type_module,
                              const char *name) {
  // main_module.def("goal", [](const std::shared_ptr<Type> &var) { goal(*var);
  // });
  main_module.def("goal", [](const Type &var) { goal(var); });
  return ptr_class<Type>(type_module, name)
      .def(py::init<>())
      .def("__repr__",
           [name](const Type &v) {
             std::stringstream ss;
             ss << value(v);
             return ss.str();
           })
      .def("_internal_make_variable", [](Type &_this) { variable(_this); })
      .def("_internal_make_parameter", [](Type &_this) { parameter(_this); })
      .def("_internal_make_output", [](Type &_this) { output(_this); });
}

template <class Type> struct TypePythonizer {
  static auto pythonize(py::module &main_module, py::module &type_module,
                        const char *name) {
    return pythonizeTypeBase<Type>(main_module, type_module, name);
  }
};

// template <class Type> struct TypePythonizer<Var<Type>> {
//   static auto pythonize(py::module &main_module, py::module &type_module,
//                         const char *name) {
//     return pythonizeTypeBase<Var<Type>>(main_module, type_module, name)
//         .def_property(
//             "value", [](const Var<Type> &v) { return (Type)v.value(); },
//             [](Var<Type> &v, const Type &p) { v.value() = p; });
//   }
// };

template <class Type>
static auto pythonizeType(py::module &main_module, py::module &type_module,
                          const char *name) {
  return TypePythonizer<Type>::pythonize(main_module, type_module, name);
}

} // namespace tractor
