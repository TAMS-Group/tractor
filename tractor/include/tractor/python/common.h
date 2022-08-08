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
static auto pythonizeType(py::module &main_module, py::module &type_module,
                          const char *name) {

  auto t =
      // py::class_<Type>(type_module, name)
      ptr_class<Type>(type_module, name)
          // py::class_<Type, std::unique_ptr<Type>>(type_module, name)
          .def(py::init<>())
          .def("__repr__",
               [name](const Type &v) {
                 std::stringstream ss;
                 ss << value(v);
                 return ss.str();
               })
          .def("_internal_make_variable", [](Type &_this) { variable(_this); })
          .def("_internal_make_parameter",
               [](Type &_this) { parameter(_this); })
          .def("_internal_make_output", [](Type &_this) { output(_this); });

  // main_module.def("parameter", [](const std::shared_ptr<Type> &var) {
  //   if (auto *rec = Recorder::instance()) {
  //     rec->reference(var);
  //   }
  //   parameter(*var);
  // });
  // main_module.def("variable", [](const std::shared_ptr<Type> &var) {
  //   if (auto *rec = Recorder::instance()) {
  //     rec->reference(var);
  //   }
  //   variable(*var);
  // });
  // main_module.def("output", [](const std::shared_ptr<Type> &var) {
  //   if (auto *rec = Recorder::instance()) {
  //     rec->reference(var);
  //   }
  //   output(*var);
  // });

  main_module.def("goal", [](const std::shared_ptr<Type> &var) { goal(*var); });

  return t;
}

} // namespace tractor
