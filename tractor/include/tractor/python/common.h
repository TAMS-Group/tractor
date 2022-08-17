// (c) 2022 Philipp Ruppel

#pragma once

#include <tractor/core/any.h>
#include <tractor/core/constraints.h>
#include <tractor/core/operator.h>
#include <tractor/core/ops.h>
#include <tractor/core/type.h>
#include <tractor/core/var.h>
#include <tractor/robot/robot.h>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <moveit/robot_model/robot_model.h>

namespace tractor {

namespace py = pybind11;

template <class Geometry> struct PyRobotModel : RobotModel<Geometry> {
  moveit::core::RobotModelConstPtr moveit_model;
  PyRobotModel(const moveit::core::RobotModelConstPtr &m)
      : RobotModel<Geometry>(*m), moveit_model(m) {
    TRACTOR_DEBUG("robot model created");
  }
  ~PyRobotModel() { TRACTOR_DEBUG("robot model destroyed"); }
};

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

// #define TRACTOR_PYTHON_TYPED_SCALAR(name)                                      \
//   static int _tractor_python_typed = []() {                                    \
//     PythonRegistry::instance()->add([](py::module m) {                         \
//       name<float>(m, m.attr("types_float").cast<py::module>());                \
//       name<double>(m, m.attr("types_double").cast<py::module>());              \
//     });                                                                        \
//     return 0;                                                                  \
//   }();

#define TRACTOR_PYTHON_TYPED(name)                                             \
  static int _tractor_python_typed = []() {                                    \
    PythonRegistry::instance()->add([](py::module m) {                         \
      name<float>(m, m.attr("types_float").cast<py::module>());                \
      name<double>(m, m.attr("types_double").cast<py::module>());              \
    });                                                                        \
    return 0;                                                                  \
  }();

#define TRACTOR_PYTHON_TWIST(name)                                             \
  static int _tractor_python_twist = []() {                                    \
    PythonRegistry::instance()->add([](py::module m) {                         \
      name<GeometryFast<Var<float>>>(                                          \
          m, m.attr("types_float_twist").cast<py::module>());                  \
      name<GeometryFast<Var<double>>>(                                         \
          m, m.attr("types_double_twist").cast<py::module>());                 \
    });                                                                        \
    return 0;                                                                  \
  }();

// name<GeometryFast<Var<Batch<float, 4>>>>(                                \
//     m, m.attr("types_float_twist_4").cast<py::module>());                \
// name<GeometryFast<Var<Batch<double, 4>>>>(                               \
//     m, m.attr("types_double_twist_4").cast<py::module>());               \

#define TRACTOR_PYTHON_GEOMETRY(name)                                          \
  static int _tractor_python_geometry = []() {                                 \
    PythonRegistry::instance()->add([](py::module m) {                         \
      name<GeometryFast<Var<float>>>(                                          \
          m, m.attr("types_float_twist").cast<py::module>());                  \
      name<GeometryFast<Var<double>>>(                                         \
          m, m.attr("types_double_twist").cast<py::module>());                 \
      name<GeometryScalar<Var<float>>>(                                        \
          m, m.attr("types_float_scalar").cast<py::module>());                 \
      name<GeometryScalar<Var<double>>>(                                       \
          m, m.attr("types_double_scalar").cast<py::module>());                \
    });                                                                        \
    return 0;                                                                  \
  }();

template <class Type>
static auto pythonizeTypeBase(py::module &main_module, py::module &type_module,
                              const char *name) {
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

template <class Type>
static auto pythonizeType(py::module &main_module, py::module &type_module,
                          const char *name) {
  return TypePythonizer<Type>::pythonize(main_module, type_module, name);
}

} // namespace tractor
