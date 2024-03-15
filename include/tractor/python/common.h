// 2022-2024 Philipp Ruppel

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
#include <moveit/robot_model_loader/robot_model_loader.h>

namespace tractor {

namespace py = pybind11;

template <class Geometry>
struct PyRobotModel : RobotModel<Geometry> {
  robot_model_loader::RobotModelLoaderConstPtr moveit_loader;
  moveit::core::RobotModelConstPtr moveit_model;
  PyRobotModel(const moveit::core::RobotModelConstPtr &model,
               const robot_model_loader::RobotModelLoaderConstPtr &loader)
      : RobotModel<Geometry>(*model),
        moveit_model(model),
        moveit_loader(loader) {
    TRACTOR_DEBUG("robot model created");
  }
  ~PyRobotModel() { TRACTOR_DEBUG("robot model destroyed"); }
};

template <class T, class... Args>
using ptr_class = py::class_<T, std::shared_ptr<T>, Args...>;

class PythonRegistry {
  std::vector<std::function<void(py::module &)>> _ff;

 public:
  void add(const std::function<void(py::module &)> &f) { _ff.push_back(f); }
  void run(py::module &m);
  static const std::shared_ptr<PythonRegistry> &instance();
};

#define TRACTOR_PYTHON_STRINGIFY(name) #name

#define TRACTOR_PYTHON_GLOBAL(name)                                  \
  static int _tractor_python_global = []() {                         \
    PythonRegistry::instance()->add([](py::module &m) { name(m); }); \
    return 0;                                                        \
  }();

#define TRACTOR_PYTHON_TYPED(name)                                \
  static int _tractor_python_typed = []() {                       \
    PythonRegistry::instance()->add([](py::module m) {            \
      name<float>(m, m.attr("types_float").cast<py::module>());   \
      name<double>(m, m.attr("types_double").cast<py::module>()); \
    });                                                           \
    return 0;                                                     \
  }();

#define TRACTOR_PYTHON_TYPED_BATCH(name)                          \
  static int _tractor_python_typed = []() {                       \
    PythonRegistry::instance()->add([](py::module m) {            \
      name<float>(m, m.attr("types_float").cast<py::module>());   \
      name<double>(m, m.attr("types_double").cast<py::module>()); \
      name<tractor::Batch<float, 4>>(                             \
          m, m.attr("types_float_4").cast<py::module>());         \
      name<tractor::Batch<double, 4>>(                            \
          m, m.attr("types_double_4").cast<py::module>());        \
    });                                                           \
    return 0;                                                     \
  }();

#define TRACTOR_PYTHON_TWIST(name)                       \
  static int _tractor_python = []() {                    \
    PythonRegistry::instance()->add([](py::module m) {   \
      name<GeometryFast<Var<float>>>(                    \
          m, m.attr("types_float").cast<py::module>());  \
      name<GeometryFast<Var<double>>>(                   \
          m, m.attr("types_double").cast<py::module>()); \
    });                                                  \
    return 0;                                            \
  }();

#define TRACTOR_PYTHON_GEOMETRY(name)                           \
  static int _tractor_python_geometry = []() {                  \
    PythonRegistry::instance()->add([](py::module m) {          \
      name<GeometryFast<Var<float>>>(                           \
          m, m.attr("types_float").cast<py::module>());         \
      name<GeometryFast<Var<double>>>(                          \
          m, m.attr("types_double").cast<py::module>());        \
      name<GeometryScalar<Var<float>>>(                         \
          m, m.attr("types_float_scalar").cast<py::module>());  \
      name<GeometryScalar<Var<double>>>(                        \
          m, m.attr("types_double_scalar").cast<py::module>()); \
    });                                                         \
    return 0;                                                   \
  }();

#define TRACTOR_PYTHON_GEOMETRY_BATCH(name)                     \
  static int _tractor_python_geometry = []() {                  \
    PythonRegistry::instance()->add([](py::module m) {          \
      name<GeometryFast<Var<float>>>(                           \
          m, m.attr("types_float").cast<py::module>());         \
      name<GeometryFast<Var<double>>>(                          \
          m, m.attr("types_double").cast<py::module>());        \
      name<GeometryScalar<Var<float>>>(                         \
          m, m.attr("types_float_scalar").cast<py::module>());  \
      name<GeometryScalar<Var<double>>>(                        \
          m, m.attr("types_double_scalar").cast<py::module>()); \
      name<GeometryFast<Var<Batch<float, 4>>>>(                 \
          m, m.attr("types_float_4").cast<py::module>());       \
      name<GeometryFast<Var<Batch<double, 4>>>>(                \
          m, m.attr("types_double_4").cast<py::module>());      \
    });                                                         \
    return 0;                                                   \
  }();

template <class Type>
static auto pythonizeTypeBase(py::module &main_module, py::module &type_module,
                              const char *name) {
  main_module.def("goal", [](const Type &var) { goal(var); });
  main_module.def("goal",
                  [](const Type &var, int priority) { goal(var, priority); });
  main_module.def("constraint", [](const Type &var) { goal(var, 1); });
  return ptr_class<Type>(type_module, name)
      //.def(py::init<>())
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

template <class Type>
struct TypePythonizer {
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

}  // namespace tractor
