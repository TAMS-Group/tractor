// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/log.h>
#include <tractor/core/profiler.h>

#include <pybind11/eval.h>

#include <boost/stacktrace.hpp>

#include <signal.h>

namespace tractor {

template <class Scalar>
static void pythonizeGeometryScalar(py::module main_module,
                                    py::module type_module) {
  pythonizeType<Var<Scalar>>(main_module, type_module, "Scalar")
      .def(py::init<Scalar>())
      .def_property_readonly(
          "value", [](const Var<Scalar> &v) { return (Scalar)v.value(); })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self *= py::self)
      .def(py::self /= py::self);
}

TRACTOR_PYTHON_TYPED(pythonizeGeometryScalar);

static void pythonizeMain(py::module &m) {

  // py::options options;
  // options.enable_function_signatures();

  // auto mod_scalar = m.def_submodule("scalar");
  // pythonizeGeometryScalar<float>(m, mod_scalar.def_submodule("float"));
  // pythonizeGeometryScalar<double>(m, mod_scalar.def_submodule("double"));

  static auto printStackTrace = []() {
    TRACTOR_INFO(boost::stacktrace::stacktrace());
    TRACTOR_FATAL("fatal error, exiting");
    exit(-1);
  };
  signal(SIGSEGV, [](int sig) {
    TRACTOR_FATAL("SEGFAULT");
    printStackTrace();
  });
  std::set_terminate(printStackTrace);

  m.attr("__version__") = "dev";
  // m.doc();

  m.def_submodule("types_float");
  m.def_submodule("types_double");

  m.def_submodule("types_float_twist");
  m.def_submodule("types_double_twist");

  m.def_submodule("types_float_scalar");
  m.def_submodule("types_double_scalar");

  struct PythonObjectHolder {
    pybind11::object object;
    PythonObjectHolder(const pybind11::object &object) : object(object) {}
    virtual ~PythonObjectHolder() {}
  };

  m.def("variable", [](const py::object &o) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(std::make_shared<PythonObjectHolder>(o));
    }
    o.attr("_internal_make_variable")();
  });

  m.def("parameter", [](const py::object &o) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(std::make_shared<PythonObjectHolder>(o));
    }
    o.attr("_internal_make_parameter")();
  });

  m.def("output", [](const py::object &o) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(std::make_shared<PythonObjectHolder>(o));
    }
    o.attr("_internal_make_output")();
  });

  class Log {};
  py::class_<Log>(m, "logger")
      .def_property_static(
          "verbosity", [](py::object) { return getLogVerbosity(); },
          [](py::object, int v) { setLogVerbosity(v); });

  PythonRegistry::instance()->run(m);

  auto profiler = m.def_submodule("profiler");
  profiler.def("start", []() { tractor::ProfilerThread::start(); });

  // m.attr("types_double_twist").attr("Scalar") =
  //     m.attr("types_double").attr("Scalar");
  // m.attr("types_double_scalar").attr("Scalar") =
  //     m.attr("types_double").attr("Scalar");

  for (auto *op : Operator::all()) {
    op->pythonize(m);
  }
}

void initTractorPython(pybind11::module &m) {
  TRACTOR_DEBUG("building module");
  tractor::pythonizeMain(m);
}

} // namespace tractor
