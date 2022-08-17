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
      .def_property(
          "value",
          [](const Var<Scalar> &_this) { return (Scalar)_this.value(); },
          [](Var<Scalar> &_this, const Scalar &v) { _this.value() = v; })
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

  m.def("debug", []() {
    static auto printStackTrace = []() {
      TRACTOR_INFO(boost::stacktrace::stacktrace());
    };
    signal(SIGSEGV, [](int sig) {
      printStackTrace();
      TRACTOR_FATAL("SEGFAULT");
      exit(-1);
    });
    signal(SIGFPE, [](int sig) {
      printStackTrace();
      TRACTOR_FATAL("SIGFPE");
      exit(-1);
    });
    std::set_terminate([]() {
      printStackTrace();
      TRACTOR_FATAL("uncaught exception");
      exit(-1);
    });
  });
  m.def("throw_runtime_error",
        [](const std::string &s) { throw std::runtime_error(s); });
  m.def("raise_segfault", []() { raise(SIGSEGV); });
  m.def("raise_fpe", []() {
    int a = 0;
    int c = 1 / a;
  });

  m.attr("__version__") = "0.0.0";
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
  profiler.def("start", [](double interval) {
    tractor::ProfilerThread::start(interval);
  });

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
