// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/log.h>
#include <tractor/core/profiler.h>

namespace tractor {

static void pythonizeMain(py::module &m) {

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

  // m.def(
  //     "test",
  //     [](const py::object &param) {
  //
  //     },
  //     py::arg("a") = 5);

  class Log {};
  py::class_<Log>(m, "logger")
      .def_property_static(
          "verbosity", [](py::object) { return getLogVerbosity(); },
          [](py::object, int v) { setLogVerbosity(v); });

  m.def_submodule("types_float");
  m.def_submodule("types_double");

  PythonRegistry::instance()->run(m);

  auto profiler = m.def_submodule("profiler");
  profiler.def("start", []() { tractor::ProfilerThread::start(); });
}

void initTractorPython(pybind11::module &m) {
  TRACTOR_DEBUG("building module");
  tractor::pythonizeMain(m);
}

} // namespace tractor
