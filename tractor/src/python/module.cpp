// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/engine.h>
#include <tractor/core/log.h>
#include <tractor/core/profiler.h>
#include <tractor/engines/simple.h>

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

  py::class_<Memory, std::shared_ptr<Memory>>(m, "Memory");

  py::class_<Executable, std::shared_ptr<Executable>>(m, "Executable")
      .def("run", [](Executable &executable, std::shared_ptr<Memory> &memory) {
        Buffer buffer;
        buffer.gather(executable.parameters());
        executable.parameterize(buffer, memory);
        buffer.gather(executable.inputs());
        executable.input(buffer, memory);
        executable.execute(memory);
        executable.output(memory, buffer);
        buffer.scatter(executable.outputs());
      });

  py::class_<Engine, std::shared_ptr<Engine>>(m, "Engine")
      .def("compile",
           [](Engine &engine, const Program &program) {
             return engine.compile(program);
           })
      .def("createMemory",
           [](Engine &engine) { return engine.createMemory(); });

  py::class_<SimpleEngine, std::shared_ptr<SimpleEngine>, Engine>(
      m, "DefaultEngine")
      .def(py::init<>());

  for (auto *op : Operator::all()) {
    op->pythonize(m);
  }

  auto profiler = m.def_submodule("profiler");
  profiler.def("start", []() { tractor::ProfilerThread::start(); });
}

void initTractorPython(pybind11::module &m) {
  TRACTOR_DEBUG("building module");
  tractor::pythonizeMain(m);
}

} // namespace tractor
