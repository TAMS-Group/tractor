// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/constraints.h>
#include <tractor/core/engine.h>
#include <tractor/core/gradients.h>
#include <tractor/core/log.h>
#include <tractor/core/profiler.h>
#include <tractor/core/solver.h>
#include <tractor/engines/simple.h>
#include <tractor/geometry/fast.h>
#include <tractor/neural/ops.h>
#include <tractor/tensor/ops.h>

namespace tractor {

static void pythonizeMain(py::module &m) {

  class Log {};
  py::class_<Log>(m, "logger")
      .def_property_static(
          "verbosity", [](py::object) { return getLogVerbosity(); },
          [](py::object, int v) { setLogVerbosity(v); });

  py::class_<Solver>(m, "Solver")
      .def("compile", [](Solver &solver,
                         const Program &program) { solver.compile(program); })
      .def("parameterize", [](Solver &solver) { solver.parameterize(); })
      .def("solve",
           [](Solver &solver) {
             solver.parameterize();
             solver.gather();
             solver.solve();
             solver.scatter();
           })
      .def("gather", &Solver::gather)
      .def("scatter", &Solver::scatter)
      .def("step", &Solver::step)
      .def_property("tolerance", &Solver::tolerance, &Solver::setTolerance)
      .def_property(
          "timeout", [](const Solver &solver) { return solver.timeout(); },
          [](Solver &solver, const double &v) { solver.setTimeout(v, false); });

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

  py::class_<PyInstructionList>(m, "InstructionList")
      .def("__iter__",
           [](const PyInstructionList &l) {
             return py::make_iterator(l.begin(), l.end());
           })
      .def("__repr__", [](const PyInstructionList &v) {
        std::stringstream ret;
        ret << "[";
        bool first = true;
        for (const auto &inst : v) {
          if (!first)
            ret << ", ";
          ret << inst.str();
          first = false;
        }
        ret << "]";
        return ret.str();
      });

  py::class_<PyInstruction>(m, "Instruction")
      .def("__repr__", [](const PyInstruction &v) { return v.str(); });

  py::class_<Program::Input>(m, "Input")
      .def("__repr__", [](const Program::Input &v) {
        return std::string() + "input(" + v.typeInfo().name() + "," +
               std::to_string(v.address()) + ")";
      });

  py::class_<Program::Output>(m, "Output")
      .def("__repr__", [](const Program::Output &v) {
        return std::string() + "output(" + v.typeInfo().name() + "," +
               std::to_string(v.address()) + ")";
      });

  py::class_<Program::Goal>(m, "Goal");

  py::class_<Program::Parameter>(m, "Parameter");

  py::class_<Program::Constant>(m, "Constant")
      .def("__repr__", [](const Program::Constant &v) {
        return std::string() + "const(" + v.typeInfo().name() + "," +
               std::to_string(v.address()) + ")";
      });

  py::class_<Program, std::shared_ptr<Program>>(m, "Program")
      .def_property_readonly("instructions",
                             [](const std::shared_ptr<Program> &program) {
                               return PyInstructionList(program);
                             })
      .def_property_readonly("inputs",
                             [](const std::shared_ptr<Program> &program) {
                               std::vector<Program::Input> ret;
                               for (auto &v : program->inputs())
                                 ret.push_back(v);
                               return ret;
                             })
      .def_property_readonly("outputs",
                             [](const std::shared_ptr<Program> &program) {
                               std::vector<Program::Output> ret;
                               for (auto &v : program->outputs())
                                 ret.push_back(v);
                               return ret;
                             })
      .def_property_readonly("constants",
                             [](const std::shared_ptr<Program> &program) {
                               std::vector<Program::Constant> ret;
                               for (auto &v : program->constants())
                                 ret.push_back(v);
                               return ret;
                             })
      .def("__repr__", [](const Program &v) {
        std::stringstream ss;
        ss << v;
        std::string s = ss.str();
        while (!s.empty() && std::isspace(s.back())) {
          s.pop_back();
        }
        return s;
      });
  m.def("record", [](const std::function<void()> &f) {
    return std::make_shared<Program>(f);
  });

  struct PyDerivatives {
    Program prepare, forward, reverse, hessian, accumulate;
  };
  py::class_<PyDerivatives>(m, "Derivatives")
      .def_readonly("prepare", &PyDerivatives::prepare)
      .def_readonly("forward", &PyDerivatives::forward)
      .def_readonly("reverse", &PyDerivatives::reverse)
      .def_readonly("hessian", &PyDerivatives::hessian)
      .def_readonly("accumulate", &PyDerivatives::accumulate);
  m.def("derive", [](const Program &src) {
    PyDerivatives ret;
    buildGradients(src, ret.prepare, &ret.forward, &ret.reverse, &ret.hessian,
                   &ret.accumulate);
    return ret;
  });

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
