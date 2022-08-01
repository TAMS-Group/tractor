// TAMS Hand Synergies
// (c) 2022 Philipp Ruppel

#include <tractor/python/python.h>

#include <tractor/core/engine.h>
#include <tractor/core/profiler.h>
#include <tractor/core/solver.h>
#include <tractor/engines/simple.h>
#include <tractor/geometry/fast.h>
#include <tractor/solvers/gd.h>
#include <tractor/solvers/sq.h>
#include <tractor/tensor/newtensor.h>

namespace tractor {

namespace py = pybind11;

py::object toPython(const Any &v) {
  if (v.is<float>())
    return py::float_(v.value<float>());
  if (v.is<double>())
    return py::float_(v.value<double>());
  throw std::runtime_error("not convertible");
}

void setFromPython(Any &a, const py::object &o) {
  if (a.is<float>())
    a.value<float>() = o.cast<float>();
  else if (a.is<double>())
    a.value<double>() = o.cast<double>();
  else
    throw std::runtime_error("not convertible");
}

template <class Scalar>
void makeTypeModule(py::module &main, const char *name) {

  auto m = main.def_submodule(name);

  m.def("Scalar", []() { return Any(Var<Scalar>(0)); });
  m.def("Scalar", [](Scalar v) { return Any(Var<Scalar>(v)); });

  py::class_<LeastSquaresSolver<Scalar>, Solver>(m, "LeastSquaresSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("regularization",
                     &LeastSquaresSolver<Scalar>::_regularization)
      .def_readwrite("step_scaling", &LeastSquaresSolver<Scalar>::_step_scaling)
      .def_readwrite("max_linear_iterations",
                     &LeastSquaresSolver<Scalar>::_max_linear_iterations);

  py::class_<GradientDescentSolver<Scalar>, Solver>(m, "GradientDescentSolver")
      .def(py::init<std::shared_ptr<Engine>>());

  static auto tensor_assign = [](Tensor2<Scalar> &v,
                                 const py::array_t<Scalar> &p) {
    std::vector<size_t> ss;
    ss.resize(p.ndim());
    for (size_t i = 0; i < p.ndim(); i++) {
      ss[i] = p.shape(i);
    }
    TensorShape shape{ss};
    std::vector<Scalar> temp;
    temp.resize(shape.elementCount());
    {
      auto r = p.data();
      for (size_t i = 0; i < shape.elementCount(); i++) {
        temp[i] = *r;
        r++;
      }
    }
    v = Tensor2<Scalar>(shape, temp.data());
  };

  py::class_<Tensor2<Scalar>>(m, "Tensor")
      .def(py::init<>())
      .def(py::init([](const py::array_t<Scalar> &a) {
        Tensor2<Scalar> ret;
        tensor_assign(ret, a);
        return ret;
      }))
      .def("copy",
           [](const Tensor2<Scalar> &v) {
             Tensor2<Scalar> r = v;
             return r;
           })
      .def_property(
          "value",
          [](const Tensor2<Scalar> &v) {
            py::array_t<Scalar> ret;
            ret.resize(v.shape());
            {
              auto r = ret.mutable_data();
              for (size_t i = 0; i < v.shape().elementCount(); i++) {
                *r = v.data()[i];
                r++;
              }
            }
            return ret;
          },
          tensor_assign);

  main.def("add", [](const Tensor2<Scalar> &a, const Tensor2<Scalar> &b) {
    Tensor2<Scalar> r(a.shape());
    add(a, b, r);
    return r;
  });

  main.def("input", [](Tensor2<Scalar> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->input(var.type(), var.data(), var.data());
    }
  });
  main.def("output", [](Tensor2<Scalar> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->output(var.type(), var.data(), var.data());
    }
  });
  main.def("goal", [](Tensor2<Scalar> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->goal(var.type(), var.data());
    }
  });
}

void buildMainModule(py::module &m) {

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

  makeTypeModule<float>(m, "types_float");
  makeTypeModule<double>(m, "types_double");

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

  py::class_<Any>(m, "Var").def_property(
      "value", [](const Any &v) { return toPython(v); },
      [](Any &v, const py::object &p) { setFromPython(v, p); });

  m.def("input", [](Any &var) {
    if (auto *rec = Recorder::instance()) {
      rec->input(var.type(), var.data(), var.data());
    }
  });
  m.def("output", [](Any &var) {
    if (auto *rec = Recorder::instance()) {
      rec->output(var.type(), var.data(), var.data());
    }
  });
  m.def("goal", [](Any &var) {
    if (auto *rec = Recorder::instance()) {
      rec->goal(var.type(), var.data());
    }
  });

  {
    std::map<std::string, std::vector<const Operator *>> map;
    for (auto *op : Operator::all()) {
      if (op->isMode<compute>()) {
        map[op->label()].push_back(op);
      }
    }
    for (auto &p : map) {
      std::cout << p.first << std::endl;
      auto variants = p.second;
      m.def(p.first.c_str(), [variants](py::args args) {
        for (auto *op : variants) {
          if (args.size() > op->arguments().size()) {
            continue;
          }
          bool types_match = true;
          for (size_t i = 0; i < args.size(); i++) {
            if (args[i].cast<Any &>().type() != op->arg(i).typeInfo()) {
              types_match = false;
              break;
            }
          }
          if (!types_match) {
            continue;
          }
          std::vector<uintptr_t> argp;
          for (auto &a : args) {
            argp.push_back((uintptr_t)a.cast<Any &>().data());
          }
          std::deque<Any> ret;
          while (argp.size() < op->arguments().size()) {
            if (op->arg(argp.size()).isInput()) {
              throw std::invalid_argument(
                  "function expects more arguments than specified");
            }
            ret.emplace_back(op->arg(argp.size()).typeInfo());
            argp.push_back((uintptr_t)ret.back().data());
          }
          op->callIndirect(nullptr, argp.data());
          if (auto *rec = Recorder::instance()) {
            rec->op(op);
            for (auto &a : argp) {
              rec->push(a);
            }
          }
          if (ret.empty()) {
            return (py::object)py::none();
          } else if (ret.size() == 1) {
            return py::cast(ret[0]);
          } else {
            return py::cast(ret);
          }
        }
        throw std::invalid_argument("no matching function overload");
      });
    }
  }

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
                             });
  m.def("record", [](const std::function<void()> &f) {
    return std::make_shared<Program>(f);
  });

  m.def("test", []() { std::cout << "test" << std::endl; });

  auto profiler = m.def_submodule("profiler");
  profiler.def("start", []() { static ProfilerThread p; });
}

} // namespace tractor

PYBIND11_MODULE(tractor, m) { tractor::buildMainModule(m); }
