// TAMS Hand Synergies
// (c) 2022 Philipp Ruppel

#include <tractor/python/module.h>

#include <tractor/core/constraints.h>
#include <tractor/core/engine.h>
#include <tractor/core/profiler.h>
#include <tractor/core/solver.h>
#include <tractor/engines/simple.h>
#include <tractor/geometry/fast.h>
#include <tractor/solvers/gd.h>
#include <tractor/solvers/sq.h>
#include <tractor/tensor/newtensor.h>

namespace tractor {

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

  py::class_<Var<Scalar>>(m, "Scalar")
      .def(py::init<>())
      .def(py::init<Scalar>())
      .def_property(
          "value", [](const Var<Scalar> &v) { return (Scalar)v.value(); },
          [](Var<Scalar> &v, const Scalar &p) { v.value() = p; })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def("__repr__", [](const Var<Scalar> &v) {
        std::stringstream ss;
        ss << "Scalar(" << v.value() << ")";
        return ss.str();
      });
  m.def("parameter", [](Var<Scalar> &var) { tractor::parameter(var); });
  m.def("variable", [](Var<Scalar> &var) { tractor::variable(var); });
  m.def("output", [](Var<Scalar> &var) { tractor::output(var); });
  m.def("goal", [](Var<Scalar> &var) { tractor::goal(var); });

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

  main.def("variable", [](Tensor2<Scalar> &var) {
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

  for (auto *op : Operator::all()) {
    op->pythonize(m);
  }

  // {
  //   std::map<std::string, std::vector<const Operator *>> map;
  //   for (auto *op : Operator::all()) {
  //     if (op->isMode<compute>()) {
  //       map[op->label()].push_back(op);
  //     }
  //   }
  //   for (auto &p : map) {
  //     std::cout << p.first << std::endl;
  //     auto variants = p.second;
  //     auto matchVariant = [](const Operator *op, const py::args &args) {
  //       if (args.size() > op->arguments().size()) {
  //         return false;
  //       }
  //       for (size_t i = 0; i < op->argumentCount(); i++) {
  //         if (op->arg(i).isInput()) {
  //           if (i >= args.size()) {
  //             return false;
  //           }
  //           try {
  //             if (args[i].cast<Any &>().type() != op->arg(i).typeInfo()) {
  //               return false;
  //             }
  //           } catch (const py::cast_error &e) {
  //           }
  //         }
  //       }
  //       return true;
  //     };
  //     auto findVariant = [variants, matchVariant](const py::args &args) {
  //       const Operator *match = nullptr;
  //       for (const Operator *op : variants) {
  //         if (matchVariant(op, args)) {
  //           if (match == nullptr) {
  //             match = op;
  //           } else {
  //             throw std::invalid_argument("ambiguous call " + match->name() +
  //                                         " " + op->name());
  //           }
  //         }
  //       }
  //       if (match) {
  //         return match;
  //       }
  //       throw std::invalid_argument("no matching function overload");
  //     };
  //     auto wrapper = [variants,
  //                     findVariant](const py::args &py_args) -> py::object {
  //       const Operator *op = findVariant(py_args);
  //       // std::cout << "----- call " << op->name() << std::endl;
  //       std::vector<uintptr_t> arg_p;
  //       std::deque<Any> any_args;
  //       for (size_t i = 0; i < py_args.size(); i++) {
  //         try {
  //           arg_p.push_back((uintptr_t)py_args[i].cast<Any &>().data());
  //         } catch (const py::cast_error &) {
  //           any_args.emplace_back(op->arg(i).typeInfo());
  //           arg_p.push_back((uintptr_t)any_args.back().data());
  //         }
  //       }
  //       std::deque<Any> ret;
  //       while (arg_p.size() < op->arguments().size()) {
  //         if (op->arg(arg_p.size()).isInput()) {
  //           throw std::invalid_argument(
  //               "function expects more arguments than specified");
  //         }
  //         ret.emplace_back(op->arg(arg_p.size()).typeInfo());
  //         arg_p.push_back((uintptr_t)ret.back().data());
  //       }
  //       op->callIndirect(nullptr, arg_p.data());
  //       if (auto *rec = Recorder::instance()) {
  //         rec->op(op);
  //         for (auto &a : arg_p) {
  //           rec->push(a);
  //         }
  //       }
  //       // std::cout << "----- ready " << op->name() << std::endl;
  //       if (ret.empty()) {
  //         return (py::object)py::none();
  //       } else if (ret.size() == 1) {
  //         return py::cast(ret[0]);
  //       } else {
  //         return py::cast(ret);
  //       }
  //       throw std::invalid_argument("no matching function overload");
  //     };
  //     wrappers[p.first] = wrapper;
  //     m.def(p.first.c_str(), wrapper);
  //   }
  // }

  // m.def("test", []() {
  //   // std::cout << "test" << std::endl;
  //   // m.def("bla", []() { std::cout << "bla" << std::endl; });
  //   return py::detail::get_type_handle(typeid(Var<double>), true);
  // });
  //
  // m.def("test2", [](const py::object &o) {
  //   // return py::detail::get_type_handle(typeid(Var<double>), true) ==
  //   //  py::type::of(o)
  //   //       o.get_type();
  //   auto t = py::detail::get_type_handle(typeid(Var<double>), true);
  //   return o.is(t);
  // });
  //
  // m.def("test3", [](const py::object &o) {
  //   return py::detail::get_type_handle(typeid(Var<double>), true) ==
  //          o.get_type();
  // });

  auto profiler = m.def_submodule("profiler");
  profiler.def("start", []() { static ProfilerThread p; });
}

} // namespace tractor

PYBIND11_MODULE(tractor, m) {
  std::cout << "building module" << std::endl;
  tractor::buildMainModule(m);
}
