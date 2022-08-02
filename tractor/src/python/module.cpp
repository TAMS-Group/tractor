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

template <class Type>
static auto pythonizeType(py::module &main_module, py::module &type_module,
                          const char *name) {

  auto t = py::class_<Type>(type_module, name);
  t.def(py::init<>());
  t.def("__repr__", [name](const Type &v) {
    std::stringstream ss;
    ss << value(v);
    return ss.str();
  });

  main_module.def("parameter", [](Type &var) { parameter(var); });
  main_module.def("variable", [](Type &var) { variable(var); });
  main_module.def("output", [](Type &var) { output(var); });
  main_module.def("goal", [](Type &var) { goal(var); });

  return t;
}

template <class Scalar>
static void pythonizeScalar(py::module &main_module, py::module &type_module) {
  pythonizeType<Var<Scalar>>(main_module, type_module, "Scalar")
      .def(py::init<Scalar>())
      .def_property(
          "value", [](const Var<Scalar> &v) { return (Scalar)v.value(); },
          [](Var<Scalar> &v, const Scalar &p) { v.value() = p; })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self);
}

template <class Scalar>
static void pythonizeGeometry(py::module &main_module,
                              py::module &type_module) {

  pythonizeType<Var<Twist<Scalar>>>(main_module, type_module, "Twist")
      .def(py::self + py::self);

  pythonizeType<Var<Pose<Scalar>>>(main_module, type_module, "Pose")
      .def(py::self * py::self);

  pythonizeType<Var<Quaternion<Scalar>>>(main_module, type_module, "Quaternion")
      .def(py::init([](const Var<Scalar> &x, const Var<Scalar> &y,
                       const Var<Scalar> &z, const Var<Scalar> &w) {
        Var<Quaternion<Scalar>> ret;
        quat_pack(x, y, z, w, ret);
        return ret;
      }))
      .def(py::init([](const Scalar &x, const Scalar &y, const Scalar &z,
                       const Scalar &w) {
        return Var<Quaternion<Scalar>>(Quaternion<Scalar>(x, y, z, w));
      }))
      .def(py::self * py::self)
      .def(py::self * Var<Vector3<Scalar>>());

  pythonizeType<Var<Vector3<Scalar>>>(main_module, type_module, "Vector3")
      .def(py::init(
          [](const Var<Scalar> &x, const Var<Scalar> &y, const Var<Scalar> &z) {
            Var<Vector3<Scalar>> ret;
            vec3_pack(x, y, z, ret);
            return ret;
          }))
      .def(py::init([](const Scalar &x, const Scalar &y, const Scalar &z) {
        return Var<Vector3<Scalar>>(Vector3<Scalar>(x, y, z));
      }))
      .def(py::init([](const Var<Scalar> &v) {
        Var<Vector3<Scalar>> ret;
        vec3_pack(v, v, v, ret);
        return ret;
      }))
      .def(py::init([](const Scalar &v) {
        return Var<Vector3<Scalar>>(Vector3<Scalar>(v, v, v));
      }))
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * Var<Scalar>())
      .def(Var<Scalar>() * py::self);
}

template <class Scalar>
static void pythonizeTensor(py::module &main_module, py::module &type_module) {

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

  py::class_<Tensor2<Scalar>>(type_module, "Tensor")
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
      .def_property_readonly("shape",
                             [](const Tensor2<Scalar> &t) {
                               const auto &s = t.shape();
                               py::tuple ret = py::tuple(s.dimensions());
                               for (size_t i = 0; i < s.dimensions(); i++) {
                                 ret[i] = s[i];
                               }
                               return ret;
                             })
      .def_property_readonly(
          "dimensions",
          [](const Tensor2<Scalar> &t) { return t.shape().dimensions(); })
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
          tensor_assign)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self);

  // main_module.def("add",
  //                 [](const Tensor2<Scalar> &a, const Tensor2<Scalar> &b) {
  //                   Tensor2<Scalar> r(a.shape());
  //                   add(a, b, r);
  //                   return r;
  //                 });

  main_module.def("variable", [](Tensor2<Scalar> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->input(var.type(), var.data(), var.data());
    }
  });

  main_module.def("output", [](Tensor2<Scalar> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->output(var.type(), var.data(), var.data());
    }
  });

  main_module.def("goal", [](Tensor2<Scalar> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->goal(var.type(), var.data());
    }
  });
}

template <class Scalar>
static void pythonizeSolvers(py::module &main_module, py::module &type_module) {
  py::class_<LeastSquaresSolver<Scalar>, Solver>(type_module,
                                                 "LeastSquaresSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("regularization",
                     &LeastSquaresSolver<Scalar>::_regularization)
      .def_readwrite("step_scaling", &LeastSquaresSolver<Scalar>::_step_scaling)
      .def_readwrite("max_linear_iterations",
                     &LeastSquaresSolver<Scalar>::_max_linear_iterations);

  py::class_<GradientDescentSolver<Scalar>, Solver>(type_module,
                                                    "GradientDescentSolver")
      .def(py::init<std::shared_ptr<Engine>>());
}

template <class Scalar>
static void pythonizeTemplates(py::module &main_module, const char *name) {
  auto type_module = main_module.def_submodule(name);
  pythonizeScalar<Scalar>(main_module, type_module);
  pythonizeGeometry<Scalar>(main_module, type_module);
  pythonizeTensor<Scalar>(main_module, type_module);
  pythonizeSolvers<Scalar>(main_module, type_module);
}

template <class... Args> static void checkAllTensor(const Tensor2<Args> &...) {}

static void pythonizeMain(py::module &m) {

  // Tensor2<double> a(TensorShape(2, 3));
  // Tensor2<double> b(TensorShape(3, 3));
  // std::cout << sub(a, b).shape() << std::endl;
  // tanh(a);
  // tanh(1.0);
  // zero(a);
  // exit(0);

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

  pythonizeTemplates<float>(m, "types_float");
  pythonizeTemplates<double>(m, "types_double");

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

  auto profiler = m.def_submodule("profiler");
  profiler.def("start", []() { static ProfilerThread p; });
}

void initTractorPython(pybind11::module &m) {
  std::cout << "building module" << std::endl;
  tractor::pythonizeMain(m);
}

} // namespace tractor

// PYBIND11_MODULE(tractor, m) {
//   std::cout << "building module" << std::endl;
//   tractor::pythonizeMain(m);
// }
