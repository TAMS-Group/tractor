// 2022-2024 Philipp Ruppel

#include <tractor/core/gradients.h>
#include <tractor/core/program.h>
#include <tractor/python/common.h>

namespace tractor {

static void pythonizeProgram(py::module &m) {
  class PyInstruction {
    std::shared_ptr<Program> _program;
    Program::InstructionIterator<Program::Instruction> _iterator;

   public:
    PyInstruction(
        const std::shared_ptr<Program> &program,
        const Program::InstructionIterator<Program::Instruction> &iterator)
        : _program(program), _iterator(iterator) {}
    const Operator &op() const { return *(*_iterator).op(); }
    const Program::Instruction &inst() const { return *_iterator; }
    std::string str() const {
      std::string ret = op().name();
      ret += "(";
      for (size_t i = 0; i < inst().argumentCount(); i++) {
        if (i > 0) ret += ",";
        ret += std::to_string(inst().arg(i));
      }
      ret += ")";
      return ret;
    }
  };

  class PyInstructionIterator {
    std::shared_ptr<Program> _program;
    Program::InstructionIterator<Program::Instruction> _iterator;

   public:
    PyInstructionIterator(
        const std::shared_ptr<Program> &program,
        const Program::InstructionIterator<Program::Instruction> &iterator)
        : _program(program), _iterator(iterator) {}
    PyInstruction operator*() { return PyInstruction(_program, _iterator); }
    PyInstructionIterator &operator++() {
      ++_iterator;
      return *this;
    }
    bool operator==(const PyInstructionIterator &other) const {
      return _iterator == other._iterator;
    }
    bool operator!=(const PyInstructionIterator &other) const {
      return _iterator != other._iterator;
    }
  };

  class PyInstructionList {
    std::shared_ptr<Program> _program;
    ArrayRef<Program::Instruction,
             Program::InstructionIterator<Program::Instruction>>
        _instructions;

   public:
    PyInstructionList(const std::shared_ptr<Program> &program)
        : _program(program), _instructions(program->instructions()) {}
    PyInstructionIterator begin() const {
      return PyInstructionIterator(_program, _instructions.begin());
    }
    PyInstructionIterator end() const {
      return PyInstructionIterator(_program, _instructions.end());
    }
  };

  py::class_<PyInstruction>(m, "Instruction")
      .def("__repr__", [](const PyInstruction &v) { return v.str(); });

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
          if (!first) ret << ", ";
          ret << inst.str();
          first = false;
        }
        ret << "]";
        return ret.str();
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

  m.def("record", [](const std::function<void()> &f, bool simplify) {
    return std::make_shared<Program>(f, simplify);
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
}

TRACTOR_PYTHON_GLOBAL(pythonizeProgram);

}  // namespace tractor
