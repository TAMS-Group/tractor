// (c) 2022 Philipp Ruppel

#pragma once

#include <tractor/core/any.h>
#include <tractor/core/operator.h>
#include <tractor/core/type.h>
#include <tractor/core/var.h>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace tractor {

namespace py = pybind11;

#define TRACTOR_PYTHONIZE_ENUM(m, Name)                                        \
  {                                                                            \
    auto e = py::enum_<ActivationType>(m, "ActivationType");                   \
    for (auto &p : enumerateEachActivationType()) {                            \
      e.value(p.second.c_str(), p.first);                                      \
    }                                                                          \
    e.def(py::init(                                                            \
        [](const std::string &s) { return parseActivationType(s); }));         \
    py::implicitly_convertible<std::string, ActivationType>();                 \
  }

template <class Type>
static auto pythonizeType(py::module &main_module, py::module &type_module,
                          const char *name) {

  auto t = py::class_<Type, std::shared_ptr<Type>>(type_module, name);
  t.def(py::init<>());
  t.def("__repr__", [name](const Type &v) {
    std::stringstream ss;
    ss << value(v);
    return ss.str();
  });

  main_module.def("parameter", [](const std::shared_ptr<Type> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(var);
    }
    parameter(*var);
  });

  main_module.def("variable", [](const std::shared_ptr<Type> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(var);
    }
    variable(*var);
  });

  main_module.def("output", [](const std::shared_ptr<Type> &var) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(var);
    }
    output(*var);
  });

  main_module.def("goal", [](const std::shared_ptr<Type> &var) { goal(*var); });

  return t;
}

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
      if (i > 0)
        ret += ",";
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

} // namespace tractor
