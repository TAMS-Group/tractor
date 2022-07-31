// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>
#include <tractor/core/type.h>
#include <tractor/core/var.h>

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace tractor {

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

class PyVar {
  std::type_index _type = typeid(void);
  size_t _alignment = 0;
  std::vector<uint8_t> _data;
  static void recordMove(const std::type_index &type, const void *from,
                         void *to) {
    const Operator *move_op = Operator::tryFind(
        OpMode(typeid(compute *)), OpType(typeid(op_move *)), {type});
    // std::cout << "pyvar move " << move_op << " " << from << " " << to
    //           << std::endl;
    if (auto *rec = Recorder::instance()) {
      rec->op(move_op);
      rec->push((uintptr_t)from);
      rec->push((uintptr_t)to);
    }
  }

public:
  explicit PyVar(const TypeInfo &type) {
    _type = type.type();
    _alignment = type.alignment();
    _data.resize(type.size(), 0);
  }
  PyVar(const PyVar &other) {
    _type = other._type;
    _alignment = other._alignment;
    _data = other._data;
    recordMove(_type, other._data.data(), _data.data());
  }
  PyVar &operator=(const PyVar &other) {
    _type = other._type;
    _alignment = other._alignment;
    _data = other._data;
    recordMove(_type, other._data.data(), _data.data());
    return *this;
  }
  template <class T> explicit PyVar(const Var<T> &v) {
    _type = typeid(T);
    _alignment = std::alignment_of<T>::value;
    _data.resize(sizeof(T));
    std::memcpy(_data.data(), &v.value(), sizeof(T));
    recordMove(typeid(T), &v.value(), _data.data());
  }
  TypeInfo type() const { return TypeInfo(_data.size(), _type, _alignment); }
  const void *data() const { return _data.data(); }
  void *data() { return _data.data(); }
  template <class T> const T &value() const { return *(const T *)data(); }
  template <class T> T &value() { return *(T *)data(); }
  template <class T> bool is() const { return _type == typeid(T); }
  pybind11::object toPython() const {
    if (is<float>())
      return pybind11::float_(value<float>());
    if (is<double>())
      return pybind11::float_(value<double>());
    throw std::runtime_error("not convertible");
  }
  void setFromPython(const pybind11::object &v) {
    if (is<float>())
      value<float>() = v.cast<float>();
    else if (is<double>())
      value<double>() = v.cast<double>();
    else
      throw std::runtime_error("not convertible");
  }
};

} // namespace tractor
