// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/allocator.h>
#include <tractor/core/program.h>

#include <cstring>
#include <deque>
#include <unordered_set>

namespace tractor {

template <class T>
class Var;

class Recorder {
  Program *_program = nullptr;
  std::vector<Program::Instruction> _instructions;
  std::vector<Program::Input> _inputs;
  std::vector<Program::Output> _outputs;
  std::vector<Program::Parameter> _parameters;
  std::vector<Program::Goal> _goals;
  std::vector<Program::Constant> _constants;
  std::vector<uint8_t> _const_data;
  std::vector<uint8_t> _bound_data;
  std::unordered_set<std::shared_ptr<const void>> _references;
  Allocator _alloc;
  const Operator *_pending_op = nullptr;
  std::vector<const void *> _pending_args;
  std::unordered_set<const void *> _known_addresses;

  template <class T>
  void outputImpl(const Var<T> *p, bool bind) {
    uintptr_t temp =
        (((uintptr_t)_alloc.alloc(TypeInfo::get<T>())) | 0x8000000000000000ul);
    move(&p->value(), (T *)temp);
    _outputs.emplace_back(TypeInfo::get<T>(), temp, 0,
                          bind ? (uintptr_t)&p->value() : 0);
  }

  void finish(Program &program);

 protected:
  Recorder(Program *program);

 public:
  static Recorder *instance();

  ~Recorder();

  Recorder(const Recorder &) = delete;
  Recorder &operator=(const Recorder &) = delete;

  void move(const TypeInfo &type, const void *from, void *to);

  const auto &instructions() const { return _instructions; }

  void arg(const TypeInfo &type, const void *a);

  template <class Arg>
  inline void arg(const Var<Arg> *arg) {
    arg(TypeInfo::get<Arg>(), (uintptr_t)(const void *)arg);
  }

  void op(const Operator *op);

  void opIndirect(const Operator *op, size_t argc, void **argv);

  template <class... Args>
  inline void op(const Operator *o, Args *...args) {
    void *pointers[] = {(void *)args...};
    opIndirect(o, sizeof...(Args), pointers);
  }

  template <class T>
  inline void move(const T *from, T *to) {
    move(TypeInfo::get<T>(), from, to);
  }

  template <class T>
  inline void rewrite(const T *from, T *to) {
    if (!_instructions.empty()) {
      if (_instructions.back().code() == (uintptr_t)from) {
        _instructions.back() = Program::Instruction((uintptr_t)to);
        return;
      }
    }
    move(from, to);
  }

  template <class T>
  auto *input(Var<T> *p,
              Program::InputMode mode = Program::InputMode::Variable) {
    uintptr_t addr = _alloc.alloc(TypeInfo::get<T>());
    _inputs.emplace_back(TypeInfo::get<T>(), addr, 0, (uintptr_t)&p->value(),
                         -1, -1, mode);
    uintptr_t temp = (addr | 0x8000000000000000ul);
    move((const T *)temp, (T *)&p->value());
    return &_inputs.back();
  }

  template <class T>
  void parameter(const Var<T> *p) {
    // _parameters.emplace_back(TypeInfo::get<T>(), (uintptr_t)&p->value(), 0,
    //                          (uintptr_t)&p->value());
    uintptr_t addr = _alloc.alloc(TypeInfo::get<T>());
    _parameters.emplace_back(TypeInfo::get<T>(), addr, 0,
                             (uintptr_t)&p->value());
    uintptr_t temp = (addr | 0x8000000000000000ul);
    move((const T *)temp, (T *)&p->value());
  }

  template <class T>
  void output(const Var<T> *p) {
    outputImpl(p, true);
  }

  template <class T>
  inline void goal(const Var<T> &v, int priority = 0,
                   const std::string &name = std::string()) {
    _goals.emplace_back(_outputs.size(), priority);
    outputImpl(&v, false);
    _outputs.back().name() = name;
  }

  void input(const TypeInfo &type, void *var, void *binding,
             const char *name = nullptr);
  void parameter(const TypeInfo &type, void *var, void *binding,
                 const char *name = nullptr);
  void output(const TypeInfo &type, void *var, void *binding,
              const char *name = nullptr);
  void goal(const TypeInfo &type, const void *var, size_t priority = 0,
            const char *name = nullptr);

  void constant(const TypeInfo &type, const void *var);

  template <class T>
  void constant(const Var<T> *p) {
    constant(TypeInfo::get<T>(), p);
  }

  void reference(const std::shared_ptr<const void> &ref);
};

template <class T>
inline void goal(const Var<T> &v, int priority = 0,
                 const std::string &name = std::string()) {
  if (auto inst = Recorder::instance()) {
    inst->goal(v, priority, name);
  }
}

template <class T>
inline void goal(const T &v, int priority = 0,
                 const std::string &name = std::string()) {}

template <class T>
inline void output(Var<T> &p) {
  if (auto inst = Recorder::instance()) {
    inst->output(&p);
  }
}

template <class T>
inline void parameter(T &p) {}

template <class T>
inline void parameter(Var<T> &p) {
  if (auto inst = Recorder::instance()) {
    inst->parameter(&p);
  }
}

class Operator;

template <class... Args>
inline void recordOperation(const Operator *op, Args *...args) {
  if (auto inst = Recorder::instance()) {
    inst->op(op, args...);
  }
}

void callAndRecord(const Operator *op, void **args);

}  // namespace tractor
