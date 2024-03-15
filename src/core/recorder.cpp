// 2020-2024 Philipp Ruppel

#include <tractor/core/recorder.h>

#include <tractor/core/factory.h>
#include <tractor/core/log.h>
#include <tractor/core/operator.h>
#include <tractor/core/ops.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace tractor {

static thread_local Recorder *g_recorder_instance = nullptr;

void callAndRecord(const Operator *op, void **args) {
  op->callIndirect(args);
  if (auto *rec = Recorder::instance()) {
    rec->op(op);
    for (size_t i = 0; i < op->argumentCount(); i++) {
      rec->arg(op->arg(i).typeInfo(), args[i]);
    }
  }
}

void Recorder::opIndirect(const Operator *o, size_t argc, void **argv) {
  if (o->argumentCount() != argc) {
    throw std::runtime_error("op indirect wrong number of arguments");
  }
  op(o);
  for (size_t i = 0; i < argc; i++) {
    arg(o->arg(i).typeInfo(), argv[i]);
  }
}

void Recorder::arg(const TypeInfo &type, const void *a) {
  if (!_pending_op) {
    throw std::runtime_error(
        "attempting to emit arguments without current operation");
  }
  auto &arg = _pending_op->arg(_pending_args.size());
  if (arg.typeInfo() != type) {
    throw std::runtime_error("emit arg type mismatch");
  }
  if ((((uintptr_t)a & 0x8000000000000000ul) == 0) &&
      _known_addresses.find(a) == _known_addresses.end() && arg.isInput()) {
    // TRACTOR_DEBUG("implicit constant " << type.name() << " " << a);
    size_t start = _const_data.size();
    _const_data.resize(start + type.size());
    std::memcpy(_const_data.data() + start, a, type.size());
    _constants.emplace_back(type, (uintptr_t)a, (uintptr_t)start);
  }
  // if (arg.isOutput()) {
  _known_addresses.insert(a);
  //}
  _pending_args.push_back(a);
  if (_pending_args.size() == _pending_op->argumentCount()) {
    _instructions.push_back((uintptr_t)_pending_op);
    for (auto &a : _pending_args) {
      _instructions.push_back((uintptr_t)a);
    }
    _pending_args.clear();
    _pending_op = nullptr;
  }
}

void Recorder::op(const Operator *op) {
  if (_pending_op) {
    throw std::runtime_error(
        "attempted to emit operation before finishing previous one " +
        _pending_op->name() + " " + op->name());
  }
  if (!_pending_args.empty()) {
    throw std::runtime_error(
        "attempted to emit operation before finishing previous one " +
        op->name());
  }
  _pending_op = op;
}

void Recorder::goal(const TypeInfo &type, const void *var, size_t priority,
                    const char *name) {
  _goals.emplace_back(_outputs.size(), priority);
  uintptr_t temp = (((uintptr_t)_alloc.alloc(type)) | 0x8000000000000000ul);
  move(type, var, (void *)temp);
  _outputs.emplace_back(type, temp, 0, 0);
  if (name) {
    _outputs.back().name() = name;
  }
}

void Recorder::reference(const std::shared_ptr<const void> &ref) {
  _references.insert(ref);
}

void Recorder::constant(const TypeInfo &type, const void *var) {
  size_t start = _const_data.size();
  _const_data.resize(start + type.size());
  std::memcpy(_const_data.data() + start, var, type.size());

  uintptr_t addr = _alloc.alloc(type);

  uintptr_t temp = (addr | 0x8000000000000000ul);
  _constants.emplace_back(type, temp, (uintptr_t)start);
  move(type, (const void *)temp, (void *)var);
}

void Recorder::move(const TypeInfo &type, const void *from, void *to) {
  const Operator *move_op = Operator::find<compute, op_move>({type});
  op(move_op);
  arg(type, from);
  arg(type, to);
}

void Recorder::input(const TypeInfo &type, void *var, void *binding,
                     const char *name) {
  uintptr_t addr = _alloc.alloc(type);
  _inputs.emplace_back(type, addr, 0, (uintptr_t)binding, -1, -1,
                       Program::InputMode::Variable);
  uintptr_t temp = (addr | 0x8000000000000000ul);
  move(type, (void *)temp, var);
}

void Recorder::parameter(const TypeInfo &type, void *var, void *binding,
                         const char *name) {
  uintptr_t addr = _alloc.alloc(type);
  _parameters.emplace_back(type, addr, 0, (uintptr_t)binding);
  uintptr_t temp = (addr | 0x8000000000000000ul);
  move(type, (void *)temp, var);
}

void Recorder::output(const TypeInfo &type, void *var, void *binding,
                      const char *name) {
  uintptr_t temp = (((uintptr_t)_alloc.alloc(type)) | 0x8000000000000000ul);
  move(type, var, (void *)temp);
  _outputs.emplace_back(type, temp, 0, (uintptr_t)binding);
}

Recorder *Recorder::instance() { return g_recorder_instance; }

Recorder::Recorder(Program *program) : _program(program) {
  _const_data.resize(64, 0);
  //_memory_size = memory_alignment;
  _alloc.clear();
  if (g_recorder_instance) {
    throw std::runtime_error("already recording");
  }
  g_recorder_instance = this;
}

Recorder::~Recorder() {
  finish(*_program);
  g_recorder_instance = nullptr;
}

void Recorder::finish(Program &program) {
  TRACTOR_INFO("rec finish");

  {
    program.clear();

    program.createNewContext();
    program.context()->references.assign(_references.begin(),
                                         _references.end());

    std::unordered_map<uintptr_t, uintptr_t> host_to_buffer_address;
    auto map = [&](uintptr_t a, const TypeInfo &type, bool alloc = false) {
      if (a & 0x8000000000000000ul) {
        return a & ~0x8000000000000000ul;
      }
      auto &addr = host_to_buffer_address[a];
      if (!addr || alloc) {
        addr = _alloc.alloc(type);
      }
      return addr;
    };

    for (auto port : _constants) {
      port.address() = map(port.address(), port.typeInfo());
      program.addConstant(port);
    }

    {
      size_t offset = 0;
      for (auto port : _inputs) {
        port.address() = port.address();
        port.offset() = offset;
        offset += port.size();
        program.addInput(port);
      }
    }

    {
      size_t offset = 0;
      for (auto port : _parameters) {
        port.address() = port.address();
        port.offset() = offset;
        offset += port.size();
        program.addParameter(port);
      }
    }

    std::vector<Program::Instruction> prog_insts;
    for (auto &rec_inst :
         ArrayRef<Program::Instruction,
                  Program::InstructionIterator<const Program::Instruction>>(
             _instructions)) {
      auto *op = rec_inst.op();
      prog_insts.emplace_back(rec_inst.code());
      for (size_t i = 0; i < op->argumentCount(); i++) {
        auto &rec_arg = rec_inst.arg(i);
        auto &op_arg = op->arg(i);
        auto prog_arg = map(rec_arg, op_arg.typeInfo(), op_arg.isOutput());
        prog_insts.emplace_back(prog_arg);
      }
    }

    {
      size_t offset = 0;
      for (auto port : _outputs) {
        port.address() = map(port.address(), port.typeInfo());
        port.offset() = offset;
        offset += port.size();
        program.addOutput(port);
      }
    }

    for (auto &goal : _goals) {
      program.addGoal(goal);
    }

    program.setBoundData(_bound_data);
    program.setConstData(_const_data);
    program.setInstructions(prog_insts);
    _alloc.apply(program);
  }
  TRACTOR_INFO("rec ready");
}

}  // namespace tractor
