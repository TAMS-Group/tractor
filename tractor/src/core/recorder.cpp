// (c) 2020-2022 Philipp Ruppel

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
      rec->push((uintptr_t)args[i]);
    }
  }
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
  _references.push_back(ref);
}

void Recorder::constant(const TypeInfo &type, const void *var) {

  // std::string key((const char *)var, type.size());
  // {
  //   auto it = _const_map.find(key);
  //   if (it != _const_map.end()) {
  //     move(type, (const void *)it->second, (void *)var);
  //     TRACTOR_INFO_STREAM("merge constants");
  //     // TRACTOR_INFO_STREAM("merge constants " << *(double *)var);
  //     return;
  //   }
  // }

  size_t start = _const_data.size();
  _const_data.resize(start + type.size());
  std::memcpy(_const_data.data() + start, var, type.size());

  uintptr_t addr = _alloc.alloc(type);
  _constants.emplace_back(type, addr, (uintptr_t)start);

  uintptr_t temp = (addr | 0x8000000000000000ul);
  move(type, (const void *)temp, (void *)var);

  //_const_map[key] = temp;
}

void Recorder::move(const TypeInfo &type, const void *from, void *to) {
  const Operator *move_op = Operator::find<compute, op_move>({type});
  op(move_op);
  push((uintptr_t)from);
  push((uintptr_t)to);
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

void Recorder::op(const Operator *op) {
  // TRACTOR_DEBUG_STREAM("record op " << op->name());
  _instructions.push_back((uintptr_t)op);
}

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

static void removeUnusedConstants(Program &program) {

  TRACTOR_DEBUG_STREAM("removing unused constants");

  std::unordered_set<uintptr_t> used;

  for (auto &inst : program.instructions()) {
    for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
      if (inst.op()->arg(iarg).isInput()) {
        used.insert(inst.arg(iarg));
      }
    }
  }

  for (auto &port : program.outputs()) {
    used.insert(port.address());
  }

  std::vector<Program::Constant> constants;
  for (auto &constant : program.constants()) {
    if (used.find(constant.address()) != used.end()) {
      constants.push_back(constant);
    }
  }
  program.setConstants(constants);
}

static void removeUnusedInstructions(Program &program) {

  TRACTOR_DEBUG_STREAM("removing unused instructions");

  std::vector<uint8_t> used(program.memorySize(), 0);

  for (auto &port : program.outputs()) {
    used[port.address()] = true;
  }

  size_t used_count = 0;

  std::vector<const Program::Instruction *> instructions;
  for (auto &inst : program.instructions()) {
    instructions.push_back(&inst);
  }
  std::reverse(instructions.begin(), instructions.end());

  std::vector<Program::Instruction> new_insts;

  for (auto *instp : instructions) {
    auto &inst = *instp;
    bool op_used = false;
    for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
      if (inst.op()->arg(iarg).isOutput()) {
        if (used[inst.arg(iarg)]) {
          op_used = true;
        }
      }
    }
    if (op_used) {
      used_count++;
      for (ssize_t iarg = inst.argumentCount() - 1; iarg >= 0; iarg--) {
        new_insts.push_back(inst.arg(iarg));
      }
      new_insts.push_back(inst.op());
      for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
        if (inst.op()->arg(iarg).isInput()) {
          used[inst.arg(iarg)] = true;
        }
      }
    }
  }

  program.setInstructions(new_insts.rbegin(), new_insts.rend());

  TRACTOR_DEBUG_STREAM(instructions.size() << " ops");
  TRACTOR_DEBUG_STREAM(used_count << " used ("
                                  << used_count * 100 / instructions.size()
                                  << "%)");
  TRACTOR_DEBUG_STREAM((instructions.size() - used_count)
                       << " unused ("
                       << (instructions.size() - used_count) * 100 /
                              instructions.size()
                       << "%)");
}

static void precomputeConstants(Program &program) {

  TRACTOR_DEBUG_STREAM("precomputing constants");

  AlignedStdVector<uint8_t> constness(program.memorySize(), 0);
  for (auto &port : program.constants()) {
    constness[port.address()] = true;
  }

  AlignedStdVector<uint8_t> memory(program.memorySize(), 0);
  for (auto &port : program.constants()) {
    std::memcpy(memory.data() + port.address(),
                program.constData().data() + port.offset(), port.size());
  }

  std::vector<Program::Instruction> new_insts;
  auto new_const_data = program.constData();

  Allocator alloc;
  alloc.keep(program);

  size_t const_move_count = 0;
  size_t const_op_count = 0;
  size_t op_count = 0;
  for (auto &inst : program.instructions()) {

    bool is_const = false;
    for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
      if (inst.op()->arg(iarg).isInput()) {
        is_const = true;
        if (!constness[inst.arg(iarg)]) {
          is_const = false;
          break;
        }
      }
    }

    for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
      if (inst.op()->arg(iarg).isOutput()) {
        constness[inst.arg(iarg)] = is_const;
      }
    }

    if (is_const) {
      // TRACTOR_DEBUG_STREAM("op is const " << inst.op()->name());
      if (inst.op()->is<op_move>()) {
        const_move_count++;
      } else {
        const_op_count++;
      }
      inst.op()->callIndirect(memory.data(), &inst.arg(0));
      for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
        if (inst.op()->arg(iarg).isOutput()) {
          auto *move_op = Operator::find<compute, op_move>(
              {inst.op()->arg(iarg).typeInfo()});
          auto new_addr = alloc.alloc(inst.op()->arg(iarg).typeInfo());
          new_insts.push_back(move_op);
          new_insts.push_back(new_addr);
          new_insts.push_back(inst.arg(iarg));
          program.addConstant(Program::Constant(inst.op()->arg(iarg).typeInfo(),
                                                new_addr,
                                                new_const_data.size()));
          for (size_t i = 0; i < inst.op()->arg(iarg).size(); i++) {
            new_const_data.push_back(memory[inst.arg(iarg) + i]);
          }
        }
      }
    } else {
      new_insts.push_back(inst.op());
      for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
        new_insts.push_back(inst.arg(iarg));
      }
    }
    op_count++;
  }

  program.setInstructions(new_insts);
  program.setConstData(new_const_data);
  alloc.apply(program);

  TRACTOR_DEBUG_STREAM(op_count << " ops");
  TRACTOR_DEBUG_STREAM(const_move_count << " const move ("
                                        << const_move_count * 100 / op_count
                                        << "%)");
  TRACTOR_DEBUG_STREAM(const_op_count << " const ops ("
                                      << const_op_count * 100 / op_count
                                      << "%)");
}

static void skipMoves(Program &program) {

  TRACTOR_DEBUG_STREAM("skipping redundant moves");

  std::unordered_map<size_t, size_t> move_dst_to_src;
  std::vector<Program::Instruction> new_instructions;
  size_t arg_count = 0;
  size_t rewrite_count = 0;
  for (auto &inst : program.instructions()) {
    new_instructions.push_back(inst.op());
    for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
      auto argc = inst.arg(iarg);
      if (inst.op()->arg(iarg).isOutput()) {
        move_dst_to_src.erase(inst.arg(iarg));
      }
      if (inst.op()->arg(iarg).isInput()) {
        auto it = move_dst_to_src.find(argc);
        if (it != move_dst_to_src.end()) {
          argc = it->second;
          rewrite_count++;
        }
      }
      arg_count++;
      new_instructions.push_back(argc);
    }
    if (inst.op()->is<op_move>()) {
      auto argc = inst.arg(0);
      {
        auto it = move_dst_to_src.find(argc);
        if (it != move_dst_to_src.end()) {
          argc = it->second;
        }
      }
      move_dst_to_src[inst.arg(1)] = argc;
    }
  }
  /*
  for (auto &port : program.outputs()) {
    auto &argc = port.address();
    auto it = move_dst_to_src.find(argc);
    if (it != move_dst_to_src.end()) {
      argc = it->second;
      rewrite_count++;
    }
  }
  */
  program.setInstructions(new_instructions.begin(), new_instructions.end());
  TRACTOR_DEBUG_STREAM(rewrite_count << " moves / " << arg_count
                                     << " args skipped");
}

static void checkMemory(const Program &program) {

  TRACTOR_DEBUG_STREAM("checking memory");

  TRACTOR_DEBUG_STREAM("memory size " << program.memorySize());
  std::vector<uint8_t> memory;
  memory.resize(program.memorySize(), 0);

  for (auto &port : program.inputs()) {
    memory.at(port.address()) = 1;
  }

  for (auto &port : program.constants()) {
    memory.at(port.address()) = 1;
  }

  for (auto &port : program.parameters()) {
    memory.at(port.address()) = 1;
  }

  for (auto &inst : program.instructions()) {
    for (size_t iarg = 0; iarg < inst.op()->argumentCount(); iarg++) {
      if ((inst.arg(iarg) % inst.op()->arg(iarg).typeInfo().alignment()) != 0) {
        // throw std::runtime_error("memory check: alignment error");
        throw std::runtime_error("memory check: alignment error " +
                                 inst.op()->name() + " " +
                                 std::to_string(iarg) + " " +
                                 std::to_string(inst.op()->arg(iarg).size()) +
                                 " " + std::to_string(inst.arg(iarg)));
      }
      if (inst.op()->arg(iarg).isInput()) {
        if (!memory.at(inst.arg(iarg))) {
          for (auto &inst2 : program.instructions()) {
            TRACTOR_DEBUG_STREAM("op " << inst2.op()->name());
            for (size_t iarg = 0; iarg < inst2.op()->argumentCount(); iarg++) {
              TRACTOR_DEBUG_STREAM("arg " << inst2.arg(iarg) << ":"
                                          << inst2.op()->arg(iarg).size());
            }
            if (&inst2 == &inst) {
              break;
            }
          }
          for (auto &port : program.inputs()) {
            TRACTOR_DEBUG_STREAM("input " << port.address() << " "
                                          << port.size());
          }
          for (auto &port : program.constants()) {
            TRACTOR_DEBUG_STREAM("constant " << port.address() << " "
                                             << port.size());
          }
          for (auto &port : program.parameters()) {
            TRACTOR_DEBUG_STREAM("parameter " << port.address() << " "
                                              << port.size());
          }
          throw std::runtime_error(
              "parameter read from uninitialized memory z " +
              inst.op()->name() + " " + std::to_string(iarg) + " " +
              std::to_string(inst.op()->arg(iarg).size()) + " " +
              std::to_string(inst.arg(iarg)));
        }
      }
      if (inst.op()->arg(iarg).isOutput()) {
        memory.at(inst.arg(iarg)) = 1;
      }
    }
  }

  for (auto &port : program.outputs()) {
    if (!memory.at(port.address())) {
      throw std::runtime_error("output read from uninitialized memory");
    }
  }
}

// static void pruneConstants(Program &program) {
//
//   Allocator const_alloc;
//   std::vector<uint8_t> new_const_data;
//   std::unordered_map<std::string, uintptr_t> const_map;
//
//   for (auto &constant : program.constants()) {
//
//     std::string key((const char *)program.constData() + constant.offset(),
//                     constant.size());
//
//     if (const_map.find(key) == const_map.end()) {
//
//       auto addr = const_alloc.alloc(constant.type());
//       new_const_data.resize(addr + constant.size());
//       std::memcpy(new_const_data.data() + addr,
//                   program.constData() + constant.offset(), constant.size());
//     }
//
//     constant.address() = const_map[key];
//   }
//
//   program.setConstData(new_const_data);
// }

static void defragmentMemory(Program &program) {

  Allocator allocator;
  std::unordered_map<uintptr_t, uintptr_t> mapping;
  auto map = [&](const TypeInfo &type, uintptr_t a) {
    {
      auto it = mapping.find(a);
      if (it != mapping.end()) {
        return it->second;
      }
    }
    uintptr_t b = allocator.alloc(type);
    mapping[a] = b;
    return mapping[a];
  };

  for (auto &port : program.constants()) {
    port.address() = map(port.typeInfo(), port.address());
  }
  for (auto &port : program.parameters()) {
    port.address() = map(port.typeInfo(), port.address());
  }
  for (auto &port : program.inputs()) {
    port.address() = map(port.typeInfo(), port.address());
  }
  for (auto &port : program.outputs()) {
    port.address() = map(port.typeInfo(), port.address());
  }

  // std::vector<Program::Instruction> new_insts;
  for (auto &inst : program.instructions()) {
    for (size_t iarg = 0; iarg < inst.op()->argumentCount(); iarg++) {
      inst.arg(iarg) = map(inst.op()->arg(iarg).typeInfo(), inst.arg(iarg));
    }
  }

  TRACTOR_DEBUG_STREAM("defragmentation reducing memory size from "
                       << program.memorySize() << " to " << allocator.top());

  allocator.apply(program);
}

void Recorder::finish(Program &program) {

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

    for (auto port : _constants) {
      program.addConstant(port);
    }

    program.setBoundData(_bound_data);
    program.setConstData(_const_data);
    program.setInstructions(prog_insts);
    _alloc.apply(program);
  }

  checkMemory(program);

  // return;

  TRACTOR_DEBUG_STREAM("code size " << program.code().size());

  TRACTOR_DEBUG_STREAM("precompute constants");
  precomputeConstants(program);
  // checkMemory(program);

  skipMoves(program);
  // checkMemory(program);

  removeUnusedInstructions(program);
  // checkMemory(program);

  removeUnusedConstants(program);
  // checkMemory(program);

  defragmentMemory(program);

  TRACTOR_DEBUG_STREAM("code size " << program.code().size());

  checkMemory(program);
}

} // namespace tractor
