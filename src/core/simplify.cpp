// 2020-2024 Philipp Ruppel

#include <tractor/core/simplify.h>

#include <tractor/core/allocator.h>
#include <tractor/core/operator.h>
#include <tractor/core/ops.h>
#include <tractor/core/profiler.h>
#include <tractor/core/program.h>

#include <unordered_set>

namespace tractor {

void simplify(Program &program) {
  TRACTOR_PROFILER("simplify program");
  TRACTOR_DEBUG("simplify program start");
  precomputeConstants(program);
  removeDuplicateConstants(program);
  compressZeroConstants(program);
  skipMoves(program);
  removeUnusedInstructions(program);
  removeUnusedConstants(program);
  defragmentMemory(program);
  TRACTOR_DEBUG("simplify program finished");
}

static bool allZero(const void *data, size_t size) {
  const uint8_t *d = (const uint8_t *)data;
  for (size_t i = 0; i < size; i++) {
    if (d[i]) {
      return false;
    }
  }
  return true;
}

void compressZeroConstants(Program &program) {
  TRACTOR_PROFILER("compress zero constants");

  std::vector<Program::Constant> new_constants;
  std::vector<Program::Instruction> new_instructions;

  for (auto &constant : program.constants()) {
    if (allZero(program.constData().data() + constant.offset(),
                constant.typeInfo().size())) {
      new_instructions.push_back(
          Operator::find<compute, op_zero>({constant.typeInfo()}));
      new_instructions.push_back(constant.address());
    } else {
      new_constants.push_back(constant);
    }
  }

  for (auto &c : program.code()) {
    new_instructions.push_back(c);
  }

  program.setConstants(new_constants);
  program.setInstructions(new_instructions);
}

void removeDuplicateConstants(Program &program) {
  TRACTOR_PROFILER("compress duplicate constants");

  std::unordered_map<std::string, uintptr_t> const_map;

  std::vector<Program::Constant> new_constants;
  std::vector<Program::Instruction> new_instructions;

  for (auto &constant : program.constants()) {
    std::string key;
    key.append(constant.typeInfo().name());
    key.append((const char *)program.constData().data() + constant.offset(),
               constant.typeInfo().size());

    auto it = const_map.find(key);
    if (it != const_map.end()) {
      new_instructions.push_back(
          Operator::find<compute, op_move>({constant.typeInfo()}));
      new_instructions.push_back(it->second);
      new_instructions.push_back(constant.address());
    } else {
      const_map[key] = constant.address();
      new_constants.push_back(constant);
    }
  }

  for (auto &c : program.code()) {
    new_instructions.push_back(c);
  }

  program.setConstants(new_constants);
  program.setInstructions(new_instructions);
}

void removeUnusedConstants(Program &program) {
  TRACTOR_PROFILER("remove unused constants");

  TRACTOR_DEBUG("remove unused constants");

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

void removeUnusedInstructions(Program &program) {
  TRACTOR_PROFILER("remove unused instructions");

  TRACTOR_DEBUG("removing unused instructions");

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

  TRACTOR_DEBUG(instructions.size() << " ops");
  TRACTOR_DEBUG(used_count << " used ("
                           << used_count * 100 / instructions.size() << "%)");
  TRACTOR_DEBUG((instructions.size() - used_count)
                << " unused ("
                << (instructions.size() - used_count) * 100 /
                       instructions.size()
                << "%)");
}

void precomputeConstants(Program &program) {
  TRACTOR_PROFILER("precompute constants");

  TRACTOR_DEBUG("precompute constants");

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
      // TRACTOR_DEBUG("op is const " << inst.op()->name());
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

  TRACTOR_DEBUG(op_count << " ops");
  TRACTOR_DEBUG(const_move_count << " const move ("
                                 << const_move_count * 100 / op_count << "%)");
  TRACTOR_DEBUG(const_op_count << " const ops ("
                               << const_op_count * 100 / op_count << "%)");
}

void skipMoves(Program &program) {
  TRACTOR_PROFILER("skip redundant moves");

  TRACTOR_DEBUG("skip redundant moves");

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
  program.setInstructions(new_instructions.begin(), new_instructions.end());
  TRACTOR_DEBUG(rewrite_count << " moves / " << arg_count << " args skipped");
}

void defragmentMemory(Program &program) {
  TRACTOR_PROFILER("defragment memory");

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

  for (auto &inst : program.instructions()) {
    for (size_t iarg = 0; iarg < inst.op()->argumentCount(); iarg++) {
      inst.arg(iarg) = map(inst.op()->arg(iarg).typeInfo(), inst.arg(iarg));
    }
  }

  TRACTOR_DEBUG("defragmentation reducing memory size from "
                << program.memorySize() << " to " << allocator.top());

  allocator.apply(program);
}

}  // namespace tractor
