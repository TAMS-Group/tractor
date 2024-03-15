// 2020-2024 Philipp Ruppel

#include <tractor/engines/loop.h>

#include <tractor/core/log.h>
#include <tractor/core/error.h>

#include <algorithm>
#include <unordered_map>

namespace tractor {

void scheduleSimple(std::vector<const Program::Instruction *> &instructions) {
  std::vector<
      std::tuple<size_t, const Operator *, const Program::Instruction *>>
      level_inst;
  std::unordered_map<size_t, size_t> addr_level;
  for (auto *instp : instructions) {
    auto &inst = *instp;
    size_t level = 0;
    for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
      if (inst.op()->arg(iarg).isInput()) {
        level = std::max(level, addr_level[inst.arg(iarg)]);
      }
    }
    level++;
    for (size_t iarg = 0; iarg < inst.argumentCount(); iarg++) {
      if (inst.op()->arg(iarg).isOutput()) {
        addr_level[inst.arg(iarg)] = level;
      }
    }
    level_inst.emplace_back(level, inst.op(), &inst);
  }

  std::sort(level_inst.begin(), level_inst.end());

  instructions.clear();
  for (auto &tup : level_inst) {
    instructions.push_back(std::get<2>(tup));
  }
}

void LoopEngine::ExecutableImpl::_compile(const Program &program) {
  _instructions.clear();
  _arguments.clear();

  std::vector<const Program::Instruction *> instructions;
  for (auto &inst : program.instructions()) {
    instructions.push_back(&inst);
  }

  scheduleSimple(instructions);

  std::vector<const Operator *> ops;

  for (auto *instp : instructions) {
    auto *op = instp->op();
    // if (_instructions.empty() ||
    //     _instructions.back().loop_function != op->functionPointers().iterate)
    //     {
    if (_instructions.empty() ||
        _instructions.back().op_function != op->functionPointers().indirect) {
      Instruction inst;
      // inst.loop_function = op->functionPointers().iterate;
      // TRACTOR_ASSERT(inst.loop_function);
      // TRACTOR_INFO(op->)
      // if (op->name() == "collision_axes_d") {
      //   inst.single_threaded = false;
      //   // throw 1;
      // }
      inst.op_function = op->functionPointers().indirect;
      inst.base = _arguments.size();
      inst.iterations = 1;
      inst.arity = instp->argumentCount();
      _instructions.push_back(inst);
      ops.push_back(op);
    } else {
      auto &inst = _instructions.back();
      inst.iterations++;
    }
    for (auto &arg : instp->args()) {
      _arguments.push_back(arg);
    }
  }

  TRACTOR_DEBUG("loops");
  for (size_t i = 0; i < ops.size(); i++) {
    TRACTOR_DEBUG(_instructions[i].iterations << " " << ops[i]->name());
  }
}

void LoopEngine::ExecutableImpl::_execute(
    const std::shared_ptr<Memory> &memory) const {
  auto &temp = *(MemoryImpl *)(memory.get());
  temp.resize(std::max(temp.size(), _memory_size));
  for (auto &port : _constants) {
    std::memcpy((uint8_t *)temp.data() + port.address(),
                _const_data.data() + port.offset(), port.size());
  }

  for (auto &inst : _instructions) {
    // inst.loop_function(temp.data(), _arguments.data() + inst.base,
    //                    inst.iterations);
    size_t n = inst.iterations;
    auto f = inst.op_function;
    auto base = _arguments.data() + inst.base;
    size_t a = inst.arity;
    if (inst.single_threaded) {
      for (size_t i = 0; i < n; i++) {
        f(temp.data(), base);
        base += a;
      }
      continue;
    } else {
#pragma omp parallel for
      for (size_t i = 0; i < n; i++) {
        f(temp.data(), base + i * a);
      }
      continue;
    }
  }
}

}  // namespace tractor
