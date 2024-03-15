// 2020-2024 Philipp Ruppel

#include <tractor/core/profiler.h>
#include <tractor/engines/simple.h>

namespace tractor {

void SimpleEngine::ExecutableImpl::_compile(const Program &program) {
  _instructions.clear();
  _arguments.clear();
  for (auto &instp : program.instructions()) {
    Instruction inst;
    auto *op = instp.op();
    inst.op = op->functionPointers().indirect;
    inst.base = _arguments.size();
    _instructions.push_back(inst);
    for (auto &v : op->functionPointers().context) {
      _arguments.push_back(v);
    }
    for (auto &arg : instp.args()) {
      _arguments.push_back(arg);
    }
  }
}

void SimpleEngine::ExecutableImpl::_execute(
    const std::shared_ptr<Memory> &memory) const {
  auto &temp = *(MemoryImpl *)(memory.get());
  temp.resize(std::max(temp.size(), _memory_size));
  // std::memset(temp.data(), 0, temp.size());
  {
    TRACTOR_PROFILER("load constants");
    for (auto &port : _constants) {
      std::memcpy((uint8_t *)temp.data() + port.address(),
                  _const_data.data() + port.offset(), port.size());
    }
  }
  {
    TRACTOR_PROFILER("execute instructions");
    for (auto &inst : _instructions) {
      inst.op(temp.data(), _arguments.data() + inst.base);
    }
  }
}

}  // namespace tractor
