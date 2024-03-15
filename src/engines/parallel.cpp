// 2020-2024 Philipp Ruppel

#include <tractor/core/profiler.h>
#include <tractor/engines/parallel.h>

#include <omp.h>

namespace tractor {

void ParallelEngine::ExecutableImpl::_compile(const Program &program) {
  _waves.clear();

  std::unordered_map<size_t, size_t> levels;

  for (auto &instp : program.instructions()) {
    size_t level = 0;
    for (size_t iarg = 0; iarg < instp.argumentCount(); iarg++) {
      if (instp.op()->arg(iarg).isInput()) {
        level = std::max(level, levels[instp.arg(iarg)]);
      }
    }
    level++;
    for (size_t iarg = 0; iarg < instp.argumentCount(); iarg++) {
      if (instp.op()->arg(iarg).isOutput()) {
        levels[instp.arg(iarg)] = std::max(levels[instp.arg(iarg)], level);
      }
    }

    while (_waves.size() <= level) {
      _waves.emplace_back();
    }
    auto &wave = _waves.at(level);

    Instruction inst;
    auto *op = instp.op();
    inst.op = op->functionPointers().indirect;
    inst.base = wave.arguments.size();

    wave.instructions.push_back(inst);
    for (auto &v : op->functionPointers().context) {
      wave.arguments.push_back(v);
    }
    for (auto &arg : instp.args()) {
      wave.arguments.push_back(arg);
    }
  }

  for (size_t iw = 0; iw < _waves.size(); iw++) {
    auto &wave = _waves[iw];
    TRACTOR_DEBUG("wave " << iw << " " << wave.instructions.size());
  }
}

void ParallelEngine::ExecutableImpl::_execute(
    const std::shared_ptr<Memory> &memory) const {
  auto &temp = *(MemoryImpl *)(memory.get());
  temp.resize(std::max(temp.size(), _memory_size));

  {
    TRACTOR_PROFILER("load constants");
    for (auto &port : _constants) {
      std::memcpy((uint8_t *)temp.data() + port.address(),
                  _const_data.data() + port.offset(), port.size());
    }
  }

  {
    TRACTOR_PROFILER("execute instructions");
    {
#pragma omp parallel
      {
        size_t ti = omp_get_thread_num();
        size_t tn = omp_get_num_threads();
        auto *tempdata = temp.data();
        for (auto &wave : _waves) {
          auto *instructions = wave.instructions.data();
          auto *arguments = wave.arguments.data();
          size_t n = wave.instructions.size();
          for (size_t i = ti; i < n; i += tn) {
            auto &inst = instructions[i];
            inst.op(tempdata, arguments + inst.base);
          }
#pragma omp barrier
        }
      }
    }
  }
}

}  // namespace tractor
