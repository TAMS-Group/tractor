// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/engines/base.h>

namespace tractor {

class ParallelEngine : public EngineBase {
 private:
  class ExecutableImpl : public ExecutableBase {
    struct Instruction {
      OpFunction op = nullptr;
      size_t base = 0;
    };

    struct Wave {
      std::vector<Instruction> instructions;
      std::vector<uintptr_t> arguments;
    };
    std::vector<Wave> _waves;

   protected:
    virtual void _compile(const Program &program) override;
    virtual void _execute(const std::shared_ptr<Memory> &memory) const override;
  };

 public:
  virtual std::shared_ptr<Executable> createExecutable() const override {
    return std::make_shared<ExecutableImpl>();
  }
};

}  // namespace tractor
