// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/engines/base.h>

namespace tractor {

class LoopEngine : public EngineBase {
 private:
  class ExecutableImpl : public ExecutableBase {
    struct Instruction {
      // LoopFunction loop_function = nullptr;
      OpFunction op_function = nullptr;
      size_t base = 0;
      size_t iterations = 0;
      size_t arity = 0;
      bool single_threaded = true;
    };
    std::vector<Instruction> _instructions;
    std::vector<uintptr_t> _arguments;

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
