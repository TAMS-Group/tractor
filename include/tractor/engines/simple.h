// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/engines/base.h>

namespace tractor {

class SimpleEngine : public EngineBase {
 private:
  class ExecutableImpl : public ExecutableBase {
    struct Instruction {
      OpFunction op = nullptr;
      size_t base = 0;
    };
    std::vector<Instruction> _instructions;
    std::vector<uintptr_t> _arguments;

   protected:
    virtual void _compile(const Program &program) override;
    virtual void _execute(const std::shared_ptr<Memory> &memory) const override
        TRACTOR_FAST;
  };

 public:
  virtual std::shared_ptr<Executable> createExecutable() const override {
    return std::make_shared<ExecutableImpl>();
  }
};

}  // namespace tractor
