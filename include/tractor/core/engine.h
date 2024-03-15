// 2020-2024 Philipp Ruppel

#pragma once

#include <cstring>

#include <tractor/core/operator.h>

namespace tractor {

struct Memory {
  virtual ~Memory() {}
  virtual void copyTo(const std::shared_ptr<Memory> &other) const = 0;
};

class Executable {
 private:
  size_t _input_size = 0, _output_size = 0, _param_size = 0;
  bool _compiled = false;
  void _checkCompiled() const {
    if (!_compiled) {
      throw std::runtime_error("call compile(...) before use");
    }
  }
  virtual void _compile(const Program &program) = 0;
  virtual void _execute(const std::shared_ptr<Memory> &memory) const = 0;
  virtual void _input(const Buffer &input,
                      const std::shared_ptr<Memory> &memory) const = 0;
  virtual void _output(const std::shared_ptr<const Memory> &memory,
                       Buffer &output) const = 0;
  virtual void _parameterize(const Buffer &data,
                             const std::shared_ptr<Memory> &memory) const = 0;

 protected:
  std::vector<Program::Input> _inputs;
  std::vector<Program::Output> _outputs;
  std::vector<Program::Constant> _constants;
  std::vector<Program::Parameter> _parameters;
  size_t _memory_size = 0;
  std::vector<uint8_t> _const_data;
  std::shared_ptr<Program::Context> _program_context;

 public:
  virtual ~Executable() {}
  void compile(const Program &program);
  void execute(const std::shared_ptr<Memory> &memory) const;
  void input(const Buffer &input, const std::shared_ptr<Memory> &memory) const;
  void output(const std::shared_ptr<const Memory> &memory,
              Buffer &output) const;
  ArrayRef<const Program::Input> inputs() const;
  ArrayRef<const Program::Output> outputs() const;
  ArrayRef<const Program::Parameter> parameters() const;
  size_t inputBufferSize() const;
  size_t outputBufferSize() const;
  template <class Vector>
  void inputVector(Vector &&inputv,
                   const std::shared_ptr<Memory> &memory) const {
    _checkCompiled();
    Buffer temp;
    temp.fromVector(inputv);
    input(temp, memory);
  }
  template <class Vector>
  void outputVector(const std::shared_ptr<const Memory> &memory,
                    Vector &&outputv) const {
    _checkCompiled();
    Buffer temp;
    output(memory, temp);
    temp.toVector(outputv);
  }
  template <class Input, class Output>
  void run(Input &&input, const std::shared_ptr<Memory> &memory,
           Output &&output) const {
    inputVector(input, memory);
    execute(memory);
    outputVector(memory, output);
  }
  void parameterize(const Buffer &data,
                    const std::shared_ptr<Memory> &memory) const;
  void parameterize(const std::shared_ptr<Memory> &memory) const;
  template <class Vector>
  void parameterVector(const Vector &paramv,
                       const std::shared_ptr<Memory> &memory) const {
    _checkCompiled();
    Buffer temp;
    temp.fromVector(paramv);
    parameterize(temp, memory);
  }
};

struct Engine {
  virtual std::shared_ptr<Memory> createMemory() const = 0;
  virtual std::shared_ptr<Executable> createExecutable() const = 0;
  std::shared_ptr<Executable> compile(const Program &program) const;
};

}  // namespace tractor
