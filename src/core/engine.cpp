// 2020-2024 Philipp Ruppel

#include <tractor/core/engine.h>

#include <tractor/core/profiler.h>

namespace tractor {

ArrayRef<const Program::Input> Executable::inputs() const {
  _checkCompiled();
  return _inputs;
}

ArrayRef<const Program::Output> Executable::outputs() const {
  _checkCompiled();
  return _outputs;
}

ArrayRef<const Program::Parameter> Executable::parameters() const {
  _checkCompiled();
  return _parameters;
}

size_t Executable::inputBufferSize() const {
  _checkCompiled();
  return _input_size;
}

size_t Executable::outputBufferSize() const {
  _checkCompiled();
  return _output_size;
}

void Executable::execute(const std::shared_ptr<Memory> &memory) const {
  _checkCompiled();
  TRACTOR_PROFILER("execute");
  _execute(memory);
}

void Executable::input(const Buffer &input,
                       const std::shared_ptr<Memory> &memory) const {
  _checkCompiled();
  TRACTOR_PROFILER("input");
  _input(input, memory);
}

void Executable::output(const std::shared_ptr<const Memory> &memory,
                        Buffer &output) const {
  _checkCompiled();
  TRACTOR_PROFILER("output");
  _output(memory, output);
}

void Executable::parameterize(const Buffer &data,
                              const std::shared_ptr<Memory> &memory) const {
  _checkCompiled();
  TRACTOR_PROFILER("parameterize");
  _parameterize(data, memory);
}

void Executable::parameterize(const std::shared_ptr<Memory> &memory) const {
  _checkCompiled();
  TRACTOR_PROFILER("parameterize");
  Buffer temp;
  temp.gather(parameters());
  _parameterize(temp, memory);
}

void Executable::compile(const Program &program) {
  for (auto &inst : program.instructions()) {
    if (!inst.op()) {
      throw std::runtime_error("invalid op");
    }
  }
  _program_context = program.context();
  _inputs.clear();
  _input_size = 0;
  for (auto &port : program.inputs()) {
    _inputs.push_back(port);
    _input_size = std::max(_input_size, port.offset() + port.size());
  }
  _outputs.clear();
  _output_size = 0;
  for (auto &port : program.outputs()) {
    _outputs.push_back(port);
    _output_size = std::max(_output_size, port.offset() + port.size());
  }
  _param_size = 0;
  for (auto &port : program.parameters()) {
    _parameters.push_back(port);
    _param_size = std::max(_param_size, port.offset() + port.size());
  }
  _constants.assign(&*program.constants().begin(), &*program.constants().end());
  _const_data.assign(&*program.constData().begin(),
                     &*program.constData().end());
  _memory_size = program.memorySize();
  _compile(program);
  _compiled = true;
}

std::shared_ptr<Executable> Engine::compile(const Program &program) const {
  auto x = createExecutable();
  x->compile(program);
  return x;
}

}  // namespace tractor
