// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/engine.h>
#include <tractor/core/error.h>
#include <tractor/engines/jit.h>
#include <tractor/engines/parallel.h>
#include <tractor/engines/simple.h>
#include <tractor/engines/loop.h>

namespace tractor {

static void pythonizeEngine(py::module &m) {
  py::class_<Memory, std::shared_ptr<Memory>>(m, "Memory");

  py::class_<Executable, std::shared_ptr<Executable>>(m, "Executable")
      .def("run",
           [](Executable &executable, std::shared_ptr<Memory> &memory) {
             Buffer buffer;
             buffer.gather(executable.parameters());
             executable.parameterize(buffer, memory);
             buffer.gather(executable.inputs());
             executable.input(buffer, memory);
             executable.execute(memory);
             executable.output(memory, buffer);
             buffer.scatter(executable.outputs());
           })
      .def("gather",
           [](Executable &executable, std::shared_ptr<Memory> &memory) {
             Buffer buffer;
             buffer.gather(executable.inputs());
             executable.input(buffer, memory);
           })
      .def("parameterize",
           [](Executable &executable, std::shared_ptr<Memory> &memory) {
             executable.parameterize(memory);
           })
      .def("execute",
           [](Executable &executable, std::shared_ptr<Memory> &memory) {
             executable.execute(memory);
           })
      .def_property_readonly("input_buffer_size", &Executable::inputBufferSize)
      .def_property_readonly("output_buffer_size",
                             &Executable::outputBufferSize)
      .def("input",
           [](const std::shared_ptr<Executable> &executable, py::array buffer,
              const std::shared_ptr<Memory> &memory) {
             TRACTOR_ASSERT(buffer.nbytes() >= executable->inputBufferSize());
             Buffer b;
             b.resize(executable->inputBufferSize());
             std::memcpy(b.data(), buffer.data(),
                         executable->inputBufferSize());
             TRACTOR_DEBUG("input data " << buffer.nbytes());
             executable->input(b, memory);
           })
      .def("output", [](const std::shared_ptr<Executable> &executable,
                        const std::shared_ptr<Memory> &memory) {
        Buffer b;
        executable->output(memory, b);
        return py::bytes(std::string((const char *)b.data(), b.size()));
      });

  py::class_<Engine, std::shared_ptr<Engine>>(m, "Engine")
      .def("compile",
           [](Engine &engine, const Program &program) {
             return engine.compile(program);
           })
      .def("createMemory",
           [](Engine &engine) { return engine.createMemory(); });

  py::class_<SimpleEngine, std::shared_ptr<SimpleEngine>, Engine>(
      m, "DefaultEngine")
      .def(py::init<>());

  py::class_<ParallelEngine, std::shared_ptr<ParallelEngine>, Engine>(
      m, "ParallelEngine")
      .def(py::init<>());

  py::class_<LoopEngine, std::shared_ptr<LoopEngine>, Engine>(m, "LoopEngine")
      .def(py::init<>());
}

TRACTOR_PYTHON_GLOBAL(pythonizeEngine);

}  // namespace tractor
