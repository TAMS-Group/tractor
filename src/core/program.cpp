// 2020-2024 Philipp Ruppel

#include <tractor/core/program.h>

#include <tractor/core/log.h>
#include <tractor/core/operator.h>
#include <tractor/core/recorder.h>
#include <tractor/core/simplify.h>
#include <tractor/core/verify.h>

namespace tractor {

size_t Program::Instruction::argumentCount() const {
  return op()->argumentCount();
}

template <class T>
void printBuffer(std::ostream &stream, const char *label, const T &data) {
  std::string str((char *)&*data.begin(), (char *)&*data.end());
  stream << label << " " << data.size() << " " << std::hash<std::string>()(str)
         << std::endl;
}

template <class T>
void printPorts(std::ostream &stream, const char *label, const T &data) {
  printBuffer(stream, label, data);
  for (auto &port : data) {
    stream << "  - " << (const void *)port.address() << " " << port.name()
           << std::endl;
  }
}

void Program::updateMemorySize(size_t s) {
  TRACTOR_DEBUG("update memory size " << s);
  _memory_size = s;
}

void Program::clear() {
  _memory_size = 0;
  _instructions.clear();
  _inputs.clear();
  _outputs.clear();
  _constants.clear();
  _const_data.clear();
  _goals.clear();
  _context.reset();
  _bound_data.clear();
}

void Program::record(const std::function<void()> &function, bool _simplify) {
  _context = nullptr;
  {
    struct RecorderImpl : Recorder {
      RecorderImpl(Program *prog) : Recorder(prog) {}
    };
    RecorderImpl rec(this);
    // try {
    function();
    // } catch (const std::exception &ex) {
    //   TRACTOR_DEBUG(ex.what());
    //   throw;
    // }
    if (Recorder::instance() != &rec) {
      throw std::runtime_error("recorder not active anymore");
    }
  }

  TRACTOR_INFO("prog rec verify");
  verify(*this);

  if (_simplify) {
    TRACTOR_INFO("prog rec simplify");
    simplify(*this);

    TRACTOR_INFO("prog rec verify");
    verify(*this);
  }

  TRACTOR_INFO("prog rec ready");
}

std::ostream &operator<<(std::ostream &stream, const Program &prog) {
  // printPorts(stream, "inputs", prog.inputs());
  // printPorts(stream, "parameters", prog.parameters());
  // printPorts(stream, "outputs", prog.outputs());
  // printBuffer(stream, "goals", prog.goals());
  // printPorts(stream, "constants", prog.constants());
  // printBuffer(stream, "constdata", prog.constData());
  // printBuffer(stream, "code", prog.code());

  for (auto &port : prog.inputs()) {
    stream << "input address:" << (void *)port.address()
           << " offset:" << (void *)port.offset()
           << " binding:" << (void *)port.binding() << " size:" << port.size()
           << " type:" << port.typeInfo().name() << std::endl;
  }

  for (auto &port : prog.parameters()) {
    stream << "parameter address:" << (void *)port.address()
           << " offset:" << (void *)port.offset()
           << " port:" << (void *)port.binding() << " binding:" << port.size()
           << " type:" << port.typeInfo().name() << std::endl;
  }

  for (auto &port : prog.outputs()) {
    stream << "output address:" << (void *)port.address()
           << " offset:" << (void *)port.offset()
           << " binding:" << (void *)port.binding() << " size:" << port.size()
           << " type:" << port.typeInfo().name() << std::endl;
  }

  for (auto &goal : prog.goals()) {
    stream << "goal port:" << goal.port() << " priority:" << goal.priority()
           << std::endl;
  }

  for (auto &port : prog.constants()) {
    stream << "constant address:" << (void *)port.address()
           << " port:" << port.size() << " offset:" << port.offset()
           << " type:" << port.typeInfo().name();
    if (port.typeInfo() == TypeInfo::get<double>()) {
      stream << " value:"
             << *(double *)(prog.constData().data() + port.offset());
    }
    stream << std::endl;
  }

  stream << "memory size " << prog.memorySize() << std::endl;

  for (auto &inst : prog.instructions()) {
    stream << "instruction ";
    stream << inst.op()->name();
    for (auto arg : inst.args()) {
      stream << " " << (void *)arg;
    }
    stream << std::endl;
  }

  return stream;
}

}  // namespace tractor
