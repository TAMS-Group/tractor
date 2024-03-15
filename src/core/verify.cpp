// 2020-2024 Philipp Ruppel

#include <tractor/core/verify.h>

#include <tractor/core/allocator.h>
#include <tractor/core/operator.h>
#include <tractor/core/profiler.h>
#include <tractor/core/program.h>

#include <map>

namespace tractor {

void verify(const Program &program) {
  TRACTOR_PROFILER("verify program");
  TRACTOR_DEBUG("verify program start");
  checkMemory(program);
  checkMemory2(program);
  TRACTOR_DEBUG("verify program finished");
}

class MemoryChecker2 {
  typedef std::map<uintptr_t, TypeInfo> Map;
  typedef typename Map::iterator Iterator;
  Map _map;
  bool _doRangesOverlap(uintptr_t a_start, uintptr_t a_length,
                        uintptr_t b_start, uintptr_t b_length) {
    if (a_start >= b_start + b_length) {
      return false;
    }
    if (b_start >= a_start + a_length) {
      return false;
    }
    return true;
  }
  bool _erase(const Iterator &it, uintptr_t address, const TypeInfo &type) {
    if (it != _map.end()) {
      if (_doRangesOverlap(it->first, it->second.size(), address,
                           type.size())) {
        _map.erase(it);
        return true;
      }
    }
    return false;
  }
  Iterator _prev(Iterator it) {
    if (it != _map.end() && it != _map.begin()) {
      it--;
      return it;
    } else {
      return _map.end();
    }
  }

 public:
  void write(uintptr_t address, const TypeInfo &type) {
    while (_erase(_map.lower_bound(address), address, type)) {
    }
    _erase(_prev(_map.lower_bound(address)), address, type);
    _map[address] = type;
  }
  template <class F>
  void read(uintptr_t address, const TypeInfo &type, const F &f) {
    auto it = _map.find(address);
    if (it == _map.end()) {
      throw std::runtime_error(
          std::string() + "memcheck2 read from uninitialized memory address " +
          f());
    }
    if (it->second != type) {
      throw std::runtime_error(
          std::string() + "memcheck2 type mismatch output:" +
          it->second.name() + " input:" + type.name() + " " + f());
    }
  }
};

void checkMemory2(const Program &program) {
  TRACTOR_PROFILER("memcheck2");
  TRACTOR_DEBUG("memcheck2 begin");
  MemoryChecker2 chk;
  for (auto &port : program.inputs()) {
    chk.write(port.address(), port.typeInfo());
  }
  for (auto &port : program.constants()) {
    chk.write(port.address(), port.typeInfo());
  }
  for (auto &port : program.parameters()) {
    chk.write(port.address(), port.typeInfo());
  }
  for (auto &inst : program.instructions()) {
    for (size_t iarg = 0; iarg < inst.op()->argumentCount(); iarg++) {
      if (inst.op()->arg(iarg).isInput()) {
        chk.read(inst.arg(iarg), inst.op()->arg(iarg).typeInfo(), [&]() {
          return std::string() + "op:" + inst.op()->name() +
                 " arg:" + std::to_string(iarg);
        });
      } else {
        chk.write(inst.arg(iarg), inst.op()->arg(iarg).typeInfo());
      }
    }
  }
  for (auto &port : program.outputs()) {
    chk.read(port.address(), port.typeInfo(), [&]() { return "output"; });
  }
  TRACTOR_DEBUG("memcheck2 passed");
}

void checkMemory(const Program &program) {
  TRACTOR_PROFILER("memcheck1");

  TRACTOR_DEBUG("checking memory");

  TRACTOR_DEBUG("memory size " << program.memorySize());
  std::vector<uint8_t> memory;
  memory.resize(program.memorySize(), 0);

  for (auto &port : program.inputs()) {
    memory.at(port.address()) = 1;
  }

  for (auto &port : program.constants()) {
    memory.at(port.address()) = 1;
  }

  for (auto &port : program.parameters()) {
    memory.at(port.address()) = 1;
  }

  for (auto &inst : program.instructions()) {
    for (size_t iarg = 0; iarg < inst.op()->argumentCount(); iarg++) {
      if ((inst.arg(iarg) % inst.op()->arg(iarg).typeInfo().alignment()) != 0) {
        // throw std::runtime_error("memory check: alignment error");
        throw std::runtime_error("memory check: alignment error " +
                                 inst.op()->name() + " " +
                                 std::to_string(iarg) + " " +
                                 std::to_string(inst.op()->arg(iarg).size()) +
                                 " " + std::to_string(inst.arg(iarg)));
      }
      if (inst.op()->arg(iarg).isInput()) {
        if (!memory.at(inst.arg(iarg))) {
          for (auto &inst2 : program.instructions()) {
            TRACTOR_DEBUG("op " << inst2.op()->name());
            for (size_t iarg = 0; iarg < inst2.op()->argumentCount(); iarg++) {
              TRACTOR_DEBUG("arg " << inst2.arg(iarg) << ":"
                                   << inst2.op()->arg(iarg).size());
            }
            if (&inst2 == &inst) {
              break;
            }
          }
          for (auto &port : program.inputs()) {
            TRACTOR_DEBUG("input " << port.address() << " " << port.size());
          }
          for (auto &port : program.constants()) {
            TRACTOR_DEBUG("constant " << port.address() << " " << port.size());
          }
          for (auto &port : program.parameters()) {
            TRACTOR_DEBUG("parameter " << port.address() << " " << port.size());
          }
          throw std::runtime_error(
              "parameter read from uninitialized memory z " +
              inst.op()->name() + " " + std::to_string(iarg) + " " +
              std::to_string(inst.op()->arg(iarg).size()) + " " +
              std::to_string(inst.arg(iarg)));
        }
      }
      if (inst.op()->arg(iarg).isOutput()) {
        memory.at(inst.arg(iarg)) = 1;
      }
    }
  }

  for (auto &port : program.outputs()) {
    if (!memory.at(port.address())) {
      throw std::runtime_error("output read from uninitialized memory");
    }
  }
}

}  // namespace tractor
