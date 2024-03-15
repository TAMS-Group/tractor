// 2020-2024 Philipp Ruppel

#include <tractor/core/gradients.h>

#include <tractor/core/allocator.h>
#include <tractor/core/ops.h>
#include <tractor/core/profiler.h>

#include <algorithm>
#include <map>
#include <unordered_map>

#include <boost/container/small_vector.hpp>

namespace tractor {

template <class Ports>
static void packPortOffsets(Ports &&ports) {
  size_t offset = 0;
  for (auto &port : ports) {
    port.offset() = offset;
    offset += port.size();
  }
}

struct SumInfo {
  TypeInfo type_info;
  boost::container::small_vector<uint64_t, 8> arguments;
  bool initialized = false;
  bool finalized = false;
  uint64_t destination = 0;
};

static void buildSumTree(Allocator &alloc, SumInfo &sum_info,
                         std::vector<Program::Instruction> &instructions) {
  TRACTOR_PROFILER("build sum tree");
  auto &type_info = sum_info.type_info;
  if (!sum_info.finalized) {
    if (sum_info.arguments.size() > 1) {
      auto *add_op = Operator::find<compute, op_add>({type_info, type_info});
      while (sum_info.arguments.size() > 1) {
        boost::container::small_vector<uint64_t, 8> args2;
        size_t i = 0;
        for (; i + 1 < sum_info.arguments.size(); i += 2) {
          uintptr_t o = alloc.alloc(type_info);
          instructions.push_back((uintptr_t)add_op);
          instructions.push_back(sum_info.arguments[i + 0]);
          instructions.push_back(sum_info.arguments[i + 1]);
          instructions.push_back(o);
          args2.push_back(o);
        }
        for (; i < sum_info.arguments.size(); i += 1) {
          args2.push_back(sum_info.arguments[i]);
        }
        sum_info.arguments = args2;
      }
    }
    if (sum_info.arguments.size() == 1) {
      auto *move_op = Operator::find<compute, op_move>({type_info});
      instructions.push_back(move_op);
      instructions.push_back(sum_info.arguments[0]);
      instructions.push_back(sum_info.destination);
    }
    sum_info.finalized = true;
    sum_info.arguments.clear();
  }
}

void buildGradients(const Program &src, Program &prep, Program *_fprop,
                    Program *_bprop, Program *_hessian, Program *accumulate) {
  TRACTOR_PROFILER("build gradients");

  Program fprop_dummy(src.context()), bprop_dummy(src.context());
  if (_hessian || accumulate) {
    if (!_fprop) {
      _fprop = &fprop_dummy;
    }
    if (!_bprop) {
      _bprop = &bprop_dummy;
    }
  }

  std::vector<ssize_t> prep_index;
  std::vector<Program::Instruction> prep_insts;
  {
    prep.clear();
    Allocator alloc;
    alloc.keep(src);
    size_t inst_index = 0;
    for (auto &inst : src.instructions()) {
      auto *prep_op = inst.op()->tryFindVariant<prepare>();
      if (prep_op) {
        prep_index.push_back(prep_insts.size());
        prep_insts.emplace_back((uintptr_t)prep_op);
        for (size_t i = 0; i < inst.argumentCount(); i++) {
          prep_insts.emplace_back(inst.arg(i));
        }
        for (size_t i = inst.argumentCount(); i < prep_op->argumentCount();
             i++) {
          prep_insts.emplace_back(alloc.alloc(prep_op->arg(i).typeInfo()));
        }
      } else {
        prep_index.push_back(-1);
      }
      inst_index++;
    }
    alloc.apply(prep);
    prep.setInstructions(prep_insts.begin(), prep_insts.end());
  }

  if (_bprop) {
    TRACTOR_DEBUG("building reverse gradient program");

    auto &bprop = *_bprop;
    bprop.clear();

    for (auto port : src.inputs()) {
      port.binding() = 0;
      port.address() += prep.memorySize();
      port.typeInfo() = port.typeInfo().gradientType();
      bprop.addOutput(Program::Output(port));
    }
    packPortOffsets(bprop.outputs());

    for (auto port : src.outputs()) {
      port.binding() = 0;
      port.address() += prep.memorySize();
      port.typeInfo() = port.typeInfo().gradientType();
      bprop.addInput(Program::Input(port));
    }
    packPortOffsets(bprop.inputs());

    {
      std::vector<Program::Instruction> instructions;
      size_t inst_index = 0;
      for (auto &inst : src.instructions()) {
        auto *op = inst.op()->variant<reverse>();
        if (!op) {
          throw std::runtime_error("no reverse op " + inst.op()->name());
        }
        for (ssize_t i = inst.argumentCount() - 1; i >= 0; i--) {
          auto addr = inst.arg(i) + prep.memorySize();
          instructions.emplace_back(addr);
        }
        ssize_t prep_i = prep_index[inst_index];
        if (prep_i < 0) {
          for (ssize_t i = inst.argumentCount() - 1; i >= 0; i--) {
            instructions.emplace_back(inst.arg(i));
          }
        } else {
          auto &prep_inst = prep_insts[prep_i];
          for (ssize_t i = prep_inst.argumentCount() - 1;
               i >= inst.argumentCount(); i--) {
            instructions.emplace_back(prep_inst.arg(i));
          }
        }
        instructions.emplace_back((uintptr_t)op);
        inst_index++;
      }
      std::reverse(instructions.begin(), instructions.end());
      bprop.setInstructions(instructions.begin(), instructions.end());
      bprop.updateMemorySize(src.memorySize() + prep.memorySize());
    }

    {
      TRACTOR_DEBUG("building reverse gradient sum trees");
      Allocator alloc;
      alloc.keep(bprop);
      std::vector<Program::Instruction> instructions;
      std::unordered_map<uint64_t, SumInfo> sum_tree;
      std::vector<uint64_t> temp_args;
      for (auto &inst : bprop.instructions()) {
        auto *op = inst.op();
        temp_args.clear();
        for (size_t i = 0; i < inst.argumentCount(); i++) {
          auto type_info = op->arg(i).typeInfo();
          auto &sum_info = sum_tree[inst.arg(i)];
          if (!sum_info.initialized) {
            sum_info.type_info = type_info;
            sum_info.destination = inst.arg(i);
            sum_info.initialized = true;
          }
          if (sum_info.type_info != type_info) {
            TRACTOR_FATAL("bprop sum tree data type mismatch "
                          << sum_info.type_info.name() << " "
                          << type_info.name() << " arg " << i << " op "
                          << op->name());
            throw std::runtime_error("bprop sum tree data type mismatch");
          }
          if (op->arg(i).isInput()) {
            buildSumTree(alloc, sum_info, instructions);
            temp_args.push_back(inst.arg(i));
          } else {
            if (sum_info.finalized) {
              throw std::runtime_error("bprop write after read");
            }
            uint64_t addr = alloc.alloc(type_info);
            temp_args.push_back(addr);
            sum_info.arguments.push_back(addr);
          }
        }
        instructions.push_back((uintptr_t)op);
        for (auto &arg : temp_args) {
          instructions.push_back(arg);
        }
      }
      for (auto &p : sum_tree) {
        buildSumTree(alloc, p.second, instructions);
      }
      alloc.apply(bprop);
      bprop.setInstructions(instructions.begin(), instructions.end());
    }

    {
      std::vector<Program::Instruction> instructions;
      std::unordered_set<size_t> input_set;
      for (auto &input : bprop.inputs()) {
        input_set.insert(input.address());
      }
      for (auto &inst : bprop.instructions()) {
        auto *op = inst.op();
        for (size_t i = 0; i < inst.argumentCount(); i++) {
          if (inst.arg(i) >= prep.memorySize()) {
            if (input_set.insert(inst.arg(i)).second) {
              if (op->arg(i).isInput()) {
                auto *zero_op =
                    Operator::find<compute, op_zero>({op->arg(i).typeInfo()});
                instructions.emplace_back(zero_op);
                instructions.emplace_back(inst.arg(i));
              }
            }
          }
        }
      }
      for (auto &code : bprop.code()) {
        instructions.emplace_back(code);
      }
      bprop.setInstructions(instructions.begin(), instructions.end());
    }
  }

  if (_fprop) {
    auto &fprop = *_fprop;
    fprop.clear();

    for (auto port : src.inputs()) {
      port.binding() = 0;
      port.address() += prep.memorySize();
      port.typeInfo() = port.typeInfo().gradientType();
      fprop.addInput(port);
    }
    packPortOffsets(fprop.inputs());

    for (auto port : src.outputs()) {
      port.binding() = 0;
      port.address() += prep.memorySize();
      port.typeInfo() = port.typeInfo().gradientType();
      fprop.addOutput(port);
    }
    packPortOffsets(fprop.outputs());

    fprop.updateMemorySize(src.memorySize() + prep.memorySize());

    {
      for (auto &port : src.constants()) {
        auto *zero_op =
            Operator::find<compute, op_zero>({port.typeInfo().gradientType()});
        fprop.addCode(zero_op);
        fprop.addCode(port.address() + prep.memorySize());
      }
      for (auto &port : src.parameters()) {
        auto *zero_op =
            Operator::find<compute, op_zero>({port.typeInfo().gradientType()});
        fprop.addCode(zero_op);
        fprop.addCode(port.address() + prep.memorySize());
      }
    }

    size_t inst_index = 0;
    for (auto &inst : src.instructions()) {
      auto *op = inst.op()->variant<forward>();
      if (!op) {
        throw std::runtime_error("no forward op " + inst.op()->name());
      }
      fprop.addCode((uintptr_t)op);
      ssize_t prep_i = prep_index[inst_index];
      if (prep_i < 0) {
        for (size_t i = 0; i < inst.argumentCount(); i++) {
          fprop.addCode(inst.arg(i));
        }
      } else {
        auto &prep_inst = prep_insts[prep_i];
        for (size_t i = inst.argumentCount(); i < prep_inst.argumentCount();
             i++) {
          fprop.addCode(prep_inst.arg(i));
        }
      }
      for (ssize_t i = 0; i < inst.argumentCount(); i++) {
        auto addr = inst.arg(i) + prep.memorySize();
        fprop.addCode(addr);
      }
      inst_index++;
    }
  }

  if (_hessian) {
    auto &hprop = *_hessian;
    hprop.clear();
    const auto &fprop = *_fprop;
    const auto &bprop = *_bprop;
    for (auto &port : fprop.inputs()) {
      hprop.addInput(port);
    }
    for (auto &port : bprop.outputs()) {
      hprop.addOutput(port);
    }
    for (auto &inst : fprop.code()) {
      hprop.addCode(inst);
    }
    for (auto &inst : bprop.code()) {
      hprop.addCode(inst);
    }
    hprop.updateMemorySize(std::max(bprop.memorySize(), fprop.memorySize()));
  }

  if (accumulate) {
    const auto &fprop = *_fprop;
    auto &accu = *accumulate;
    accu.clear();
    Allocator alloc;
    alloc.keep(src);
    std::vector<Program::Input> aa;
    std::vector<Program::Input> bb;
    std::vector<Program::Output> xx;
    for (auto port : src.inputs()) {
      port.address() = alloc.alloc(port.typeInfo());
      aa.push_back(port);
      accu.addInput(port);
    }
    for (auto port : fprop.inputs()) {
      port.address() = alloc.alloc(port.typeInfo());
      bb.push_back(port);
      accu.addInput(port);
    }
    packPortOffsets(accu.inputs());
    for (auto port : src.inputs()) {
      port.address() = alloc.alloc(port.typeInfo());
      xx.push_back(Program::Output(port));
      accu.addOutput(Program::Output(port));
    }
    packPortOffsets(accu.outputs());
    for (size_t i = 0; i < aa.size(); i++) {
      auto *add_op =
          Operator::find<compute, op_add>({aa[i].typeInfo(), bb[i].typeInfo()});
      accu.addCode(add_op);
      accu.addCode(aa[i].address());
      accu.addCode(bb[i].address());
      accu.addCode(xx[i].address());
    }
    alloc.apply(accu);
  }
}

void buildConstraints(const Program &prog, const Program &fprop,
                      const Program &bprop, const Program &hprop,
                      const TypeInfo &padding_type, Program *proj,
                      Program *barrier_init, Program *barrier_step,
                      Program *barrier_diagonal, Program *penalty_init,
                      Program *penalty_step, Program *penalty_diagonal) {
  proj->clear();

  barrier_init->clear();
  barrier_step->clear();
  barrier_diagonal->clear();

  penalty_init->clear();
  penalty_step->clear();
  penalty_diagonal->clear();

  Allocator alloc;
  alloc.init(std::max(prog.memorySize(),
                      std::max(std::max(fprop.memorySize(), bprop.memorySize()),
                               hprop.memorySize())));

  struct PortInfo {
    TypeInfo type;
    uintptr_t input = 0;
    uintptr_t output = 0;
    PortInfo() {}
    PortInfo(TypeInfo type, uintptr_t input, uintptr_t output)
        : type(type), input(input), output(output) {}
  };

  std::unordered_map<uintptr_t, PortInfo> port_map;

  {
    auto it_nonlinear = prog.inputs().begin();
    auto it_linear = fprop.inputs().begin();
    while (it_nonlinear != prog.inputs().end()) {
      proj->addInput(*it_linear);

      barrier_init->addInput(*it_linear);
      barrier_step->addInput(*it_linear);

      uintptr_t out_addr = alloc.alloc(it_linear->typeInfo());

      port_map[it_nonlinear->address()] =
          PortInfo(it_linear->typeInfo(), it_linear->address(), out_addr);
      auto output = Program::Output(it_linear->typeInfo(), out_addr,
                                    it_linear->offset(), 0);
      proj->addOutput(output);

      barrier_init->addOutput(output);
      barrier_step->addOutput(output);
      barrier_diagonal->addOutput(output);

      ++it_nonlinear;
      ++it_linear;
    }
  }

  auto padding_addr = alloc.alloc(padding_type);

  proj->addParameter(Program::Parameter(padding_type, padding_addr, 0, 0));

  for (auto &inst : prog.instructions()) {
    auto *op_proj = inst.op()->tryFindVariant<project>();

    auto *op_barrier_init = inst.op()->tryFindVariant<tractor::barrier_init>();
    auto *op_barrier_step = inst.op()->tryFindVariant<tractor::barrier_step>();
    auto *op_barrier_diagonal =
        inst.op()->tryFindVariant<tractor::barrier_diagonal>();

    if (!op_proj && !op_barrier_init && !op_barrier_step &&
        !op_barrier_diagonal) {
      continue;
    }

    if (!op_proj || !op_barrier_init || !op_barrier_step ||
        !op_barrier_diagonal) {
      throw std::runtime_error("incomplete constraint op " + inst.op()->name());
    }

    auto it_port_info = port_map.find(inst.arg(0));
    if (it_port_info == port_map.end()) {
      throw std::runtime_error("add slack variable");
    }
    auto &port_info = it_port_info->second;

    {
      proj->addCode(op_proj);
      for (size_t i = 0; i < inst.argumentCount(); i++) {
        if (inst.op()->arg(i).isInput()) {
          proj->addCode(inst.arg(i));
        }
      }
      proj->addCode(port_info.input);
      proj->addCode(padding_addr);
      proj->addCode(port_info.output);
    }

    auto temp_addr = alloc.alloc(
        op_barrier_init->arg(op_barrier_init->argumentCount() - 1).typeInfo());

    {
      barrier_init->addCode(op_barrier_init);
      for (size_t i = 0; i < inst.argumentCount(); i++) {
        if (inst.op()->arg(i).isInput()) {
          barrier_init->addCode(inst.arg(i));
        }
      }
      barrier_init->addCode(port_info.input);
      barrier_init->addCode(port_info.output);
      barrier_init->addCode(temp_addr);

      barrier_step->addCode(op_barrier_step);
      barrier_step->addCode(temp_addr);
      barrier_step->addCode(port_info.input);
      barrier_step->addCode(port_info.output);

      barrier_diagonal->addCode(op_barrier_diagonal);
      barrier_diagonal->addCode(temp_addr);
      barrier_diagonal->addCode(port_info.output);
    }

    port_map.erase(it_port_info);
  }

  for (auto &port_info_pair : port_map) {
    auto &port_info = port_info_pair.second;

    {
      auto *move_op = Operator::find<compute, op_move>({port_info.type});
      proj->addCode(move_op);
      proj->addCode(port_info.input);
      proj->addCode(port_info.output);
    }

    {
      auto *zero_op = Operator::find<compute, op_zero>({port_info.type});

      {
        barrier_init->addCode(zero_op);
        barrier_init->addCode(port_info.output);

        barrier_step->addCode(zero_op);
        barrier_step->addCode(port_info.output);

        barrier_diagonal->addCode(zero_op);
        barrier_diagonal->addCode(port_info.output);
      }
    }
  }

  alloc.apply(*proj);
  alloc.apply(*barrier_init);
  alloc.apply(*barrier_step);
  alloc.apply(*barrier_diagonal);
}

}  // namespace tractor
