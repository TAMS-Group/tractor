// 2020-2024 Philipp Ruppel

#include <tractor/solvers/base.h>

#include <tractor/core/log.h>

namespace tractor {

void SolverBase::_compileGradients(const Program &prog, const TypeInfo &type) {
  _p_prog = prog;
  _x_prog->compile(_p_prog);

  buildGradients(_p_prog, _p_prep, &_p_fprop, &_p_bprop, &_p_hprop, &_p_accu);
  _x_prep->compile(_p_prep);
  _x_fprop->compile(_p_fprop);
  _x_bprop->compile(_p_bprop);
  _x_hprop->compile(_p_hprop);
  _x_accu->compile(_p_accu);

  buildConstraints(_p_prog, _p_fprop, _p_bprop, _p_hprop, type, &_p_project,
                   &_p_barrier_init, &_p_barrier_step, &_p_barrier_diagonal,
                   &_p_penalty_init, &_p_penalty_step, &_p_penalty_diagonal);

  _x_project->compile(_p_project);

  _x_barrier_init->compile(_p_barrier_init);
  _x_barrier_step->compile(_p_barrier_step);
  _x_barrier_diagonal->compile(_p_barrier_diagonal);

  _x_penalty_init->compile(_p_penalty_init);
  _x_penalty_step->compile(_p_penalty_step);
  _x_penalty_diagonal->compile(_p_penalty_diagonal);

  _constraint_indices.clear();
  _priority_list.clear();
  for (auto &goal : _p_prog.goals()) {
    auto &output = _p_fprop.output(goal.port());
    if (output.size() % type.size() != 0) {
      throw std::runtime_error("output type mismatch");
    }
    for (size_t element = 0; element < output.size() / type.size(); element++) {
      if (goal.priority() == 1) {
        _constraint_indices.push_back(output.offset() / type.size() + element);
      }
      _priority_list.emplace_back(goal.priority());
    }
  }
  TRACTOR_LOG_VAR(_constraint_indices.size());
}

SolverBase::SolverBase(const std::shared_ptr<Engine> &engine) : Solver(engine) {
  _memory = engine->createMemory();
  _x_accu = engine->createExecutable();
  _x_barrier_diagonal = engine->createExecutable();
  _x_barrier_init = engine->createExecutable();
  _x_barrier_step = engine->createExecutable();
  _x_bprop = engine->createExecutable();
  _x_fprop = engine->createExecutable();
  _x_hprop = engine->createExecutable();
  _x_penalty_diagonal = engine->createExecutable();
  _x_penalty_init = engine->createExecutable();
  _x_penalty_step = engine->createExecutable();
  _x_prep = engine->createExecutable();
  _x_prog = engine->createExecutable();
  _x_project = engine->createExecutable();
}

}  // namespace tractor
