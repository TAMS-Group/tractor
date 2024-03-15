// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/eigen.h>
#include <tractor/core/gradients.h>
#include <tractor/core/log.h>
#include <tractor/core/profiler.h>
#include <tractor/core/program.h>
#include <tractor/core/solver.h>

#define TRACTOR_STRINGIFY_2(x) TRACTOR_STRINGIFY(x)

#define TRACTOR_CHECK_FINITE(x)                                        \
  if (!std::isfinite(x)) {                                             \
    throw std::runtime_error(TRACTOR_STRINGIFY(                        \
        x) " not finite " __FILE__ ":" TRACTOR_STRINGIFY_2(__LINE__)); \
  }

#define TRACTOR_CHECK_ALL_FINITE(x)           \
  {                                           \
    for (size_t i = 0; i < (x).size(); i++) { \
      TRACTOR_CHECK_FINITE((x)[i]);           \
    }                                         \
  }

#if 1
#define TRACTOR_LOG_VAR(x) TRACTOR_DEBUG(TRACTOR_STRINGIFY_2(x) << ": " << x);
#else
#define TRACTOR_LOG_VAR(x)
#endif

#if 0
#define TRACTOR_LOG_VEC(x) \
  TRACTOR_DEBUG(TRACTOR_STRINGIFY_2(x)) << x << std::endl << std::endl;
#else
#define TRACTOR_LOG_VEC(x)
#endif

namespace tractor {

class SolverBase : public Solver {
  Buffer _accu_in, _accu_grad;

 protected:
  Program _p_prog, _p_prep, _p_fprop, _p_bprop, _p_hprop, _p_accu;
  std::shared_ptr<Executable> _x_prog, _x_prep, _x_fprop, _x_bprop, _x_hprop,
      _x_accu;
  std::shared_ptr<Memory> _memory;
  std::shared_ptr<Executable> _x_project;
  std::shared_ptr<Executable> _x_barrier_init, _x_barrier_step,
      _x_barrier_diagonal;
  std::shared_ptr<Executable> _x_penalty_init, _x_penalty_step,
      _x_penalty_diagonal;
  Program _p_project;
  Program _p_barrier_init, _p_barrier_step, _p_barrier_diagonal;
  Program _p_penalty_init, _p_penalty_step, _p_penalty_diagonal;
  std::vector<int> _priority_list;
  std::vector<size_t> _constraint_indices;

  template <class Scalar>
  class RegularizedMatrixReplacement
      : public MatrixReplacement<Scalar>::ExecutableImpl {
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    Scalar _regularization = Scalar(0);

   public:
    virtual void mul(const Vector &input, Vector &output) const override {
      TRACTOR_PROFILER("hprop");
      TRACTOR_CHECK_ALL_FINITE(input);
      MatrixReplacement<Scalar>::ExecutableImpl::mul(input, output);
      TRACTOR_CHECK_ALL_FINITE(output);
      if (_regularization > Scalar(0)) {
        output.array() += input.array() * _regularization;
      }
      TRACTOR_CHECK_ALL_FINITE(output);
    }
    void setRegularization(const Scalar &regularization) {
      _regularization = regularization;
    }
  };

  void _compileGradients(const Program &prog, const TypeInfo &type);

  template <class Scalar>
  void _compileGradients(const Program &prog) {
    _compileGradients(prog, TypeInfo::get<Scalar>());
  }

  template <class Pos, class Grad>
  void accumulate(Pos &pos, const Grad &grad) {
    _accu_in.fromVector(pos);
    _accu_grad.fromVector(grad);
    _accu_in.append(_accu_grad);
    _x_accu->input(_accu_in, _memory);
    _x_accu->execute(_memory);
    _x_accu->outputVector(_memory, pos);
  }

  SolverBase(const std::shared_ptr<Engine> &engine);
};

}  // namespace tractor
