// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/matrix.h>
#include <tractor/solvers/base.h>

namespace tractor {

template <class Scalar>
class InteriorPointSolver : public SolverBase {
 public:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> _m_fprop, _m_bprop;

  Scalar _initial_barrier_weight = Scalar(1);
  Scalar _min_barrier_weight = Scalar(1e-3);
  Scalar _current_barrier_weight_start = Scalar(0);
  Scalar _current_barrier_weight = Scalar(0);

  Scalar _barrier_decrease = Scalar(0.5);
  Scalar _barrier_increase = Scalar(2);

  Scalar _constraint_padding = Scalar(0.1);

  Vector _qp_residuals;
  Vector _nonlinear_solution;
  Vector _previous_nonlinear_solution;
  Vector _vec_right;
  Vector _diag_in, _diag_out, _objective_diagonal;
  Vector _line_search_temp_solution;
  Vector _line_search_temp_residuals;
  Vector _line_search_temp_gradients;
  Vector _qp_solution;
  Vector _step_solution;
  Vector _step_solution_test;
  Vector _v_project, _v_barrier_input, _temp_residuals, _v_barrier_gradients,
      _v_barrier_diagonal, _step_residuals, _temp_residuals_2;
  Vector _current_step, _previous_step;

  Scalar _current_regularization = 0;

  size_t _primal_variable_count = 0;
  size_t _dual_variable_count = 0;

  bool _use_matrices = false;
  bool _use_barrier = true;
  bool _use_objectives = true;
  bool _use_constraints = true;
  bool _use_penalty = false;
  bool _in_constraint_phase = false;

  bool _use_preconditioner = true;

  Scalar _computeBarrierWeight() const {
    if (_current_barrier_weight < 0) {
      return 0;
    }
    if (_current_barrier_weight > Scalar(1)) {
      return std::sqrt(_current_barrier_weight);
    } else {
      return 1;
    }
  }
  Scalar _computeObjectiveWeight() const {
    if (_in_constraint_phase || !_use_objectives) {
      return 0;
    }
    if (_current_barrier_weight < 0) {
      return 0;
    }
    if (_current_barrier_weight > Scalar(1)) {
      return Scalar(1) / std::sqrt(_current_barrier_weight);
    } else {
      return Scalar(1) / _current_barrier_weight;
    }
  }

  template <class Input, class Output>
  void _fprop(Input &&input, Output &&output) {
    if (_use_matrices) {
      output = _m_fprop * input;
    } else {
      _x_fprop->run(input, _memory, output);
    }
  }

  template <class Input, class Output>
  void _bprop(Input &&input, Output &&output) {
    if (_use_matrices) {
      output = _m_bprop * input;
    } else {
      _x_bprop->run(input, _memory, output);
    }
  }

  Vector _dualprop_temp, _dualprop_temp_2;
  template <class Input, class Output>
  void _dualprop(Input &&input, Output &&output) {
    TRACTOR_CHECK_ALL_FINITE(input);

    Scalar objective_weight = _computeObjectiveWeight();
    Scalar barrier_weight = _computeBarrierWeight();

    output.setZero(_dual_variable_count);

    _fprop(input.head(_primal_variable_count), _dualprop_temp);

    if (_use_constraints) {
      for (size_t i = 0; i < _constraint_indices.size(); i++) {
        size_t j = _constraint_indices[i];
        output[_primal_variable_count + i] = _dualprop_temp[j];
      }
    }

    _dualprop_temp *= objective_weight;

    if (_use_constraints) {
      for (size_t i = 0; i < _constraint_indices.size(); i++) {
        size_t j = _constraint_indices[i];
        _dualprop_temp[j] += input[_primal_variable_count + i];
      }
    }

    _bprop(_dualprop_temp, output.head(_primal_variable_count));

    if (_use_barrier) {
      _x_barrier_step->run(input.head(_primal_variable_count), _memory,
                           _dualprop_temp);
      output.head(_primal_variable_count) += _dualprop_temp * barrier_weight;
    }

    output.head(_primal_variable_count) +=
        input.head(_primal_variable_count) * _current_regularization;
  }

  template <class Output>
  void _dualdiagonal(Output &&output) {
    if (_use_preconditioner) {
      Scalar objective_weight = _computeObjectiveWeight();
      Scalar barrier_weight = _computeBarrierWeight();

      output.setZero(_dual_variable_count);

      output.head(_primal_variable_count) =
          _objective_diagonal * objective_weight;

      if (_use_barrier) {
        _x_barrier_diagonal->execute(_memory);
        _x_barrier_diagonal->outputVector(_memory, _dualprop_temp);
        output.head(_primal_variable_count) += _dualprop_temp * barrier_weight;
      }

      output.head(_primal_variable_count).array() += _current_regularization;

    } else {
      output.setOnes(_dual_variable_count);
    }
  }

  Vector _dualres_temp;
  template <class Input, class Output>
  void _dualres(Input &&input, Output &&output) {
    Scalar objective_weight = _computeObjectiveWeight();
    Scalar barrier_weight = _computeBarrierWeight();

    output = -_qp_residuals;

    output.head(_primal_variable_count) *= objective_weight;

    if (!_use_constraints) {
      for (size_t i = 0; i < _constraint_indices.size(); i++) {
        output[_primal_variable_count + i] = 0;
      }
    }

    if (_use_barrier) {
      _x_barrier_init->run(input.head(_primal_variable_count), _memory,
                           _dualres_temp);
      output.head(_primal_variable_count) -= _dualres_temp * barrier_weight;

      _x_barrier_step->run(input.head(_primal_variable_count), _memory,
                           _dualres_temp);
      output.head(_primal_variable_count) += _dualres_temp * barrier_weight;
    }

    output.head(_primal_variable_count) +=
        input.head(_primal_variable_count) * _current_regularization;
  }

  struct DualMatrixReplacement : public MatrixReplacement<Scalar>::Impl {
    InteriorPointSolver *_solver = nullptr;
    mutable Vector _diagonal_temp;
    virtual size_t rows() const override {
      return _solver->_dual_variable_count;
    }
    virtual size_t cols() const override {
      return _solver->_dual_variable_count;
    }
    virtual void mul(const Vector &input, Vector &output) const override {
      _solver->_dualprop(input, output);
    }
    virtual const Vector &diagonal() const override {
      _solver->_dualdiagonal(_diagonal_temp);
      return _diagonal_temp;
    }
    DualMatrixReplacement(InteriorPointSolver *solver) : _solver(solver) {}
  };

  struct DiagonalPreconditioner {
    Vector _inv_diag;
    bool _use_preconditioner = false;
    typedef typename Vector::StorageIndex StorageIndex;
    typedef typename Vector::Index Index;
    enum {
      ColsAtCompileTime = Eigen::Dynamic,
      MaxColsAtCompileTime = Eigen::Dynamic
    };
    Index rows() const { return _inv_diag.size(); }
    Index cols() const { return _inv_diag.size(); }
    DiagonalPreconditioner() {}
    DiagonalPreconditioner(const MatrixReplacement<Scalar> &m) {}
    auto &analyzePattern(const MatrixReplacement<Scalar> &m) { return *this; }
    auto &factorize(const MatrixReplacement<Scalar> &m) {
      auto &diagonal = m.diagonal();
      _inv_diag.resize(diagonal.size());
      for (size_t i = 0; i < _inv_diag.size(); i++) {
        Scalar v = diagonal[i];
        if (v != Scalar(0)) {
          v = Scalar(1) / v;
        } else {
          v = Scalar(1);
        }
        _inv_diag[i] = v;
      }
      if (!_inv_diag.allFinite()) {
        throw std::runtime_error("preconditioner not finite");
      }
      return *this;
    }
    auto &compute(const MatrixReplacement<Scalar> &m) { return factorize(m); }
    template <class A, class X>
    void _solve_impl(const A &a, X &x) const {
      if (_use_preconditioner) {
        x.array() = _inv_diag.array() * a.array();
      } else {
        x = a;
      }
    }
    template <class A>
    auto solve(const Eigen::MatrixBase<A> &a) const {
      return Eigen::Solve<DiagonalPreconditioner, A>(*this, a.derived());
    }
    Eigen::ComputationInfo info() const { return Eigen::Success; }
  };

  std::shared_ptr<DualMatrixReplacement> _dual_matrix_replacement;
  MatrixReplacement<Scalar> _hgrad;
  typedef Eigen::ConjugateGradient<MatrixReplacement<Scalar>,
                                   Eigen::Lower | Eigen::Upper,
                                   DiagonalPreconditioner>
      LinearSolver;
  LinearSolver _linear_solver;

 protected:
  virtual void _compile(const Program &prog) override {
    _compileGradients<Scalar>(prog);

    _primal_variable_count = _x_fprop->inputBufferSize() / sizeof(Scalar);
    TRACTOR_LOG_VAR(_primal_variable_count);

    _dual_variable_count = _primal_variable_count + _constraint_indices.size();
    TRACTOR_LOG_VAR(_dual_variable_count);
    TRACTOR_DEBUG(_p_accu);
  }

  virtual void _input(const Buffer &buffer) override {
    buffer.toVector(_nonlinear_solution);
    _current_barrier_weight_start = _initial_barrier_weight;
  }

  virtual void _output(Buffer &buffer) override {
    buffer.fromVector(_nonlinear_solution);
  }

  virtual void _parameterize(const Buffer &buffer) override {
    _x_prog->parameterize(buffer, _memory);
  }

  virtual double _step() override {
    TRACTOR_PROFILER("step");

    _linear_solver.setTolerance(tolerance() * 0.1);

    _current_regularization = 0;

    {
      TRACTOR_PROFILER("nl program");
      _x_prog->run(_nonlinear_solution, _memory, _vec_right);
    }
    {
      TRACTOR_PROFILER("prepare");
      _x_prep->execute(_memory);
    }
    {
      TRACTOR_PROFILER("residual");
      _qp_residuals.setZero(_dual_variable_count);
      _x_bprop->run(_vec_right, _memory,
                    _qp_residuals.head(_primal_variable_count));
      for (size_t i = 0; i < _constraint_indices.size(); i++) {
        size_t j = _constraint_indices[i];
        _qp_residuals[_primal_variable_count + i] = _vec_right[j];
      }
    }

    if (_use_matrices) {
      buildGradientMatrix(_x_fprop, _memory, _m_fprop);
      buildGradientMatrix(_x_bprop, _memory, _m_bprop);
      if ((_m_fprop - _m_bprop.transpose()).squaredNorm() > 1e-12) {
        throw std::runtime_error("jacobians inconsistent");
      }
    }

    if (_use_preconditioner) {
      _diag_in.setZero(_primal_variable_count);
      _objective_diagonal.setZero(_primal_variable_count);
      {
        TRACTOR_PROFILER("preconditioning2");
        for (size_t i = 0; i < _primal_variable_count; i++) {
          _diag_in[i] = Scalar(1);
          _x_fprop->run(_diag_in, _memory, _diag_out);
          Scalar v = _diag_out.dot(_diag_out);
          _objective_diagonal[i] = v;
          _diag_in[i] = Scalar(0);
        }
      }
    } else {
      _objective_diagonal.setZero(_qp_residuals.size());
    }

    _qp_solution.setZero(_qp_residuals.size());
    _step_solution = _qp_solution;
    _previous_step = _qp_solution;

    {
      TRACTOR_PROFILER("project inequality constraints");
      Vector padding_vec;
      padding_vec.resize(1);
      padding_vec[0] = _constraint_padding;
      _x_project->parameterVector(padding_vec, _memory);

      _x_project->run(_qp_solution.head(_primal_variable_count), _memory,
                      _qp_solution.head(_primal_variable_count));
      TRACTOR_CHECK_ALL_FINITE(_qp_solution);
    }

    _current_barrier_weight = _current_barrier_weight_start;

    size_t iteration_count = 0;

    _in_constraint_phase = false;

    if (1) {
      while (_current_barrier_weight >= _min_barrier_weight * Scalar(0.99)) {
        TRACTOR_LOG_VAR(iteration_count);
        TRACTOR_LOG_VAR(_current_barrier_weight);

        TRACTOR_CHECK_FINITE(_current_barrier_weight);

        TRACTOR_DEBUG("constraints " << _constraint_indices.size());

        TRACTOR_LOG_VAR(_qp_solution.head(10));

        _step_solution = _qp_solution;

        _in_constraint_phase = false;
        _dualres(_step_solution, _step_residuals);
        if (!_step_residuals.allFinite()) {
          for (size_t i = 0; i < _step_residuals.size(); i++) {
            if (!std::isfinite(_step_residuals[i])) {
              TRACTOR_FATAL("residual " << i << " not finite "
                                        << _step_residuals[i]);
            }
          }
        }
        TRACTOR_CHECK_ALL_FINITE(_step_residuals);
        TRACTOR_CHECK_ALL_FINITE(_qp_solution);
        {
          TRACTOR_PROFILER("solve linear");
          _linear_solver.compute(_hgrad);
          _step_solution =
              _linear_solver.solveWithGuess(_step_residuals, _qp_solution);
          TRACTOR_CHECK_ALL_FINITE(_step_solution);
        }

        Scalar line_search_result = 1;

        {
          if (1) {
            auto df = [&](const Scalar &v) {
              _line_search_temp_solution =
                  _qp_solution + (_step_solution - _qp_solution) * v;
              _dualres(_line_search_temp_solution, _line_search_temp_residuals);
              _dualprop(_line_search_temp_solution,
                        _line_search_temp_gradients);
              Scalar ret = Scalar((_qp_solution - _step_solution)
                                      .head(_primal_variable_count)
                                      .dot((_line_search_temp_residuals -
                                            _line_search_temp_gradients)
                                               .head(_primal_variable_count)));
              return ret;
            };
            if (1) {
              TRACTOR_PROFILER("qp line search");
              {
                Scalar x = 1;
                while (!(df(x) < 0)) {
                  x *= 0.5;
                  if (x >= tolerance()) {
                    continue;
                  } else {
                    x = 0;
                    break;
                  }
                }
                line_search_result = x;
              }
              TRACTOR_DEBUG("line_search_result " << line_search_result);
            }
          }
        }

        TRACTOR_LOG_VAR(line_search_result);

        _current_step = _step_solution - _qp_solution;

        line_search_result *= 0.9;

        _current_step *= line_search_result;

        _step_solution = _qp_solution + _current_step;

        Scalar step_size_sq =
            _current_step.head(_primal_variable_count).squaredNorm();
        Scalar step_size = sqrt(step_size_sq);
        TRACTOR_LOG_VAR(step_size);

        _qp_solution = _step_solution;

        iteration_count++;

        Scalar tol = tolerance() *
                     std::max(Scalar(1), std::max(_computeBarrierWeight(),
                                                  _computeObjectiveWeight()));
        if (step_size_sq <= tol * tol || line_search_result <= tolerance()) {
          _current_barrier_weight *= _barrier_decrease;
        }
      }
    }

    TRACTOR_CHECK_ALL_FINITE(_qp_solution);

    _previous_nonlinear_solution = _nonlinear_solution;
    accumulate(_nonlinear_solution, _qp_solution);

    TRACTOR_CHECK_ALL_FINITE(_nonlinear_solution);

    TRACTOR_DEBUG("finished");

    return (_nonlinear_solution - _previous_nonlinear_solution).squaredNorm();
  }

 public:
  InteriorPointSolver(const std::shared_ptr<Engine> &engine)
      : SolverBase(engine) {
    _dual_matrix_replacement = std::make_shared<DualMatrixReplacement>(this);
    _hgrad = MatrixReplacement<Scalar>(_dual_matrix_replacement);
  }
};

}  // namespace tractor
