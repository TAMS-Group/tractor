// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/solvers/base.h>

#include <tractor/core/linesearch.h>
#include <tractor/core/sparsity.h>

namespace tractor {

template <class Scalar>
struct SpQP {
  Eigen::SparseMatrix<Scalar> objective_matrix;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> objective_vector;

  Eigen::SparseMatrix<Scalar> equality_matrix;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> equality_vector;

  Eigen::SparseMatrix<Scalar> inequality_matrix;
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> inequality_vector;
};

template <class Scalar>
struct SpQPSolver {
  virtual void solve(const SpQP<Scalar> &qp,
                     Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &solution) = 0;
};

template <class Scalar>
struct LambdaSpQPSolver : SpQPSolver<Scalar> {
  std::function<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>(const SpQP<Scalar> &)>
      lambda;
  virtual void solve(
      const SpQP<Scalar> &qp,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &solution) override {
    solution = lambda(qp);
  }
};

template <class Scalar>
struct SpQPSolverBase : SpQPSolver<Scalar> {
  typedef SpQP<Scalar> Problem;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::SparseMatrix<Scalar> Matrix;
  const Problem *problem = nullptr;
  Vector solution;
  bool finished = false;
  bool infeasible = false;
  bool success = false;
  Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * 8;
  virtual void solve(
      const Problem &problem,
      Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &solution) override {
    init(problem);
    while (!finished) {
      step();
    }
    solution = this->solution;
  }
  virtual void init(const Problem &problem_) {
    auto &solution = this->solution;

    problem = &problem_;

    solution.resize(problem->objective_matrix.cols());
    solution.setZero();

    infeasible = false;
    success = false;
    finished = false;

    initImpl();
  }
  virtual void step() { stepImpl(); }
  virtual void initImpl() = 0;
  virtual void stepImpl() = 0;
};

class ScopeTimer {
  typedef std::chrono::steady_clock Clock;
  double *result = nullptr;
  Clock::time_point start;

 public:
  ScopeTimer(double *result) : result(result), start(Clock::now()) {}
  ~ScopeTimer() {
    *result = std::chrono::duration<double>(Clock::now() - start).count();
  }
};

template <class Scalar>
class SpSQPSolver : public SolverBase {
  typedef Eigen::SparseMatrix<Scalar> SparseMatrix;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  Vector _nonlinear_solution;
  Vector _nonlinear_residuals;
  Vector _linear_solution;
  SparseMatrix _objective_selector;
  SparseMatrix _equality_selector;
  SparseMatrix _inequality_selector;
  SpQP<Scalar> _qp;
  SparseMatrix _multi_matrix;

 public:
  Scalar _step_scaling = 1.0;
  bool _backoff_enable = true;
  Scalar _backoff_factor = 0.5;
  std::shared_ptr<SparseMatrixBuilder<Scalar>> _matrix_builder;
  std::shared_ptr<SpQPSolver<Scalar>> _qp_solver;

  double _time_compute = 0;
  double _time_prepare = 0;
  double _time_matrix = 0;
  double _time_select = 0;
  double _time_solve = 0;
  double _time_negate = 0;
  double _time_backoff = 0;
  double _time_accumulate = 0;
  size_t _backoff_steps = 0;

  SparseMatrix _make_selection_matrix(size_t priority) {
    std::vector<Eigen::Triplet<Scalar>> triplets;
    size_t active = 0;
    for (size_t i = 0; i < _priority_list.size(); i++) {
      if (_priority_list[i] == priority) {
        triplets.emplace_back(active, i, 1);
        active++;
      }
    }
    TRACTOR_DEBUG(active << " entries for priority " << priority);
    SparseMatrix mat(active, _priority_list.size());
    mat.setFromTriplets(triplets.begin(), triplets.end());
    return mat;
  }

  virtual void _compile(const Program &prog) override {
    _compileGradients<Scalar>(prog);
    _matrix_builder = std::make_shared<SparseMatrixBuilder<Scalar>>(
        _engine, _p_fprop, _x_fprop);
    _objective_selector = _make_selection_matrix(0);
    _equality_selector = _make_selection_matrix(1);
    _inequality_selector = _make_selection_matrix(2);
  }

  virtual void _input(const Buffer &buffer) override {
    buffer.toVector(_nonlinear_solution);
  }

  virtual void _output(Buffer &buffer) override {
    buffer.fromVector(_nonlinear_solution);
  }

  virtual void _parameterize(const Buffer &buffer) override {
    _x_prog->parameterize(buffer, _memory);
  }

  double _loss = -1;
  virtual double loss() const override { return _loss; }

  virtual double _step() override {
    TRACTOR_ASSERT(_nonlinear_solution.allFinite());

    {
      TRACTOR_PROFILER("nonlinear");
      ScopeTimer t(&_time_compute);
      _x_prog->run(_nonlinear_solution, _memory, _nonlinear_residuals);
    }

    {
      TRACTOR_PROFILER("linearize");
      ScopeTimer t(&_time_prepare);
      _x_prep->execute(_memory);
    }

    {
      TRACTOR_DEBUG("build matrix");
      TRACTOR_PROFILER("build matrix");
      ScopeTimer t(&_time_matrix);
      _multi_matrix = _matrix_builder->build(_memory);
    }

    {
      ScopeTimer t(&_time_select);

      _qp.objective_matrix = _objective_selector * _multi_matrix;
      _qp.objective_vector = _objective_selector * _nonlinear_residuals;

      _qp.equality_matrix = _equality_selector * _multi_matrix;
      _qp.equality_vector = _equality_selector * _nonlinear_residuals;

      _qp.inequality_matrix = _inequality_selector * _multi_matrix;
      _qp.inequality_vector = _inequality_selector * _nonlinear_residuals;
    }

    {
      ScopeTimer t(&_time_solve);
      _qp_solver->solve(_qp, _linear_solution);
    }

    {
      ScopeTimer t(&_time_negate);
      _linear_solution.array() = -_linear_solution.array();
    }

    {
      ScopeTimer t(&_time_backoff);
      _backoff_steps = 0;
      if (_backoff_enable) {
        for (size_t i = 0;; i++) {
          auto nl2 = _nonlinear_solution;
          accumulate(nl2, _linear_solution);
          _x_prog->run(nl2, _memory, _nonlinear_residuals);
          if (((_inequality_selector * _nonlinear_residuals).array() >= 0)
                  .all()) {
            break;
          }
          _linear_solution *= _backoff_factor;
          _backoff_steps++;
          TRACTOR_DEBUG("step back " << i);
          if (i > _backoff_steps) {
            _linear_solution *= 0;
            break;
          }
        }
      }
    }

    {
      ScopeTimer t(&_time_accumulate);
      accumulate(_nonlinear_solution, _linear_solution * _step_scaling);
    }

    TRACTOR_DEBUG("ready");
    Scalar step = _linear_solution.squaredNorm();
    return step;
  }

  SpSQPSolver(const std::shared_ptr<Engine> &engine) : SolverBase(engine) {}
};

}  // namespace tractor
