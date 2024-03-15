// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/solvers/base.h>

#include <tractor/core/linesearch.h>
#include <tractor/core/sparsity.h>

namespace tractor {

template <class Scalar>
struct SparseLinearSolver {
  typedef Eigen::SparseMatrix<Scalar> Matrix;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) = 0;
};

template <class Scalar>
struct IterativeSparseLinearSolver : SparseLinearSolver<Scalar> {
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  size_t max_iterations = 0;
  Scalar tolerance = 0;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) = 0;
};

template <class Scalar>
struct SparseLinearCG : IterativeSparseLinearSolver<Scalar> {
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  using IterativeSparseLinearSolver<Scalar>::max_iterations;
  using IterativeSparseLinearSolver<Scalar>::tolerance;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) override {
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>,
                             Eigen::Lower | Eigen::Upper>
        solver;
    if (max_iterations >= 1) {
      solver.setMaxIterations(max_iterations);
    }
    if (tolerance >= 0) {
      solver.setTolerance(Scalar(tolerance));
    }
    {
      TRACTOR_DEBUG("linear compute");
      TRACTOR_PROFILER("linear compute");
      solver.compute(matrix);
    }
    {
      TRACTOR_DEBUG("linear solve");
      TRACTOR_PROFILER("linear solve");
      solution = solver.solve(residuals);
    }
  }
};

template <class Scalar>
struct SparseLinearBiCGSTAB : IterativeSparseLinearSolver<Scalar> {
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  using IterativeSparseLinearSolver<Scalar>::max_iterations;
  using IterativeSparseLinearSolver<Scalar>::tolerance;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) override {
    Eigen::BiCGSTAB<Eigen::SparseMatrix<Scalar>> solver;
    if (max_iterations >= 1) {
      solver.setMaxIterations(max_iterations);
    }
    if (tolerance >= 0) {
      solver.setTolerance(Scalar(tolerance));
    }
    {
      TRACTOR_DEBUG("linear compute");
      TRACTOR_PROFILER("linear compute");
      solver.compute(matrix);
    }
    {
      TRACTOR_DEBUG("linear solve");
      TRACTOR_PROFILER("linear solve");
      solution = solver.solve(residuals);
    }
  }
};

template <class Scalar>
struct SparseLinearLU : SparseLinearSolver<Scalar> {
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  // Eigen::SparseLU<Matrix, Eigen::NaturalOrdering<int>>
  Eigen::SparseLU<Matrix, Eigen::COLAMDOrdering<int>> solver;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) override {
    solver.isSymmetric(true);
    {
      TRACTOR_DEBUG("linear analyze");
      TRACTOR_PROFILER("linear analyze");
      solver.analyzePattern(matrix);
    }
    {
      TRACTOR_DEBUG("linear factorize");
      TRACTOR_PROFILER("linear factorize");
      solver.factorize(matrix);
    }
    {
      TRACTOR_DEBUG("linear solve");
      TRACTOR_PROFILER("linear solve");
      solution = solver.solve(residuals);
    }
  }
};

template <class Scalar>
struct SparseLinearQR : SparseLinearSolver<Scalar> {
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) override {
    // Eigen::SparseQR<Eigen::SparseMatrix<Scalar>, Eigen::NaturalOrdering<int>>
    Eigen::SparseQR<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>>
        solver;
    {
      TRACTOR_DEBUG("linear analyze");
      TRACTOR_PROFILER("linear analyze");
      solver.analyzePattern(matrix);
    }
    {
      TRACTOR_DEBUG("linear factorize");
      TRACTOR_PROFILER("linear factorize");
      solver.factorize(matrix);
    }
    {
      TRACTOR_DEBUG("linear solve");
      TRACTOR_PROFILER("linear solve");
      solution = solver.solve(residuals);
    }
  }
};

template <class Scalar>
struct SparseLinearGS : IterativeSparseLinearSolver<Scalar> {
  Scalar sor = Scalar(1.3);
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  using IterativeSparseLinearSolver<Scalar>::max_iterations;
  using IterativeSparseLinearSolver<Scalar>::tolerance;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) override {
    {
      TRACTOR_DEBUG("get diagonal");
      Vector diagonal = matrix.diagonal();
      Vector inv_diagonal = 1.0 / diagonal.array();
      TRACTOR_DEBUG("solve gauss seidel");
      TRACTOR_PROFILER("solve gauss seidel");
      for (size_t iteration = 0;; iteration++) {
        if (iteration >= max_iterations) {
          TRACTOR_DEBUG("GS max iterations reached " << iteration);
          break;
        }
        TRACTOR_PROFILER("gauss-seidel sweep");
        bool changed = false;
        auto project = [&](size_t i) {
          Scalar rhs = residuals[i];
          Scalar current_value = solution[i];
          Scalar new_value = (rhs - (matrix.col(i).dot(solution) -
                                     diagonal[i] * current_value)) *
                             inv_diagonal[i];
          Scalar delta = new_value - current_value;
          changed |= (delta > tolerance);
          solution[i] = delta * sor + current_value;
        };
        for (ssize_t i = 0; i < matrix.cols(); i++) {
          project(i);
        }
        for (ssize_t i = matrix.cols() - 1; i >= 0; i--) {
          project(i);
        }
        if (!changed) {
          TRACTOR_DEBUG("GS tolerance reached at iteration " << iteration);
          break;
        }
      }
    }
  }
};

template <class Scalar>
class SparseLeastSquaresSolver : public SolverBase {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  Vector _nonlinear_solution;
  Vector _nonlinear_residuals;
  Vector _linear_residuals;
  Vector _linear_solution;
  Vector _test_vector_a;
  Vector _test_vector_b;
  Vector _diagonal;
  Eigen::SparseMatrix<Scalar> _hessian;
  Eigen::SparseMatrix<Scalar> _jacobian;
  Eigen::SparseMatrix<Scalar> _jacobian_transpose;

 public:
  Scalar _regularization = 0.0;
  int _max_linear_iterations = -1;
  Scalar _step_scaling = 1.0;
  Scalar _linear_tolerance = -1;
  bool _test_gradients = false;
  std::shared_ptr<SparseMatrixBuilder<Scalar>> _matrix_builder;
  std::shared_ptr<SparseLinearSolver<Scalar>> _linear_solver =
      std::make_shared<SparseLinearCG<Scalar>>();

  virtual void _compile(const Program &prog) override {
    //_compileGradients<Scalar>(prog);

    TRACTOR_INFO("spsq compile nl");
    _p_prog = prog;
    _x_prog->compile(_p_prog);

    TRACTOR_INFO("spsq build gradients");
    buildGradients(_p_prog, _p_prep, &_p_fprop, nullptr, nullptr, &_p_accu);

    TRACTOR_INFO("spsq compile prep");
    _x_prep->compile(_p_prep);

    TRACTOR_INFO("spsq compile fprop");
    _x_fprop->compile(_p_fprop);

    TRACTOR_INFO("spsq compile accu");
    _x_accu->compile(_p_accu);

    TRACTOR_INFO("spsq sparsity");
    _matrix_builder = std::make_shared<SparseMatrixBuilder<Scalar>>(
        _engine, _p_fprop, _x_fprop);
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
      _x_prog->run(_nonlinear_solution, _memory, _nonlinear_residuals);
    }

    _loss = _nonlinear_residuals.squaredNorm();

    {
      TRACTOR_PROFILER("linearize");
      _x_prep->execute(_memory);
    }

    {
      TRACTOR_DEBUG("build jacobian");
      TRACTOR_PROFILER("build jacobian");
      _jacobian = _matrix_builder->build(_memory);
    }

    {
      TRACTOR_PROFILER("linear residuals");
      _linear_residuals = _jacobian.transpose() * _nonlinear_residuals;
    }

    if (_test_gradients) {
      TRACTOR_PROFILER("test gradients");
      _x_fprop->run(_linear_residuals, _memory, _test_vector_a);
      _test_vector_b = _jacobian * _linear_residuals;
      TRACTOR_ASSERT(_test_vector_a.isApprox(_test_vector_b));
    }

    {
      TRACTOR_INFO("compute hessian begin");
      TRACTOR_DEBUG("compute hessian");
      {
        TRACTOR_PROFILER("transpose jacobian");
        _jacobian_transpose = _jacobian.transpose();
      }
      {
        TRACTOR_PROFILER("compute hessian");
        _hessian = (_jacobian_transpose * _jacobian);  // .pruned();
      }
      TRACTOR_INFO("compute hessian ready");
    }

    if (_regularization > Scalar(0)) {
      TRACTOR_DEBUG("add regularization");
      TRACTOR_PROFILER("add regularization");
      std::vector<Eigen::Triplet<Scalar>> tri;
      for (size_t i = 0; i < _hessian.rows(); i++) {
        tri.emplace_back(i, i, _regularization);
      }
      Eigen::SparseMatrix<Scalar> reg(_hessian.rows(), _hessian.cols());
      reg.setFromTriplets(tri.begin(), tri.end());
      _hessian = _hessian + reg;
    }

    _linear_solution.setZero(_hessian.rows());

    _linear_solver->solve(_hessian, _linear_residuals, _linear_solution);

    TRACTOR_DEBUG("ready");

    _linear_solution.array() =
        _linear_solution.array() * Scalar(-_step_scaling);

    Scalar step = _linear_solution.squaredNorm();

    accumulate(_nonlinear_solution, _linear_solution);

    return step;
  }

  SparseLeastSquaresSolver(const std::shared_ptr<Engine> &engine)
      : SolverBase(engine) {}
};

}  // namespace tractor
