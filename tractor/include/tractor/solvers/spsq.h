// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/solvers/base.h>

#include <tractor/core/linesearch.h>
#include <tractor/core/sparsity.h>

namespace tractor {

template <class Scalar> struct SparseLinearSolver {
  typedef Eigen::SparseMatrix<Scalar> Matrix;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) = 0;
};

template <class Scalar> struct SparseLinearCG : SparseLinearSolver<Scalar> {
  size_t max_iterations = 0;
  Scalar tolerance = 0;
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
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

template <class Scalar> struct SparseLinearLU : SparseLinearSolver<Scalar> {
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) override {
    Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::NaturalOrdering<int>>
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

template <class Scalar> struct SparseLinearGS : SparseLinearSolver<Scalar> {
  size_t max_iterations = 0;
  Scalar sor = Scalar(1.3);
  typedef typename SparseLinearSolver<Scalar>::Matrix Matrix;
  typedef typename SparseLinearSolver<Scalar>::Vector Vector;
  virtual void solve(const Matrix &matrix, const Vector &residuals,
                     Vector &solution) override {
    {
      TRACTOR_DEBUG("get diagonal");
      Vector diagonal = matrix.diagonal();
      TRACTOR_DEBUG("solve gauss seidel");
      TRACTOR_PROFILER("solve gauss seidel");
      for (size_t iteration = 0; iteration < max_iterations; iteration++) {
        auto project = [&](size_t i) {
          Scalar rhs = residuals[i];
          if (rhs != Scalar(0)) {
            Scalar current_value = solution[i];
            Scalar new_value = (rhs - (matrix.col(i).dot(solution) -
                                       diagonal[i] * current_value)) /
                               diagonal[i];
            solution[i] = (new_value - current_value) * sor + current_value;
          }
        };
        for (ssize_t i = 0; i < matrix.cols(); i++) {
          project(i);
        }
        for (ssize_t i = matrix.cols() - 1; i >= 0; i--) {
          project(i);
        }
      }
    }
  }
};

template <class Scalar> class SparseLeastSquaresSolver : public SolverBase {

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  Vector _nonlinear_solution;
  Vector _nonlinear_residuals;
  Vector _linear_residuals;
  Vector _linear_solution;
  Vector _test_vector_a;
  Vector _test_vector_b;
  Vector _diagonal;

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

    _p_prog = prog;
    _x_prog->compile(_p_prog);

    buildGradients(_p_prog, _p_prep, &_p_fprop, nullptr, nullptr, &_p_accu);
    _x_prep->compile(_p_prep);
    _x_fprop->compile(_p_fprop);
    _x_accu->compile(_p_accu);

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

  virtual double _step() override {

    TRACTOR_ASSERT(_nonlinear_solution.allFinite());

    {
      TRACTOR_PROFILER("nonlinear");
      _x_prog->run(_nonlinear_solution, _memory, _nonlinear_residuals);
    }

    {
      TRACTOR_PROFILER("linearize");
      _x_prep->execute(_memory);
    }

    Eigen::SparseMatrix<Scalar> jacobian;
    {
      TRACTOR_DEBUG("build jacobian");
      TRACTOR_PROFILER("build jacobian");
      jacobian = _matrix_builder->build(_memory);
    }

    {
      TRACTOR_PROFILER("linear residuals");
      _linear_residuals = jacobian.transpose() * _nonlinear_residuals;
    }

    if (_test_gradients) {
      TRACTOR_PROFILER("test gradients");
      _x_fprop->run(_linear_residuals, _memory, _test_vector_a);
      _test_vector_b = jacobian * _linear_residuals;
      TRACTOR_ASSERT(_test_vector_a.isApprox(_test_vector_b));
    }

    Eigen::SparseMatrix<Scalar> hessian;
    {
      TRACTOR_DEBUG("compute hessian");
      TRACTOR_PROFILER("compute hessian");
      hessian = jacobian.transpose() * jacobian;
    }

    if (_regularization > Scalar(0)) {
      TRACTOR_DEBUG("add regularization");
      TRACTOR_PROFILER("add regularization");
      // hessian.diagonal().array() += _regularization;
      // hessian =
      //     (hessian +
      //      Eigen::DiagonalMatrix<Scalar, Eigen::Dynamic>(
      //          Vector::Constant(hessian.rows(),
      //          _regularization).asDiagonal()))
      //         .eval()
      //     //.sparseView()
      //     ;
      std::vector<Eigen::Triplet<Scalar>> tri;
      for (size_t i = 0; i < hessian.rows(); i++) {
        tri.emplace_back(i, i, _regularization);
      }
      Eigen::SparseMatrix<Scalar> reg(hessian.rows(), hessian.cols());
      reg.setFromTriplets(tri.begin(), tri.end());
      hessian = hessian + reg;
    }

    _linear_solution = Vector::Zero(hessian.rows());

    // Eigen::SparseQR<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>>
    //     solver;
    // {
    //   TRACTOR_DEBUG("linear analyze");
    //   TRACTOR_PROFILER("linear analyze");
    //   solver.analyzePattern(hessian);
    // }
    // {
    //   TRACTOR_DEBUG("linear factorize");
    //   TRACTOR_PROFILER("linear factorize");
    //   solver.factorize(hessian);
    // }

    // Eigen::SparseLU<Eigen::SparseMatrix<Scalar>, Eigen::NaturalOrdering<int>>
    //     solver;
    // {
    //   TRACTOR_DEBUG("linear analyze");
    //   TRACTOR_PROFILER("linear analyze");
    //   solver.analyzePattern(hessian);
    // }
    // {
    //   TRACTOR_DEBUG("linear factorize");
    //   TRACTOR_PROFILER("linear factorize");
    //   solver.factorize(hessian);
    // }

    // Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>,
    //                          Eigen::Lower | Eigen::Upper>
    //     solver;
    // if (_max_linear_iterations >= 1) {
    //   solver.setMaxIterations(_max_linear_iterations);
    // }
    // if (_linear_tolerance >= 0) {
    //   solver.setTolerance(Scalar(_linear_tolerance));
    // }
    //
    // {
    //   TRACTOR_DEBUG("linear compute");
    //   TRACTOR_PROFILER("linear compute");
    //   solver.compute(hessian);
    // }
    //
    // {
    //   TRACTOR_DEBUG("linear solve");
    //   TRACTOR_PROFILER("linear solve");
    //   _linear_solution = solver.solve(_linear_residuals);
    // }

    // {
    //   TRACTOR_DEBUG("get diagonal");
    //   _diagonal = hessian.diagonal();
    //   TRACTOR_DEBUG("solve gauss seidel");
    //   TRACTOR_PROFILER("solve gauss seidel");
    //   for (size_t iteration = 0; iteration < _max_linear_iterations;
    //        iteration++) {
    //
    //     auto project = [&](size_t row) {
    //       Scalar rhs = _linear_residuals[row];
    //       if (rhs != Scalar(0)) {
    //         _linear_solution[row] =
    //             (rhs - (hessian.col(row).dot(_linear_solution) -
    //                     _diagonal[row] * _linear_solution[row])) /
    //             _diagonal[row];
    //       }
    //     };
    //
    //     for (ssize_t row = 0; row < hessian.rows(); row++) {
    //       project(row);
    //     }
    //
    //     for (ssize_t row = hessian.rows() - 1; row >= 0; row--) {
    //       project(row);
    //     }
    //   }
    // }

    _linear_solver->solve(hessian, _linear_residuals, _linear_solution);

    TRACTOR_DEBUG("ready");

    _linear_solution.array() =
        _linear_solution.array() * Scalar(-_step_scaling);

    accumulate(_nonlinear_solution, _linear_solution);
  }

  SparseLeastSquaresSolver(const std::shared_ptr<Engine> &engine)
      : SolverBase(engine) {}
};

} // namespace tractor
