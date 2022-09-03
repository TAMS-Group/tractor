// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/solvers/base.h>

#include <tractor/core/linesearch.h>
#include <tractor/core/sparsity.h>

namespace tractor {

template <class Scalar> class SparseLeastSquaresSolver : public SolverBase {

  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  Vector _nonlinear_solution;
  Vector _nonlinear_residuals;
  Vector _linear_residuals;
  Vector _linear_solution;

  std::shared_ptr<SparseMatrixBuilder<Scalar>> _matrix_builder;

public:
  Scalar _regularization = 0.0;
  int _max_linear_iterations = -1;
  Scalar _step_scaling = 1.0;
  Scalar _linear_tolerance = -1;

  virtual void _compile(const Program &prog) override {
    _compileGradients<Scalar>(prog);
    _matrix_builder = std::make_shared<SparseMatrixBuilder<Scalar>>(_p_fprop);
  }

  virtual void _input(const Buffer &buffer) override {
    buffer.toVector(_p_prog.inputs(), _nonlinear_solution);
  }

  virtual void _output(Buffer &buffer) override {
    buffer.fromVector(_p_prog.inputs(), _nonlinear_solution);
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
      jacobian = _matrix_builder->build(_x_fprop, _memory);
    }

    {
      TRACTOR_PROFILER("linear residuals");
      _linear_residuals = jacobian.transpose() * _nonlinear_residuals;
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

    // Eigen::SparseQR<Eigen::SparseMatrix<Scalar>, Eigen::COLAMDOrdering<int>>
    //     solver;
    //
    // {
    //   TRACTOR_DEBUG("linear analyze");
    //   TRACTOR_PROFILER("linear analyze");
    //   solver.analyzePattern(hessian);
    // }
    //
    // {
    //   TRACTOR_DEBUG("linear factorize");
    //   TRACTOR_PROFILER("linear factorize");
    //   solver.factorize(hessian);
    // }

    Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>,
                             Eigen::Lower | Eigen::Upper>
        solver;
    if (_max_linear_iterations >= 1) {
      solver.setMaxIterations(_max_linear_iterations);
    }
    if (_linear_tolerance >= 0) {
      solver.setTolerance(Scalar(_linear_tolerance));
    }

    {
      TRACTOR_DEBUG("linear compute");
      TRACTOR_PROFILER("linear compute");
      solver.compute(hessian);
    }

    {
      TRACTOR_DEBUG("linear solve");
      TRACTOR_PROFILER("linear solve");
      _linear_solution = solver.solve(_linear_residuals);
    }

    TRACTOR_DEBUG("ready");

    _linear_solution.array() =
        _linear_solution.array() * Scalar(-_step_scaling);

    accumulate(_nonlinear_solution, _linear_solution);
  }

  SparseLeastSquaresSolver(const std::shared_ptr<Engine> &engine)
      : SolverBase(engine) {}
};

} // namespace tractor
