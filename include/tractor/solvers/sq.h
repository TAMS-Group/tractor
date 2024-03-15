// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/solvers/base.h>

#include <tractor/core/linesearch.h>

namespace tractor {

template <class Scalar,
          class LinearSolver = Eigen::ConjugateGradient<
              MatrixReplacement<Scalar>, Eigen::Lower | Eigen::Upper,
              Eigen::IdentityPreconditioner>>
class LeastSquaresSolver : public SolverBase {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  LinearSolver _linear_solver;
  std::shared_ptr<RegularizedMatrixReplacement<Scalar>> _hgrad_p;
  MatrixReplacement<Scalar> _hgrad;
  Vector _line_search_left, _line_search_right;
  Vector _residuals;
  Vector _nonlinear_solution;
  Vector _previous_nonlinear_solution;
  Vector _linear_solution;
  Vector _gradient_temp;
  Vector _test;

 public:
  bool _adaptive_regularization = 0;
  bool _line_search = false;
  Scalar _regularization = 0.0;
  int _max_linear_iterations = -1;
  Scalar _step_scaling = 1.0;
  Scalar _linear_tolerance = -1;

 protected:
  virtual void _compile(const Program &prog) override {
    _compileGradients<Scalar>(prog);
  }

  virtual void _input(const Buffer &buffer) override {
    buffer.toVector(_nonlinear_solution);
    _previous_nonlinear_solution = _nonlinear_solution;
    if (_adaptive_regularization) {
      _regularization = 1.0;
    }
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
    if (!_nonlinear_solution.allFinite()) {
      throw std::runtime_error("previous solution not finite");
    }

    _hgrad_p->setRegularization(_regularization);

    {
      TRACTOR_PROFILER("nonlinear");
      _x_prog->run(_nonlinear_solution, _memory, _gradient_temp);
    }

    _loss = _gradient_temp.squaredNorm();

    TRACTOR_CHECK_ALL_FINITE(_gradient_temp);

    {
      TRACTOR_PROFILER("linearize");
      _x_prep->execute(_memory);
    }

    {
      TRACTOR_PROFILER("bprop");
      _x_bprop->run(_gradient_temp, _memory, _residuals);
    }

    TRACTOR_CHECK_ALL_FINITE(_residuals);

    _linear_solution.resize(_residuals.size());

    Scalar linear_tolerance = _linear_tolerance;
    if (linear_tolerance < 0) {
      linear_tolerance = tolerance();
    }

    _linear_solver.setTolerance(linear_tolerance);

    if (_max_linear_iterations > 0) {
      _linear_solver.setMaxIterations(_max_linear_iterations);
    } else {
      _linear_solver.setMaxIterations(_residuals.size());
    }

    {
      TRACTOR_PROFILER("sq linear");
      _linear_solver.compute(_hgrad);
      _linear_solution = _linear_solver.solve(_residuals);
    }

    TRACTOR_CHECK_ALL_FINITE(_linear_solution);

    _linear_solution.array() = -_linear_solution.array();

    {
      TRACTOR_PROFILER("adaptive regularization");
      auto f = [&](const Scalar &v) {
        _line_search_left = _nonlinear_solution;
        accumulate(_line_search_left, _linear_solution * v);
        _x_prog->inputVector(_line_search_left, _memory);
        _x_prog->execute(_memory);
        _x_prog->outputVector(_memory, _line_search_right);
        Scalar ret = _line_search_right.squaredNorm();
        return ret;
      };
      if (_adaptive_regularization) {
        if (f(1) < f(0.5)) {
          _regularization *= Scalar(0.5);
        } else {
          _regularization *= Scalar(2.0);
          _regularization = std::min(_regularization, Scalar(1.0));
        }
        TRACTOR_DEBUG("reg " << _regularization);
      }
    }

    if (_line_search) {
      TRACTOR_PROFILER("sq nl line search");
      auto f = [&](const Scalar &v) {
        _line_search_left = _nonlinear_solution;
        accumulate(_line_search_left, _linear_solution * v);

        _x_prog->inputVector(_line_search_left, _memory);
        _x_prog->execute(_memory);
        _x_prog->outputVector(_memory, _line_search_right);

        Scalar ret = _line_search_right.squaredNorm();
        if (!std::isfinite(ret)) {
          ret = std::numeric_limits<Scalar>::max();
        }
        return ret;
      };

      Scalar line_search_result = 1;
      line_search_result =
          minimizeTernary(f, Scalar(0.001), Scalar(0), Scalar(1));
      TRACTOR_DEBUG("ls " << line_search_result);
      _linear_solution *= line_search_result;
    }

    _linear_solution *= _step_scaling;

    double step = _linear_solution.squaredNorm();

    accumulate(_nonlinear_solution, _linear_solution);
    TRACTOR_CHECK_ALL_FINITE(_nonlinear_solution);

    if (_previous_nonlinear_solution.size() != _nonlinear_solution.size()) {
      TRACTOR_FATAL("internal error SQ171 "
                    << _previous_nonlinear_solution.size() << " "
                    << _nonlinear_solution.size());
      throw std::runtime_error("internal error");
    }

    return step;
  }

 public:
  LeastSquaresSolver(const std::shared_ptr<Engine> &engine)
      : SolverBase(engine) {
    _hgrad_p = std::make_shared<RegularizedMatrixReplacement<Scalar>>();
    _hgrad_p->setExecutable(_x_hprop);
    _hgrad_p->setMemory(_memory);
    _hgrad = MatrixReplacement<Scalar>(_hgrad_p);
  }
};

}  // namespace tractor
