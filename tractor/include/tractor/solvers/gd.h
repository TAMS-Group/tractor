// (c) 2020-2022 Philipp Ruppel

#pragma once

#if 1

#include <tractor/solvers/base.h>

namespace tractor {

// Steepest gradient-descent with box constraints
template <class Scalar> class GradientDescentSolver : public SolverBase {
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
  static constexpr Scalar _default_learning_rate = Scalar(0.01);
  Scalar _learning_rate = _default_learning_rate;

private:
  Vector _pl, _gl, _velocity, _residuals, _v_fprop;
  Vector _line_search_left, _line_search_right;
  Scalar _momentum = Scalar(0);
  Vector _errors;
  Vector _accu_in;

protected:
  virtual void _compile(const Program &prog) override {
    _compileGradients<Scalar>(prog);
  }

  virtual void _input(const Buffer &buffer) override {
    buffer.toVector(_p_prog.inputs(), _pl);
    _velocity.setZero(_x_bprop->outputBufferSize() / sizeof(Scalar));
  }

  virtual void _output(Buffer &buffer) override {
    buffer.fromVector(_p_prog.inputs(), _pl);
  }

  virtual void _parameterize(const Buffer &buffer) override {
    _x_prog->parameterize(buffer, _memory);
  }

  double _loss = -1;
  virtual double loss() const override { return _loss; }

  virtual double _step() override {

    // TRACTOR_DEBUG_STREAM("gd step");

    {
      TRACTOR_PROFILER("nonlinear");
      _x_prog->run(_pl, _memory, _residuals);
    }

    // TRACTOR_DEBUG_STREAM("loss " << _residuals.squaredNorm());
    _loss = _residuals.squaredNorm();

    {
      TRACTOR_PROFILER("prepare");
      _x_prep->execute(_memory);
    }

    {
      TRACTOR_PROFILER("bprop");
      _x_bprop->run(_residuals, _memory, _gl);

      //_gl.normalize();

      //_x_hprop->run(_gl, _memory, _gl);

      // TRACTOR_DEBUG_STREAM(_gl);
      // getchar();
    }

    if (_gl.allFinite()) {

      _velocity = _velocity * _momentum - _gl * _learning_rate;

      //_velocity = _velocity * _momentum - _gl.normalized() * _learning_rate;

      //_velocity *= _momentum;
      //_velocity -= _gl * (_learning_rate / (1e-12 + _gl.norm()));

      //_velocity *= _momentum;
      // _velocity -= _gl * _learning_rate;

      //_velocity += Eigen::VectorXd::Random(_velocity.size()) * 0.01;

      //   {
      //     TRACTOR_PROFILER("project");
      //     _x_project->parameterVector(std::array<Scalar, 1>({Scalar(0)}),
      //                                 _memory);
      //     _x_project->run(_velocity, _memory, _velocity);
      //     TRACTOR_CHECK_ALL_FINITE(_velocity);
      //   }

      TRACTOR_CHECK_ALL_FINITE(_velocity);

      accumulate(_pl, _velocity);

      // applyBounds(_pl, _velocity);

      /*
      {
        TRACTOR_PROFILER("sq nl line search");
        auto f = [&](const Scalar &v) {
          _line_search_left = _pl;
          accumulate(_line_search_left, _velocity * v);
          _x_prog->inputVector(_line_search_left, _memory);
          _x_prog->execute(_memory);
          _x_prog->outputVector(_memory, _line_search_right);
          Scalar ret = _line_search_right.squaredNorm();
          if (!std::isfinite(ret)) {
            ret = std::numeric_limits<Scalar>::max();
          }
          return ret;
        };
        Scalar line_search_result =
            minimizeTernary(f, tolerance(), Scalar(0), Scalar(1));
        TRACTOR_DEBUG_STREAM("ls " << line_search_result);
        line_search_result *= Scalar(0.9);
        _velocity *= line_search_result;
        accumulate(_pl, _velocity);
      }
      */

    } else {
      TRACTOR_DEBUG_STREAM("not finite " << _gl);
    }

    return _velocity.squaredNorm();
  }

public:
  GradientDescentSolver(const std::shared_ptr<Engine> &engine,
                        Scalar learning_rate = _default_learning_rate,
                        Scalar momentum = 0)
      : SolverBase(engine), _learning_rate(learning_rate), _momentum(momentum) {
  }

  const Scalar &learningRate() const { return _learning_rate; }
  void setLearningRate(const Scalar &learning_rate) {
    _learning_rate = learning_rate;
  }
};

} // namespace tractor

#endif
