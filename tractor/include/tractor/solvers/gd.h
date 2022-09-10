// (c) 2020-2022 Philipp Ruppel

#pragma once

#if 1

#include <tractor/solvers/base.h>

namespace tractor {

// Steepest gradient-descent with box constraints
template <class Scalar> class GradientDescentSolver : public SolverBase {
protected:
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
  Scalar _learning_rate = Scalar(0.01);
  Scalar _momentum = Scalar(0);

protected:
  Vector _pl, _gl, _velocity, _residuals, _v_fprop;
  Vector _line_search_left, _line_search_right;
  Vector _errors;
  Vector _accu_in;

protected:
  virtual void _compile(const Program &prog) override {
    _compileGradients<Scalar>(prog);
  }

  virtual void _input(const Buffer &buffer) override {
    buffer.toVector(_pl);
    _velocity.setZero(_x_bprop->outputBufferSize() / sizeof(Scalar));
  }

  virtual void _output(Buffer &buffer) override { buffer.fromVector(_pl); }

  virtual void _parameterize(const Buffer &buffer) override {
    _x_prog->parameterize(buffer, _memory);
  }

  double _loss = -1;
  virtual double loss() const override { return _loss; }

  virtual double _step() override {

    // TRACTOR_DEBUG("gd step");

    {
      TRACTOR_PROFILER("nonlinear");
      _x_prog->run(_pl, _memory, _residuals);
    }

    // TRACTOR_DEBUG("loss " << _residuals.squaredNorm());
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

      // TRACTOR_DEBUG(_gl);
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
        TRACTOR_DEBUG("ls " << line_search_result);
        line_search_result *= Scalar(0.9);
        _velocity *= line_search_result;
        accumulate(_pl, _velocity);
      }
      */

    } else {
      TRACTOR_DEBUG("not finite " << _gl);
    }

    return _velocity.squaredNorm();
  }

public:
  GradientDescentSolver(const std::shared_ptr<Engine> &engine)
      : SolverBase(engine) {}

  const Scalar &learningRate() const { return _learning_rate; }
  void setLearningRate(const Scalar &learning_rate) {
    _learning_rate = learning_rate;
  }
};

template <class Scalar>
class AdamSolver : public GradientDescentSolver<Scalar> {
  typedef typename GradientDescentSolver<Scalar>::Vector Vector;

public:
  Scalar b1 = Scalar(0.9);
  Scalar b2 = Scalar(0.999);
  Scalar e = 1e-8;
  Vector m;
  Vector v;
  bool initrd = false;
  Scalar t = 0;
  virtual double _step() override {
    Scalar a = this->_learning_rate;
    {
      TRACTOR_PROFILER("nonlinear");
      this->_x_prog->run(this->_pl, this->_memory, this->_residuals);
    }
    this->_loss = this->_residuals.squaredNorm();
    {
      TRACTOR_PROFILER("prepare");
      this->_x_prep->execute(this->_memory);
    }
    {
      TRACTOR_PROFILER("bprop");
      this->_x_bprop->run(this->_residuals, this->_memory, this->_gl);
    }
    auto &g = this->_gl;
    if (!initrd || this->_first_step) {
      TRACTOR_DEBUG("reset adam");
      initrd = true;
      m = g * Scalar(0);
      v = g * Scalar(0);
      t = Scalar(0);
    }
    t = t + 1;
    m.array() = b1 * m.array() + (Scalar(1) - b1) * g.array();
    v.array() = b2 * v.array() + (Scalar(1) - b2) * (g.array() * g.array());
    Vector mh = m / pow(b1, t);
    Vector vh = v / pow(b2, t);
    Vector step = (-a * mh.array() / (vh.cwiseSqrt().array() + e)).matrix();
    this->accumulate(this->_pl, step);
    return step.squaredNorm();
  }
  AdamSolver(const std::shared_ptr<Engine> &engine)
      : GradientDescentSolver<Scalar>(engine) {
    this->_learning_rate = Scalar(0.001);
  }
};

} // namespace tractor

#endif
