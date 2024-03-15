// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/solvers/base.h>

namespace tractor {

template <class Scalar>
class GradientDescentSolver : public SolverBase {
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
    {
      TRACTOR_PROFILER("nonlinear");
      _x_prog->run(_pl, _memory, _residuals);
    }

    _loss = _residuals.squaredNorm();

    {
      TRACTOR_PROFILER("prepare");
      _x_prep->execute(_memory);
    }

    {
      TRACTOR_PROFILER("bprop");
      _x_bprop->run(_residuals, _memory, _gl);
    }

    if (_gl.allFinite()) {
      _velocity = _velocity * _momentum - _gl * _learning_rate;

      TRACTOR_CHECK_ALL_FINITE(_velocity);

      accumulate(_pl, _velocity);

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

}  // namespace tractor
