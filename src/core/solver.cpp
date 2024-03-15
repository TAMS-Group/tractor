// 2020-2024 Philipp Ruppel

#include <tractor/core/solver.h>

#include <tractor/core/engine.h>
#include <tractor/core/log.h>

#include <chrono>

namespace tractor {

Solver::Solver(const std::shared_ptr<Engine> &engine) : _engine(engine) {}

void Solver::_log(const char *label, const Program &prog) {
  TRACTOR_DEBUG("solver program " << label << " " << typeid(*this).name());
}

void Solver::compile(const Program &prog) {
  _inputs.assign(&*prog.inputs().begin(), &*prog.inputs().end());
  _parameters.assign(&*prog.parameters().begin(), &*prog.parameters().end());
  _compile(prog);
  _compiled = true;
}

void Solver::parameterize(const Buffer &buffer) {
  _checkCompiled();
  _parameterize(buffer);
}

void Solver::input(const Buffer &buffer) {
  _checkCompiled();
  _input(buffer);
}

void Solver::output(Buffer &buffer) {
  _checkCompiled();
  _output(buffer);
}

void Solver::parameterize() {
  _checkCompiled();

  _buffer.gather(_parameters);
  _parameterize(_buffer);
}

void Solver::gather() {
  _checkCompiled();

  _buffer.gather(_parameters);
  _parameterize(_buffer);

  _buffer.gather(_inputs);
  _input(_buffer);
}

void Solver::scatter() {
  _checkCompiled();
  _output(_buffer);
  _buffer.scatter(_inputs);
}

double Solver::step() {
  _start_time = std::chrono::steady_clock::now();
  _first_step = true;
  return _step();
}

bool Solver::_expired() const {
  return (!_first_step || _hard_timeout) && _timeout > 0 &&
         (std::chrono::steady_clock::now() >
          _start_time + std::chrono::duration<double>(_timeout));
}

void Solver::solve() {
  _start_time = std::chrono::steady_clock::now();
  _first_step = true;
  for (size_t iteration = 0;; iteration++) {
    if (_max_iterations > 0 && iteration >= _max_iterations) {
      TRACTOR_DEBUG("solver reached max number of iterations");
      break;
    }
    double step = _step();
    _first_step = false;
    if (step < _tolerance) {
      TRACTOR_DEBUG("converged");
      break;
    }
    if (_expired()) {
      break;
    }
  }
}

}  // namespace tractor
