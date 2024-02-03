// (c) 2020-2022 Philipp Ruppel

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

    // Keep a pointer to the optimization problem that we're solving
    problem = &problem_;

    // Initialize the solution vector
    solution.resize(problem->objective_matrix.cols());
    solution.setZero();

    // Reset flags
    infeasible = false;
    success = false;
    finished = false;

    // Call implementation
    initImpl();
  }
  virtual void step() { stepImpl(); }
  virtual void initImpl() = 0;
  virtual void stepImpl() = 0;
};

template <class Scalar>
struct InteriorPointSpQPSolver : SpQPSolverBase<Scalar> {
  struct VariableBounds {
    size_t variable_index = 0;
    Scalar minimum = -std::numeric_limits<Scalar>::infinity();
    Scalar maximum = +std::numeric_limits<Scalar>::infinity();
    VariableBounds() {}
    VariableBounds(size_t variable_index, Scalar minimum, Scalar maximum)
        : variable_index(variable_index), minimum(minimum), maximum(maximum) {}
  };
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
  typedef Eigen::SparseMatrix<Scalar> Matrix;
  Eigen::SparseLU<Matrix> linear_solver;
  bool pattern_analyzed = false;
  bool barrier_initialized = false;
  size_t iterations = 0;
  Scalar barrier_weight = 1.0;
  Scalar initial_barrier_weight = 10.0;
  size_t dual_dimensions = 0;
  size_t first_equality_variable = 0;
  size_t first_inequality_variable = 0;
  size_t first_inequality_slack_variable = 0;
  Matrix dual_matrix, dual_matrix_prototype;
  Vector dual_rhs, dual_rhs_prototype;
  Vector dual_solution, next_dual_solution, previous_dual_solution;
  bool fixed_barrier = false;
  Scalar min_barrier_weight = 0.0;
  Scalar barrier_update_factor = 0.5;
  Scalar linear_tolerance = -1;
  Vector buffer;
  std::vector<VariableBounds> dual_box_constraints;
  virtual void initImpl() override {
    auto &solution = this->solution;
    auto &problem = this->problem;

    // Init initial guess
    solution.setZero();

    // Start at iteration zero
    iterations = 0;

    // Start at center
    /*if (!fixed_barrier) {
      barrier_weight = initial_barrier_weight;
    }*/

    // Block placement
    dual_dimensions =
        problem->objective_matrix.cols() + problem->equality_matrix.rows() +
        problem->inequality_matrix.rows() + problem->inequality_matrix.rows();
    first_equality_variable = problem->objective_matrix.cols();
    first_inequality_variable =
        first_equality_variable + problem->equality_matrix.rows();
    first_inequality_slack_variable =
        first_inequality_variable + problem->inequality_matrix.rows();

    // Setup normal equation for quadratic objectives
    Matrix objective_matrix =
        problem->objective_matrix.transpose() * problem->objective_matrix;
    Vector objective_rhs =
        problem->objective_matrix.transpose() * problem->objective_vector;

    // Build dual matrix prototype
    std::vector<Eigen::Triplet<Scalar>> triplets;
    for (int k = 0; k < objective_matrix.outerSize(); ++k) {
      for (typename Matrix::InnerIterator it(objective_matrix, k); it; ++it) {
        triplets.emplace_back(it.row(), it.col(), it.value());
      }
    }
    for (int k = 0; k < problem->equality_matrix.outerSize(); ++k) {
      for (typename Matrix::InnerIterator it(problem->equality_matrix, k); it;
           ++it) {
        triplets.emplace_back(first_equality_variable + it.row(), it.col(),
                              it.value());
        triplets.emplace_back(it.col(), first_equality_variable + it.row(),
                              it.value());
      }
    }
    for (int k = 0; k < problem->inequality_matrix.outerSize(); ++k) {
      for (typename Matrix::InnerIterator it(problem->inequality_matrix, k); it;
           ++it) {
        triplets.emplace_back(first_inequality_variable + it.row(), it.col(),
                              it.value());
        triplets.emplace_back(it.col(), first_inequality_variable + it.row(),
                              it.value());
      }
    }
    for (size_t i = 0; i < problem->inequality_vector.size(); i++) {
      triplets.emplace_back(first_inequality_slack_variable + i,
                            first_inequality_variable + i, -1.0);
      triplets.emplace_back(first_inequality_variable + i,
                            first_inequality_slack_variable + i, -1.0);
      triplets.emplace_back(first_inequality_slack_variable + i,
                            first_inequality_slack_variable + i,
                            std::numeric_limits<Scalar>::epsilon());
    }
    for (size_t i = 0; i < dual_dimensions; i++) {
      triplets.emplace_back(i, i, std::numeric_limits<Scalar>::epsilon());
    }
    dual_matrix_prototype.resize(dual_dimensions, dual_dimensions);
    dual_matrix_prototype.setFromTriplets(triplets.begin(), triplets.end());

    // Build dual right-hand-side prototype
    dual_rhs_prototype.resize(dual_dimensions);
    dual_rhs_prototype.setZero();
    dual_rhs_prototype.head(objective_rhs.size()) = objective_rhs;
    dual_rhs_prototype.segment(first_equality_variable,
                               problem->equality_vector.size()) =
        problem->equality_vector;
    dual_rhs_prototype.segment(first_inequality_variable,
                               problem->inequality_vector.size()) =
        problem->inequality_vector;

    // Create and initialize dual solution vectors
    dual_solution.resize(dual_dimensions);
    dual_solution.setZero();
    // dual_solution.head(solution.size()) = solution;
    for (size_t i = 0; i < problem->inequality_vector.size(); i++) {
      dual_solution[first_inequality_slack_variable + i] = 1.0;
    }
    previous_dual_solution = dual_solution;
    next_dual_solution = dual_solution;

    dual_box_constraints.clear();
    for (size_t i = 0; i < problem->inequality_vector.size(); i++) {
      VariableBounds bounds;
      bounds.variable_index = first_inequality_slack_variable + i;
      bounds.minimum = 0.0;
      dual_box_constraints.push_back(bounds);
    }

    pattern_analyzed = false;
    barrier_initialized = false;
  }
  virtual void stepImpl() override {
    auto &solution = this->solution;
    auto &problem = this->problem;
    auto &tolerance = this->tolerance;
    auto &finished = this->finished;
    auto &success = this->success;

    TRACTOR_DEBUG("step barrier weight " << barrier_weight);

    // Apply any solution changes to dual
    dual_solution.head(solution.size()) = solution;

    // Backup previous solution
    previous_dual_solution = dual_solution;

    // Sanity checks
    if (!solution.allFinite()) {
      TRACTOR_DEBUG("primal solution\n" << solution << "\n");
      throw std::runtime_error("numeric error, primal solution not finite");
    }
    if (!dual_solution.allFinite()) {
      TRACTOR_DEBUG("dual solution\n" << dual_solution << "\n");
      throw std::runtime_error("numeric error, dual solution not finite");
    }

    auto updateDual = [&]() {
      // Create barriers in the dual
      // double barrier_sum = 0.0;
      // double
      Vector barrier_diagonal(dual_solution.size());
      Vector barrier_rhs(dual_solution.size());
      barrier_diagonal.setZero();
      barrier_rhs.setZero();
      for (auto &box : dual_box_constraints) {
        Scalar &value = dual_solution[box.variable_index];
        if (std::isfinite(box.minimum)) {
          Scalar distance = (value - box.minimum);
          if (!std::isfinite(distance)) {
            throw std::runtime_error(
                "distance to constraint minimum not finite");
          }
          if (distance == 0) {
            std::cout << value << " " << box.minimum << " " << box.maximum
                      << std::endl;
            throw std::runtime_error("distance to constraint minimum is zero");
          }
          // Scalar barrier_value = -std::log(distance);
          // barrier_sum += barrier_value;
          Scalar first_derivative = -1.0 / distance;
          Scalar second_derivative = first_derivative * first_derivative;
          barrier_diagonal[box.variable_index] += second_derivative;
          barrier_rhs[box.variable_index] +=
              value * second_derivative - first_derivative;
        }
        if (std::isfinite(box.maximum)) {
          Scalar distance = (value - box.maximum);
          if (!std::isfinite(distance)) {
            throw std::runtime_error(
                "distance to constraint maximum not finite");
          }
          if (distance == 0) {
            std::cout << value << " " << box.minimum << " " << box.maximum
                      << std::endl;
            throw std::runtime_error("distance to constraint maximum is zero");
          }
          // Scalar barrier_value = -std::log(distance);
          // barrier_sum += barrier_value;
          Scalar first_derivative = -1.0 / distance;
          Scalar second_derivative = first_derivative * first_derivative;
          barrier_diagonal[box.variable_index] += second_derivative;
          barrier_rhs[box.variable_index] +=
              value * second_derivative - first_derivative;
        }
      }

      // Sanity check
      if (!barrier_rhs.allFinite()) {
        TRACTOR_DEBUG("rhs\n" << barrier_rhs << "\n");
        throw std::runtime_error("numeric error, not finite, a");
      }

      // TRACTOR_DEBUG("barrier_sum " << barrier_sum);
      // barrier_weight = barrier_diagonal.norm();

      // Scale barrier
      barrier_diagonal *= barrier_weight;
      barrier_rhs *= barrier_weight;

      // Assemble dual matrix for current barrier step
      dual_matrix = dual_matrix_prototype;
      dual_matrix += barrier_diagonal.asDiagonal();
      dual_rhs = dual_rhs_prototype + barrier_rhs;
    };
    updateDual();

    // We'll be working on next_dual_solution
    next_dual_solution = dual_solution;

    // Sanity check
    if (!dual_rhs.allFinite()) {
      TRACTOR_DEBUG("dual rhs\n" << dual_rhs << "\n");
      throw std::runtime_error("numeric error, not finite, c");
    }

    // Do we even have to do anything?
    if ((dual_rhs - dual_matrix * dual_solution).squaredNorm() > 0.0) {
      // Solve dual

#if 0
    static Eigen::ConjugateGradient<Matrix, Eigen::Lower | Eigen::Upper,
                                    Eigen::DiagonalPreconditioner<Scalar>
                                    // Eigen::IncompleteLUT<Scalar>
                                    // Eigen::IdentityPreconditioner
                                    // GaussSeidelPreconditioner
                                    >
        linear_solver;
    if (linear_tolerance >= 0.0) {
      linear_solver.setTolerance(linear_tolerance);
    } else {
      linear_solver.setTolerance(tolerance);
    }
    // linear_solver.setMaxIterations(100);
    linear_solver.compute(dual_matrix);
    next_dual_solution =
        linear_solver.solveWithGuess(dual_rhs, next_dual_solution);
#endif

#if 1
      // static Eigen::SparseLU<Eigen::SparseMatrix<Scalar>> linear_solver;
      // std::cout << "a" << std::endl;
      if (!pattern_analyzed) {
        // PROFILER("ip analyze");
        linear_solver.analyzePattern(dual_matrix);
        pattern_analyzed = true;
      }
      {
        // PROFILER("ip factorize");
        linear_solver.factorize(dual_matrix);
      }
      next_dual_solution = linear_solver.solve(dual_rhs);
      // std::cout << "b" << std::endl;
#endif

#if 0
    // std::cout << "a" << std::endl;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<Scalar>,
                             Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<Scalar>>
        linear_solver;
    if (linear_tolerance >= 0.0) {
      linear_solver.setTolerance(linear_tolerance);
    } else {
      linear_solver.setTolerance(tolerance);
    }
    linear_solver.compute(dual_matrix);
    // std::cout << dual_rhs << std::endl;
    next_dual_solution = linear_solver.solveWithGuess(dual_rhs, dual_solution);
    // std::cout << "b" << std::endl;
#endif

#if 0
    Eigen::ConjugateGradient<Eigen::SparseMatrix<float>,
                             Eigen::Lower | Eigen::Upper,
                             Eigen::DiagonalPreconditioner<float>
                             // Eigen::IncompleteLUT<Scalar>
                             // Eigen::IdentityPreconditioner
                             // GaussSeidelPreconditioner
                             >
        linear_solver;
    if (linear_tolerance >= 0.0) {
      linear_solver.setTolerance(linear_tolerance);
    } else {
      linear_solver.setTolerance(tolerance);
    }
    Eigen::SparseMatrix<float> float_dual_matrix = dual_matrix.cast<float>();
    linear_solver.compute(float_dual_matrix);
    Eigen::VectorXf float_dual_solution = next_dual_solution.cast<float>();
    Eigen::VectorXf float_dual_rhs = dual_rhs.cast<float>();
    float_dual_solution =
        linear_solver.solveWithGuess(float_dual_rhs, float_dual_solution);
    next_dual_solution = float_dual_solution.cast<Scalar>();
#endif

      TRACTOR_DEBUG("barrier_initialized " << barrier_initialized);

      Scalar step_size_2 = (next_dual_solution - dual_solution).squaredNorm();

      TRACTOR_DEBUG("step_size_2 " << step_size_2 << " "
                                   << tolerance * tolerance);

      bool decrease_barrier = (step_size_2 <= tolerance * tolerance) /*&&
                                            barrier_initialized*/
          ;

      // Are we there yet?
      if (decrease_barrier) {  // ????
        if (barrier_weight < tolerance ||
            (min_barrier_weight > 0.0 && barrier_weight < min_barrier_weight)) {
          finished = true;
          success = true;
        } else {
          barrier_weight *= 0.5;
        }
      }

      // Don't step across box constraints
      if (  // linear_solver.iterations() > 0 &&
          (dual_solution - next_dual_solution).squaredNorm() >
          tolerance * tolerance) {
        Scalar step_scale = 10.0;
        for (size_t i = 0; i < dual_box_constraints.size(); i++) {
          size_t v = dual_box_constraints[i].variable_index;
          if (std::isfinite(dual_box_constraints[i].minimum) &&
              next_dual_solution[v] < dual_box_constraints[i].minimum) {
            Scalar n = (dual_box_constraints[i].minimum - dual_solution[v]);
            Scalar d = (next_dual_solution[v] - dual_solution[v]);
            if (d != 0.0 && (n / d) < step_scale) {
              step_scale = (n / d);
            }
          }
          if (std::isfinite(dual_box_constraints[i].maximum) &&
              next_dual_solution[v] > dual_box_constraints[i].maximum) {
            Scalar n = (dual_box_constraints[i].maximum - dual_solution[v]);
            Scalar d = (next_dual_solution[v] - dual_solution[v]);
            if (d != 0.0 && (n / d) < step_scale) {
              step_scale = (n / d);
            }
          }
        }

        TRACTOR_DEBUG("step_scale 1 " << step_scale);
        if (step_scale < Scalar(2.0)) {
          step_scale = std::max(Scalar(0.0), std::min(Scalar(1.0), step_scale));
          step_scale *= Scalar(0.5);
          if (!barrier_initialized && barrier_weight < initial_barrier_weight) {
            barrier_weight *= Scalar(2.0);
            TRACTOR_DEBUG("increase barrier");
            return;
          } else {
            barrier_initialized = true;
          }
        } else {
          step_scale = 1.0;
          barrier_initialized = true;
        }
        TRACTOR_DEBUG("step_scale 2 " << step_scale);
        if (barrier_initialized) {
          dual_solution += (next_dual_solution - dual_solution) * step_scale;
        }
      }
    }

    /*// Are we there yet?
    if ((dual_solution - previous_dual_solution).squaredNorm() <=
            tolerance * tolerance &&
        barrier_initialized) {
      finished = true;
      success = true;
    }*/

    // Update primal solution
    solution = dual_solution.head(solution.size());

    iterations++;
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
  size_t _backoff_steps = 100;
  std::shared_ptr<SparseMatrixBuilder<Scalar>> _matrix_builder;
  std::shared_ptr<SpQPSolver<Scalar>> _qp_solver;

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

  //   void _split(const SparseMatrix &im, const Vector &iv, int priority,
  //               const SparseMatrix &om, const Vector &ov) {

  //               }

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

    {
      TRACTOR_DEBUG("build matrix");
      TRACTOR_PROFILER("build matrix");
      _multi_matrix = _matrix_builder->build(_memory);
    }

    _qp.objective_matrix = _objective_selector * _multi_matrix;
    _qp.objective_vector = _objective_selector * _nonlinear_residuals;

    _qp.equality_matrix = _equality_selector * _multi_matrix;
    _qp.equality_vector = _equality_selector * _nonlinear_residuals;

    _qp.inequality_matrix = _inequality_selector * _multi_matrix;
    _qp.inequality_vector = _inequality_selector * _nonlinear_residuals;

    _qp_solver->solve(_qp, _linear_solution);

    _linear_solution.array() = -_linear_solution.array();

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
        TRACTOR_DEBUG("step back " << i);
        if (i > _backoff_steps) {
          _linear_solution *= 0;
          break;
        }
      }
    }

    accumulate(_nonlinear_solution, _linear_solution * _step_scaling);

    TRACTOR_DEBUG("ready");
    Scalar step = _linear_solution.squaredNorm();
    return step;
  }

  SpSQPSolver(const std::shared_ptr<Engine> &engine) : SolverBase(engine) {}
};

}  // namespace tractor
