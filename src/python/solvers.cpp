// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/solver.h>
#include <tractor/core/sparsity.h>
#include <tractor/neural/network.h>
#include <tractor/solvers/gd.h>
#include <tractor/solvers/spsq.h>
#include <tractor/solvers/sq.h>
#include <tractor/solvers/ip.h>
#include <tractor/solvers/spsqp.h>

namespace tractor {

static void pythonizeProgramGlobal(py::module m) {
  m.def("sparsity_matrix", [](const Program &program, size_t stride) {
    return SparsityMatrix(program, stride).toEigenSparseMatrix<float>();
  });

  py::class_<Solver>(m, "Solver")
      .def("compile", [](Solver &solver,
                         const Program &program) { solver.compile(program); })
      .def("parameterize", [](Solver &solver) { solver.parameterize(); })
      .def("solve",
           [](Solver &solver) {
             solver.parameterize();
             solver.gather();
             solver.solve();
             solver.scatter();
           })
      .def("gather", &Solver::gather)
      .def("scatter", &Solver::scatter)
      .def("step", &Solver::step)
      .def_property_readonly("loss", &Solver::loss)
      .def_property("max_iterations", &Solver::maxIterations,
                    &Solver::setMaxIterations)
      .def_property("tolerance", &Solver::tolerance, &Solver::setTolerance)
      .def_property(
          "timeout", [](const Solver &solver) { return solver.timeout(); },
          [](Solver &solver, const double &v) { solver.setTimeout(v, false); })

      ;
}

TRACTOR_PYTHON_GLOBAL(pythonizeProgramGlobal);

template <class Scalar>
static void pythonizeSolvers(py::module main_module, py::module type_module) {
  py::class_<SparseMatrixBuilder<Scalar>,
             std::shared_ptr<SparseMatrixBuilder<Scalar>>>(
      type_module, "SparseMatrixBuilder")
      .def(py::init<const std::shared_ptr<Engine> &, const Program &,
                    const std::shared_ptr<Executable> &>())
      .def("build", &SparseMatrixBuilder<Scalar>::build)
      .def_property_readonly("complexity",
                             &SparseMatrixBuilder<Scalar>::complexity)
      .def_readwrite("multi_threading",
                     &SparseMatrixBuilder<Scalar>::_multi_threading)
      .def_property_readonly(
          "sparsity_matrix",
          [](const SparseMatrixBuilder<Scalar> &_this) {
            return _this.sparsityMatrix().toEigenSparseMatrix(1.0f);
          })

      ;

  py::class_<SparseLinearSolver<Scalar>,
             std::shared_ptr<SparseLinearSolver<Scalar>>>(type_module,
                                                          "SparseLinearSolver");

  py::class_<IterativeSparseLinearSolver<Scalar>,
             std::shared_ptr<IterativeSparseLinearSolver<Scalar>>,
             SparseLinearSolver<Scalar>>(type_module,
                                         "IterativeSparseLinearSolver")
      .def_readwrite("max_iterations", &SparseLinearCG<Scalar>::max_iterations)
      .def_readwrite("tolerance", &SparseLinearCG<Scalar>::tolerance);

  py::class_<SparseLinearCG<Scalar>, std::shared_ptr<SparseLinearCG<Scalar>>,
             IterativeSparseLinearSolver<Scalar>>(type_module, "SparseLinearCG")
      .def(py::init());

  py::class_<SparseLinearBiCGSTAB<Scalar>,
             std::shared_ptr<SparseLinearBiCGSTAB<Scalar>>,
             IterativeSparseLinearSolver<Scalar>>(type_module,
                                                  "SparseLinearBiCGSTAB")
      .def(py::init());

  py::class_<SparseLinearLU<Scalar>, std::shared_ptr<SparseLinearLU<Scalar>>,
             SparseLinearSolver<Scalar>>(type_module, "SparseLinearLU")
      .def(py::init());

  py::class_<SparseLinearQR<Scalar>, std::shared_ptr<SparseLinearQR<Scalar>>,
             SparseLinearSolver<Scalar>>(type_module, "SparseLinearQR")
      .def(py::init());

  py::class_<SparseLinearGS<Scalar>, std::shared_ptr<SparseLinearGS<Scalar>>,
             IterativeSparseLinearSolver<Scalar>>(type_module, "SparseLinearGS")
      .def(py::init())
      .def_readwrite("sor", &SparseLinearGS<Scalar>::sor);

  py::class_<SparseLeastSquaresSolver<Scalar>, Solver>(
      type_module, "SparseLeastSquaresSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("regularization",
                     &SparseLeastSquaresSolver<Scalar>::_regularization)
      .def_readwrite("step_scaling",
                     &SparseLeastSquaresSolver<Scalar>::_step_scaling)
      .def_readwrite("test_gradients",
                     &SparseLeastSquaresSolver<Scalar>::_test_gradients)
      .def_readwrite("matrix_builder",
                     &SparseLeastSquaresSolver<Scalar>::_matrix_builder)
      .def_readwrite("linear_solver",
                     &SparseLeastSquaresSolver<Scalar>::_linear_solver)

      ;

  py::class_<LeastSquaresSolver<Scalar>, Solver>(type_module,
                                                 "LeastSquaresSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("regularization",
                     &LeastSquaresSolver<Scalar>::_regularization)
      .def_readwrite("step_scaling", &LeastSquaresSolver<Scalar>::_step_scaling)
      .def_readwrite("linear_tolerance",
                     &LeastSquaresSolver<Scalar>::_linear_tolerance)
      .def_readwrite("adaptive_regularization",
                     &LeastSquaresSolver<Scalar>::_adaptive_regularization)
      .def_readwrite("line_search", &LeastSquaresSolver<Scalar>::_line_search)
      .def_readwrite("max_linear_iterations",
                     &LeastSquaresSolver<Scalar>::_max_linear_iterations)

      ;

  py::class_<GradientDescentSolver<Scalar>, Solver>(type_module,
                                                    "GradientDescentSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("learning_rate",
                     &GradientDescentSolver<Scalar>::_learning_rate)
      .def_readwrite("momentum", &GradientDescentSolver<Scalar>::_momentum)

      ;

  py::class_<InteriorPointSolver<Scalar>, Solver>(type_module,
                                                  "InteriorPointSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("initial_barrier_weight",
                     &InteriorPointSolver<Scalar>::_initial_barrier_weight)
      .def_readwrite("min_barrier_weight",
                     &InteriorPointSolver<Scalar>::_min_barrier_weight)
      .def_readwrite("barrier_decrease",
                     &InteriorPointSolver<Scalar>::_barrier_decrease)
      .def_readwrite("barrier_increase",
                     &InteriorPointSolver<Scalar>::_barrier_increase)
      .def_readwrite("constraint_padding",
                     &InteriorPointSolver<Scalar>::_constraint_padding)
      .def_readwrite("use_matrices",
                     &InteriorPointSolver<Scalar>::_use_matrices)
      .def_readwrite("use_barrier", &InteriorPointSolver<Scalar>::_use_barrier)
      .def_readwrite("use_objectives",
                     &InteriorPointSolver<Scalar>::_use_objectives)
      .def_readwrite("use_constraints",
                     &InteriorPointSolver<Scalar>::_use_constraints)
      .def_readwrite("use_penalty", &InteriorPointSolver<Scalar>::_use_penalty)
      .def_readwrite("in_constraint_phase",
                     &InteriorPointSolver<Scalar>::_in_constraint_phase)
      .def_readwrite("use_preconditioner",
                     &InteriorPointSolver<Scalar>::_use_preconditioner)

      ;

  py::class_<AdamSolver<Scalar>, GradientDescentSolver<Scalar>>(type_module,
                                                                "AdamSolver")
      .def(py::init<std::shared_ptr<Engine>>());

  py::class_<SpSQPSolver<Scalar>, Solver>(type_module, "SpSQPSolver")
      .def_readwrite("step_scaling", &SpSQPSolver<Scalar>::_step_scaling)
      .def_readwrite("qp_solver", &SpSQPSolver<Scalar>::_qp_solver)
      .def_readwrite("matrix_builder", &SpSQPSolver<Scalar>::_matrix_builder)
      .def_readwrite("backoff_enable", &SpSQPSolver<Scalar>::_backoff_enable)
      .def_readwrite("backoff_factor", &SpSQPSolver<Scalar>::_backoff_factor)
      .def_readwrite("backoff_steps", &SpSQPSolver<Scalar>::_backoff_steps)
      .def_readwrite("time_compute", &SpSQPSolver<Scalar>::_time_compute)
      .def_readwrite("time_prepare", &SpSQPSolver<Scalar>::_time_prepare)
      .def_readwrite("time_matrix", &SpSQPSolver<Scalar>::_time_matrix)
      .def_readwrite("time_select", &SpSQPSolver<Scalar>::_time_select)
      .def_readwrite("time_solve", &SpSQPSolver<Scalar>::_time_solve)
      .def_readwrite("time_negate", &SpSQPSolver<Scalar>::_time_negate)
      .def_readwrite("time_backoff", &SpSQPSolver<Scalar>::_time_backoff)
      .def_readwrite("time_accumulate", &SpSQPSolver<Scalar>::_time_accumulate)
      .def(py::init<std::shared_ptr<Engine>>());

  py::class_<SpQPSolver<Scalar>, std::shared_ptr<SpQPSolver<Scalar>>>(
      type_module, "SpQPSolver");

  py::class_<LambdaSpQPSolver<Scalar>,
             std::shared_ptr<LambdaSpQPSolver<Scalar>>, SpQPSolver<Scalar>>(
      type_module, "LambdaSpQPSolver")
      .def_readwrite("callback", &LambdaSpQPSolver<Scalar>::lambda)
      .def(py::init<>());

  py::class_<SpQPSolverBase<Scalar>, std::shared_ptr<SpQPSolverBase<Scalar>>,
             SpQPSolver<Scalar>>(type_module, "SpQPSolverBase")
      .def_readwrite("finished", &SpQPSolverBase<Scalar>::finished)
      .def_readwrite("infeasible", &SpQPSolverBase<Scalar>::infeasible)
      .def_readwrite("success", &SpQPSolverBase<Scalar>::success)
      .def_readwrite("tolerance", &SpQPSolverBase<Scalar>::tolerance);

  py::class_<SpQP<Scalar>>(type_module, "SpQP")
      .def_readwrite("objective_matrix", &SpQP<Scalar>::objective_matrix)
      .def_readwrite("objective_vector", &SpQP<Scalar>::objective_vector)
      .def_readwrite("equality_matrix", &SpQP<Scalar>::equality_matrix)
      .def_readwrite("equality_vector", &SpQP<Scalar>::equality_vector)
      .def_readwrite("inequality_matrix", &SpQP<Scalar>::inequality_matrix)
      .def_readwrite("inequality_vector", &SpQP<Scalar>::inequality_vector)
      .def(py::init<>());
}

TRACTOR_PYTHON_TYPED(pythonizeSolvers);

}  // namespace tractor
