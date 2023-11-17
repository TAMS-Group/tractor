// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/solver.h>
#include <tractor/core/sparsity.h>
#include <tractor/neural/network.h>
#include <tractor/solvers/gd.h>
#include <tractor/solvers/spsq.h>
#include <tractor/solvers/sq.h>

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

  py::class_<AdamSolver<Scalar>, GradientDescentSolver<Scalar>>(type_module,
                                                                "AdamSolver")
      .def(py::init<std::shared_ptr<Engine>>());
}

TRACTOR_PYTHON_TYPED(pythonizeSolvers);

}  // namespace tractor
