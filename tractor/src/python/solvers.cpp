// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/solver.h>
#include <tractor/neural/network.h>
#include <tractor/solvers/gd.h>
#include <tractor/solvers/sq.h>

namespace tractor {

static void pythonizeProgramGlobal(py::module &m) {

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
      .def_property("tolerance", &Solver::tolerance, &Solver::setTolerance)
      .def_property(
          "timeout", [](const Solver &solver) { return solver.timeout(); },
          [](Solver &solver, const double &v) { solver.setTimeout(v, false); });
}

TRACTOR_PYTHON_GLOBAL(pythonizeProgramGlobal);

template <class Scalar>
static void pythonizeSolvers(py::module &main_module, py::module &type_module) {

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
}

TRACTOR_PYTHON_TYPED(pythonizeSolvers);

} // namespace tractor
