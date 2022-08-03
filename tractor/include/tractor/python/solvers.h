// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/solver.h>
#include <tractor/neural/network.h>
#include <tractor/solvers/gd.h>
#include <tractor/solvers/sq.h>

namespace tractor {

template <class Scalar>
static void pythonizeSolvers(py::module &main_module, py::module &type_module) {

  py::class_<LeastSquaresSolver<Scalar>, Solver>(type_module,
                                                 "LeastSquaresSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("regularization",
                     &LeastSquaresSolver<Scalar>::_regularization)
      .def_readwrite("step_scaling", &LeastSquaresSolver<Scalar>::_step_scaling)
      .def_readwrite("max_linear_iterations",
                     &LeastSquaresSolver<Scalar>::_max_linear_iterations);

  py::class_<GradientDescentSolver<Scalar>, Solver>(type_module,
                                                    "GradientDescentSolver")
      .def(py::init<std::shared_ptr<Engine>>())
      .def_readwrite("learning_rate",
                     &GradientDescentSolver<Scalar>::_learning_rate);
}

} // namespace tractor
