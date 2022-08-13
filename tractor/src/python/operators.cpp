// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/operator.h>

namespace tractor {

static void pythonizeOperators(py::module &m) {
  // auto m_ops = m.def_submodule("ops");
  // for (auto *op : Operator::all()) {
  //   op->pythonize(m);
  // }
}

TRACTOR_PYTHON_GLOBAL(pythonizeOperators);

} // namespace tractor
