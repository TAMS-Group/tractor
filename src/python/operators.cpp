// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/operator.h>

namespace tractor {

static void pythonizeOperators(py::module &m) {}

TRACTOR_PYTHON_GLOBAL(pythonizeOperators);

}  // namespace tractor
