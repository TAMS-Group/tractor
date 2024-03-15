// 2022-2024 Philipp Ruppel

#include <pybind11/pybind11.h>

namespace tractor {
void initTractorPython(pybind11::module &);
}

PYBIND11_MODULE(tractor, m) { tractor::initTractorPython(m); }
