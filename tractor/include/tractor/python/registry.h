// (c) 2020-2022 Philipp Ruppel

#include <pybind11/pybind11.h>

namespace tractor {
void register_python_component(void (*)(pybind11::module &));
}

// #define TRACTOR_PYTHON_REGISTER(...)                                           \
//   static int tractor_python_reg = []() {                                       \
//     register_python_component([](pybind11::module &m) { __VA_ARGS__ });        \
//     return 0;                                                                  \
//   }();
