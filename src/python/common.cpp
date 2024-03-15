// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

namespace tractor {

void PythonRegistry::run(py::module &m) {
  for (auto &f : _ff) {
    f(m);
  }
}

const std::shared_ptr<PythonRegistry> &PythonRegistry::instance() {
  static std::shared_ptr<PythonRegistry> instance =
      std::make_shared<PythonRegistry>();
  return instance;
}

}  // namespace tractor
