// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/log.h>
#include <tractor/core/profiler.h>

#include <pybind11/eval.h>

#include <boost/stacktrace.hpp>

#include <signal.h>

#include <mcheck.h>

namespace tractor {

template <class Scalar>
struct PythonScalarInit {
  template <class Class>
  static void init(Class cls) {
    cls.def(py::init<Scalar>());
    cls.def_property(
        "value", [](const Var<Scalar> &_this) { return (Scalar)_this.value(); },
        [](Var<Scalar> &_this, const Scalar &v) { _this.value() = v; });
  }
};

template <class Scalar, size_t Size>
struct PythonScalarInit<Batch<Scalar, Size>> {
  template <class Class>
  static void init(Class cls) {
    cls.def(py::init([](const std::array<Scalar, Size> &v) {
      Batch<Scalar, Size> _this;
      for (size_t i = 0; i < Size; i++) {
        _this[i] = v[i];
      }
      return Var<Batch<Scalar, Size>>(_this);
    }));
    cls.def_property(
        "value",
        [](const Var<Batch<Scalar, Size>> &_this) {
          std::array<Scalar, Size> ret;
          for (size_t i = 0; i < Size; i++) {
            ret[i] = _this.value()[i];
          }
          return ret;
        },
        [](Var<Batch<Scalar, Size>> &_this, const std::array<Scalar, Size> &v) {
          for (size_t i = 0; i < Size; i++) {
            _this.value()[i] = v[i];
          }
        });
  }
};

template <class Scalar>
static void pythonizeGeometryScalar(py::module main_module,
                                    py::module type_module) {
  auto cls = pythonizeType<Var<Scalar>>(main_module, type_module, "Scalar")
                 .def(py::init<>())
                 .def(py::self + py::self)
                 .def(py::self - py::self)
                 .def(py::self * py::self)
                 .def(py::self / py::self)
                 .def(py::self += py::self)
                 .def(py::self -= py::self)
                 .def(py::self *= py::self)
                 .def(py::self /= py::self)
                 .def(-py::self);
  PythonScalarInit<Scalar>::init(cls);
}
TRACTOR_PYTHON_TYPED_BATCH(pythonizeGeometryScalar);

static void pythonizeMain(py::module &m) {
  m.def("print_info", [](const std::string &s) { TRACTOR_INFO(s); });

  m.def("print_debug", [](const std::string &s) { TRACTOR_DEBUG(s); });

  m.def("print_error", [](const std::string &s) { TRACTOR_ERROR(s); });

  m.def("print_success", [](const std::string &s) { TRACTOR_SUCCESS(s); });

  m.def("debug", []() {
    static auto printStackTrace = []() {
      TRACTOR_INFO(boost::stacktrace::stacktrace());
    };
    signal(SIGSEGV, [](int sig) {
      printStackTrace();
      TRACTOR_FATAL("SEGFAULT");
      exit(-1);
    });
    signal(SIGFPE, [](int sig) {
      printStackTrace();
      TRACTOR_FATAL("SIGFPE");
      exit(-1);
    });
    std::set_terminate([]() {
      printStackTrace();
      TRACTOR_FATAL("uncaught exception");
      exit(-1);
    });
  });
  m.def("throw_runtime_error",
        [](const std::string &s) { throw std::runtime_error(s); });
  m.def("raise_segfault", []() { raise(SIGSEGV); });
  m.def("raise_fpe", []() {
    int a = 0;
    int c = 1 / a;
  });

  m.attr("__version__") = "0.0.0";

  m.def_submodule("types_float_4");
  m.def_submodule("types_double_4");

  m.def_submodule("types_float");
  m.def_submodule("types_double");

  m.def_submodule("types_float_scalar");
  m.def_submodule("types_double_scalar");

  struct PythonObjectHolder {
    pybind11::object object;
    PythonObjectHolder(const pybind11::object &object) : object(object) {}
    virtual ~PythonObjectHolder() {}
  };

  m.def("variable", [](const py::object &o) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(std::make_shared<PythonObjectHolder>(o));
    }
    o.attr("_internal_make_variable")();
  });

  m.def("parameter", [](const py::object &o) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(std::make_shared<PythonObjectHolder>(o));
    }
    o.attr("_internal_make_parameter")();
  });

  m.def("output", [](const py::object &o) {
    if (auto *rec = Recorder::instance()) {
      rec->reference(std::make_shared<PythonObjectHolder>(o));
    }
    o.attr("_internal_make_output")();
  });

  class Log {};
  py::class_<Log>(m, "logger")
      .def_property_static(
          "verbosity", [](py::object) { return getLogVerbosity(); },
          [](py::object, int v) { setLogVerbosity(v); });

  PythonRegistry::instance()->run(m);

  auto profiler = m.def_submodule("profiler");
  profiler.def("start", []() { tractor::ProfilerThread::start(); });
  profiler.def("start", [](double interval) {
    tractor::ProfilerThread::start(interval);
  });

  for (auto *op : Operator::all()) {
    op->pythonize(m);
  }
}

void initTractorPython(pybind11::module &m) {
  TRACTOR_DEBUG("building module");
  tractor::pythonizeMain(m);
}

}  // namespace tractor
