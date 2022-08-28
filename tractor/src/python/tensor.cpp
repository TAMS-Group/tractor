// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/tensor.h>
#include <tractor/core/var.h>
#include <tractor/neural/ops.h>
#include <tractor/tensor/ops.h>

namespace tractor {

template <class Scalar, class Enable = void> struct PythonTensorUtils;

template <class Scalar>
struct PythonTensorUtils<
    Scalar, typename std::enable_if_t<std::is_pod<Scalar>::value, void>::type> {
  static TensorShape findShape(const py::array_t<Scalar> &array) {
    std::vector<size_t> ss;
    ss.resize(array.ndim());
    for (size_t i = 0; i < array.ndim(); i++) {
      ss[i] = array.shape(i);
    }
    return TensorShape(ss);
  };
  static Tensor<Scalar> makeTensor(const py::array_t<Scalar> &array) {
    auto tensor_shape = findShape(array);
    auto element_count = tensor_shape.elementCount();
    auto array_data = array.data();
    std::vector<Scalar> tensor_data(element_count);
    for (size_t i = 0; i < element_count; i++) {
      tensor_data[i] = *array_data;
      array_data++;
    }
    return Tensor<Scalar>(tensor_shape, tensor_data.data());
  }
  static void setValue(Tensor<Scalar> &tensor,
                       const py::array_t<Scalar> &array) {
    auto tensor_shape = find_shape(array);
    if (tensor_shape != tensor.shape()) {
      if (tensor.empty()) {
        tensor = Tensor<Scalar>(tensor_shape);
      } else {
        throw std::runtime_error("tensor shape mismatch");
      }
    }
    auto element_count = tensor_shape.elementCount();
    auto array_data = array.data();
    for (size_t i = 0; i < element_count; i++) {
      tensor.data()[i] = *array_data;
      array_data++;
    }
  }
  static py::array_t<Scalar> getValue(const Tensor<Scalar> &tensor) {
    py::array_t<Scalar> ret;
    ret.resize(tensor.shape());
    {
      auto r = ret.mutable_data();
      for (size_t i = 0; i < tensor.shape().elementCount(); i++) {
        *r = tensor.data()[i];
        r++;
      }
    }
    return ret;
  }
};

template <class Scalar>
struct PythonTensorUtils<
    Scalar, typename std::enable_if_t<!std::is_pod<Scalar>::value>::type> {
  static Tensor<Scalar> makeTensor(const std::vector<Scalar> &array) {
    return pack_tensor(array);
  }
  static void setValue(Tensor<Scalar> &tensor,
                       const std::vector<Scalar> &array) {
    TRACTOR_ASSERT(tensor.shape().dimensions() == 1);
    auto tensor_shape = TensorShape({array.size()});
    if (tensor_shape != tensor.shape()) {
      if (tensor.empty()) {
        tensor = Tensor<Scalar>(tensor_shape);
      } else {
        throw std::runtime_error("tensor shape mismatch");
      }
    }
    auto element_count = tensor_shape.elementCount();
    for (size_t i = 0; i < element_count; i++) {
      tensor.data()[i] = array[i];
    }
  }
  static std::vector<Scalar> getValue(const Tensor<Scalar> &tensor) {
    TRACTOR_ASSERT(tensor.shape().dimensions() == 1);
    std::vector<Scalar> ret(tensor.shape().elementCount());
    for (size_t i = 0; i < ret.size(); i++) {
      ret[i] = tensor.data()[i];
    }
    return ret;
  }
};

template <class Scalar>
static void pythonizeTensor(py::module main_module, py::module type_module) {

  pythonizeType<Tensor<Scalar>>(main_module, type_module, "Tensor")
      .def(py::init(&PythonTensorUtils<Scalar>::makeTensor))
      .def(py::init(
          [](const std::vector<Var<Scalar>> &a) { return pack_tensor(a); }))
      .def("copy",
           [](const Tensor<Scalar> &v) {
             Tensor<Scalar> r = v;
             return r;
           })
      .def_property_readonly("shape",
                             [](const Tensor<Scalar> &t) {
                               const auto &s = t.shape();
                               py::tuple ret = py::tuple(s.dimensions());
                               for (size_t i = 0; i < s.dimensions(); i++) {
                                 ret[i] = s[i];
                               }
                               return ret;
                             })
      .def_property_readonly(
          "dimensions",
          [](const Tensor<Scalar> &t) { return t.shape().dimensions(); })
      .def_property("value", &PythonTensorUtils<Scalar>::getValue,
                    &PythonTensorUtils<Scalar>::setValue)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self *= py::self)
      .def(py::self /= py::self);

  main_module.def("unpack",
                  [](const Tensor<Scalar> &tensor) { return unpack(tensor); });

  main_module.def("matmul", &matmul<Scalar, Scalar>);

  main_module.def("neural_bias", &neural_bias<Scalar, Scalar>);

  // type_module.def(
  //     "make_tensor",
  //     [](const std::vector<size_t> &shape, const Scalar &v) -> Tensor<Scalar>
  //     {
  //       return make_tensor(TensorShape(shape), v);
  //     });
}

TRACTOR_PYTHON_TYPED_BATCH(pythonizeTensor);

} // namespace tractor
