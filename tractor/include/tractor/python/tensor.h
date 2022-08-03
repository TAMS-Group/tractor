// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/tensor.h>
#include <tractor/core/var.h>
#include <tractor/tensor/ops.h>

namespace tractor {

template <class Scalar>
static void pythonizeTensor(py::module &main_module, py::module &type_module) {

  static auto find_shape = [](const py::array_t<Scalar> &array) {
    std::vector<size_t> ss;
    ss.resize(array.ndim());
    for (size_t i = 0; i < array.ndim(); i++) {
      ss[i] = array.shape(i);
    }
    return TensorShape(ss);
  };

  static auto importTensor = [](const py::array_t<Scalar> &array) {
    auto tensor_shape = find_shape(array);
    auto element_count = tensor_shape.elementCount();
    auto array_data = array.data();
    std::vector<Scalar> tensor_data(element_count);
    for (size_t i = 0; i < element_count; i++) {
      tensor_data[i] = *array_data;
      array_data++;
    }
    return Tensor<Scalar>(tensor_shape, tensor_data.data());
  };

  py::class_<Tensor<Scalar>>(type_module, "Tensor")
      .def(py::init<>())
      .def(py::init(
          [](const py::array_t<Scalar> &a) { return importTensor(a); }))
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
      .def_property(
          "value",
          [](const Tensor<Scalar> &tensor) {
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
          },
          [](Tensor<Scalar> &tensor, const py::array_t<Scalar> &array) {
            tensor = importTensor(array);
          })
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self);

  main_module.def("variable", [](Tensor<Scalar> &v) { variable(v); });
  main_module.def("parameter", [](Tensor<Scalar> &v) { parameter(v); });
  main_module.def("output", [](Tensor<Scalar> &v) { output(v); });
  main_module.def("goal", [](Tensor<Scalar> &v) { goal(v); });

  main_module.def("unpack",
                  [](const Tensor<Scalar> &tensor) { return unpack(tensor); });

  main_module.def("dense_mul_vec_mat", [](const Tensor<Scalar> &activations,
                                          const Tensor<Scalar> &weights) {
    return dense_mul_vec_mat(activations, weights);
  });
}

} // namespace tractor
