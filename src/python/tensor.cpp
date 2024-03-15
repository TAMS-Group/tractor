// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/core/tensor.h>
#include <tractor/core/var.h>
#include <tractor/geometry/matrix3.h>
#include <tractor/geometry/matrix3_ops.h>
#include <tractor/geometry/pose.h>
#include <tractor/geometry/pose_ops.h>
#include <tractor/geometry/quaternion.h>
#include <tractor/geometry/quaternion_ops.h>
#include <tractor/geometry/vector3.h>
#include <tractor/geometry/vector3_ops.h>
#include <tractor/neural/ops.h>
#include <tractor/tensor/ops.h>

namespace tractor {

template <class Scalar, int Options>
static TensorShape findTensorPythonShape(
    const py::array_t<Scalar, Options> &array) {
  std::vector<size_t> ss;
  ss.resize(array.ndim());
  for (size_t i = 0; i < array.ndim(); i++) {
    ss[i] = array.shape(i);
  }
  return TensorShape(ss);
}

template <class Scalar, class Class>
static void initTensorClass(
    typename std::enable_if_t<std::is_pod<Scalar>::value, Class> &cls) {
  cls.def(py::init(
      [](const py::array_t<Scalar, py::array::c_style | py::array::forcecast>
             &array) {
        auto tensor_shape = findTensorPythonShape(array);
        auto element_count = tensor_shape.elementCount();
        auto array_data = array.data();
        std::vector<Scalar> tensor_data(element_count);
        for (size_t i = 0; i < element_count; i++) {
          tensor_data[i] = *array_data;
          array_data++;
        }
        return Tensor<Scalar>(tensor_shape, tensor_data.data());
      }));

  cls.def(py::init([](const std::vector<Var<Scalar>> &array) {
    return pack_tensor(array);
  }));

  cls.def_property(
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
      [](Tensor<Scalar> &tensor,
         const py::array_t<Scalar, py::array::c_style | py::array::forcecast>
             &array) {
        auto tensor_shape = findTensorPythonShape(array);
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
      });
}

template <class Scalar, class Class>
static void initTensorClass(
    typename std::enable_if_t<!std::is_pod<Scalar>::value, Class> &cls) {
  cls.def(py::init([](const std::vector<Scalar> &array) {
    return Tensor<Scalar>(TensorShape({array.size()}), array.data());
  }));

  cls.def(py::init([](const std::vector<Var<Scalar>> &array) {
    return pack_tensor(array);
  }));

  cls.def_property(
      "value",
      [](const Tensor<Scalar> &tensor) {
        TRACTOR_ASSERT(tensor.shape().dimensions() == 1);
        std::vector<Scalar> ret(tensor.shape().elementCount());
        for (size_t i = 0; i < ret.size(); i++) {
          ret[i] = tensor.data()[i];
        }
        return ret;
      },
      [](Tensor<Scalar> &tensor, std::vector<Scalar> &array) {
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
      });
}

template <class Scalar>
static auto pythonizeBase(py::module main_module, py::module type_module,
                          const char *name) {
  auto cls =
      pythonizeType<Tensor<Scalar>>(main_module, type_module, name)
          .def(py::init<>())
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
          .def_property_readonly("dimensions", [](const Tensor<Scalar> &t) {
            return t.shape().dimensions();
          });
  initTensorClass<Scalar, decltype(cls)>(cls);
  main_module.def("unpack",
                  [](const Tensor<Scalar> &tensor) { return unpack(tensor); });
  return cls;
}

template <class Scalar>
static void pythonizeTensor(py::module main_module, py::module type_module) {
  pythonizeBase<Scalar>(main_module, type_module, "Tensor")
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self += py::self)
      .def(py::self -= py::self)
      .def(py::self *= py::self)
      .def(py::self /= py::self);
  main_module.def("matmul", &matmul<Scalar, Scalar>);
  main_module.def("neural_bias", &neural_bias<Scalar, Scalar>);

  pythonizeBase<Vector3<Scalar>>(main_module, type_module, "Vector3Tensor")
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self += py::self)
      .def(py::self -= py::self);

  pythonizeBase<Quaternion<Scalar>>(main_module, type_module,
                                    "QuaternionTensor")
      .def(py::self * py::self)
      .def(py::self *= py::self);

  pythonizeBase<Pose<Scalar>>(main_module, type_module, "PoseTensor")
      .def(py::self * py::self)
      .def(py::self *= py::self);

  pythonizeBase<Matrix3<Scalar>>(main_module, type_module, "Matrix3Tensor");
}
TRACTOR_PYTHON_TYPED_BATCH(pythonizeTensor);

}  // namespace tractor
