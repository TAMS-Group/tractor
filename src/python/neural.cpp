// 2022-2024 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/neural/network.h>

namespace tractor {

static void pythonizeNeuralGlobal(py::module &m) {
  {
    auto e = py::enum_<ActivationType>(m, "ActivationType");
    for (auto &p : enumerateEachActivationType()) {
      e.value(p.second.c_str(), p.first);
    }
    e.def(
        py::init([](const std::string &s) { return parseActivationType(s); }));
    py::implicitly_convertible<std::string, ActivationType>();
  }
}

TRACTOR_PYTHON_GLOBAL(pythonizeNeuralGlobal);

template <class Scalar>
static void pythonizeNeural(py::module main_module, py::module type_module) {
  typedef typename BatchScalar<Scalar>::Type WeightScalar;

  py::class_<NeuralNetwork<Scalar>, std::shared_ptr<NeuralNetwork<Scalar>>>(
      type_module, "NeuralNetwork")
      .def_property_readonly(
          "layers",
          [](const NeuralNetwork<Scalar> &network) { return network.layers(); })
      .def_property_readonly(
          "inputs",
          [](const NeuralNetwork<Scalar> &network) { return network.inputs(); })
      .def_property_readonly("outputs",
                             [](const NeuralNetwork<Scalar> &network) {
                               return network.outputs();
                             })
      .def("predict",
           [](NeuralNetwork<Scalar> &network, const Tensor<Scalar> &input) {
             return network.predict(input);
           })
      .def("predict",
           [](NeuralNetwork<Scalar> &network,
              const std::vector<Tensor<Scalar>> &inputs) {
             return network.predict(inputs);
           })
      .def("__call__",
           [](NeuralNetwork<Scalar> &network, const Tensor<Scalar> &input) {
             return network.predict(input);
           })
      .def("__call__",
           [](NeuralNetwork<Scalar> &network,
              const std::vector<Tensor<Scalar>> &inputs) {
             return network.predict(inputs);
           })
      .def(py::init<>())
      .def(py::init<const std::vector<std::shared_ptr<Layer<Scalar>>> &,
                    const std::vector<std::shared_ptr<Layer<Scalar>>> &>());

  py::class_<SequentialNeuralNetwork<Scalar>,
             std::shared_ptr<SequentialNeuralNetwork<Scalar>>,
             NeuralNetwork<Scalar>  //
             >(type_module, "SequentialNeuralNetwork")
      .def(py::init<>())
      .def(py::init<const std::vector<std::shared_ptr<Layer<Scalar>>> &>())
      .def("add", &SequentialNeuralNetwork<Scalar>::add)

      ;

  py::class_<Layer<Scalar>, std::shared_ptr<Layer<Scalar>>>(type_module,
                                                            "Layer")
      .def("__call__",
           [](Layer<Scalar> &layer, const std::shared_ptr<Layer<Scalar>> &a) {
             return layer({a});
           })
      .def("__call__",
           [](Layer<Scalar> &layer, const std::shared_ptr<Layer<Scalar>> &a,
              const std::shared_ptr<Layer<Scalar>> &b) {
             return layer({a, b});
           });

  py::class_<CallLayer<Scalar>, std::shared_ptr<CallLayer<Scalar>>,
             Layer<Scalar>>(type_module, "CallLayer");

  py::class_<InputLayer<Scalar>, std::shared_ptr<InputLayer<Scalar>>,
             Layer<Scalar>>(type_module, "InputLayer")
      .def(py::init<>());

  py::class_<ActivationLayer<Scalar>, std::shared_ptr<ActivationLayer<Scalar>>,
             Layer<Scalar>>(type_module, "ActivationLayer")
      .def(py::init<ActivationType>());

  py::class_<GaussianNoiseLayer<Scalar>,
             std::shared_ptr<GaussianNoiseLayer<Scalar>>, Layer<Scalar>>(
      type_module, "GaussianNoiseLayer")
      .def(py::init<WeightScalar>());

  py::class_<DropoutLayer<Scalar>, std::shared_ptr<DropoutLayer<Scalar>>,
             Layer<Scalar>>(type_module, "DropoutLayer")
      .def(py::init<WeightScalar>());

  py::class_<DenseLayer<Scalar>, std::shared_ptr<DenseLayer<Scalar>>,
             Layer<Scalar>>(type_module, "DenseLayer")
      .def(py::init<size_t, ActivationType, Scalar, Scalar, Scalar, Scalar,
                    bool>(),
           py::arg("units"), py::arg("activation") = ActivationType::Linear,
           py::arg("bias_regularization") = Scalar(0.0),
           py::arg("weight_regularization") = Scalar(0.0),
           py::arg("activity_regularization") = Scalar(0.0),
           py::arg("stdev") = Scalar(0.001), py::arg("use_bias") = true)
      .def_property_readonly(
          "bias", [](const DenseLayer<Scalar> &layer) { return layer.bias(); })
      .def_property_readonly("weights", [](const DenseLayer<Scalar> &layer) {
        return layer.weights();
      });
}

TRACTOR_PYTHON_TYPED(pythonizeNeural);

}  // namespace tractor
