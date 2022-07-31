// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/constraints.h>
#include <tractor/neural/synapse.h>

namespace tractor {

enum class Activation {
  Linear,
  TanH,
  ReLU,
};

template <class T>
Tensor<T> applyActivation(const Tensor<T> &input_tensor,
                          const Activation &activation);

struct LayerMode {
  bool training = true;
};

struct NeuralBase {
  virtual ~NeuralBase() {}
  virtual void
  serialize(const std::function<void(NeuralBase *, void *, size_t)> &fnc) {}
};

template <class Scalar> class Layer : public NeuralBase {
public:
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) = 0;
};

template <class Scalar, class Impl> class LayerBase : public Layer<Scalar> {
public:
  template <class... Inputs>
  Synapse<Scalar> operator()(const Inputs &...inputs) {
    return Synapse<Scalar>(0, *(Impl *)this, inputs...);
  }
};

template <class Scalar> class InputLayer : public Layer<Scalar> {
  size_t _units = 0;

public:
  InputLayer(size_t units) : _units(units) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    throw std::runtime_error("tried to evaluate an input layer");
  }
};

template <class Scalar> class Input : public Synapse<Scalar> {
public:
  Input(size_t units) : Synapse<Scalar>(0, InputLayer<Scalar>(units)) {}
};

template <class T> void variable(Tensor<T> &tensor) {
  size_t s = tensor.size();
  for (size_t i = 0; i < s; i++) {
    tractor::variable(tensor[i]);
  }
}

template <class Scalar, class StDev>
void randomize(Tensor<Scalar> &tensor, const StDev &stdev) {
  size_t s = tensor.size();
  static thread_local std::mt19937 gen{std::mt19937::result_type(rand())};
  std::normal_distribution<double> dist(0.0, stdev);
  for (size_t i = 0; i < s; i++) {
    tensor[i] = dist(gen);
  }
}

template <class Scalar>
class DenseLayer : public LayerBase<Scalar, DenseLayer<Scalar>> {
  typedef Var<typename BatchScalar<typename Scalar::Value>::Type> WeightScalar;
  size_t _units = 0;
  Activation _activation;
  bool _initialized = false;
  Tensor<WeightScalar> _weights;
  Tensor<WeightScalar> _bias;
  double _bias_regularization = 0;
  double _weight_regularization = 0;
  double _activity_regularization = 0;
  double _stdev = 0;
  bool _use_bias = true;

public:
  DenseLayer(size_t units, Activation activation = Activation::Linear,
             double bias_regularization = 0, double weight_regularization = 0,
             double activity_regularization = 0, double stdev = 0.001,
             bool use_bias = true)
      : _units(units), _activation(activation),
        _bias_regularization(bias_regularization),
        _weight_regularization(weight_regularization),
        _activity_regularization(activity_regularization), _stdev(stdev),
        _use_bias(use_bias) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override;
  auto &weights() const { return _weights; }
  auto &bias() const { return _bias; }
  virtual void serialize(
      const std::function<void(NeuralBase *, void *, size_t)> &fnc) override;
};

template <class Scalar>
class ActivationLayer : public LayerBase<Scalar, ActivationLayer<Scalar>> {
  Activation _activation = Activation::Linear;

public:
  ActivationLayer(Activation activation) : _activation(activation) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto &input = inputs.at(0);
    return applyActivation(input, _activation);
  }
};

template <class Scalar>
class LambdaLayer : public LayerBase<Scalar, LambdaLayer<Scalar>> {
  std::function<Tensor<Scalar>(const Tensor<Scalar>)> _lambda;

public:
  LambdaLayer(const std::function<Tensor<Scalar>(const Tensor<Scalar>)> &lambda)
      : _lambda(lambda) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    return _lambda(inputs.at(0));
  }
};

template <class Scalar>
class GaussianNoiseLayer
    : public LayerBase<Scalar, GaussianNoiseLayer<Scalar>> {
  double _standard_deviation = 0.0;

public:
  GaussianNoiseLayer(const double &standard_deviation)
      : _standard_deviation(standard_deviation) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto activations = inputs.at(0);
    if (mode.training) {
      for (size_t i = 0; i < activations.size(); i++) {
        activations[i] = add_random_normal(activations[i], _standard_deviation);
      }
    }
    return activations;
  }
};

template <class Scalar>
class DropoutLayer : public LayerBase<Scalar, DropoutLayer<Scalar>> {
  double _rate = 0.0;

public:
  DropoutLayer(const double &rate) : _rate(rate) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto activations = inputs.at(0);
    if (mode.training) {
      for (size_t i = 0; i < activations.size(); i++) {
        activations[i] = dropout(activations[i], _rate);
      }
    }
    return activations;
  }
};

template <class Scalar>
class ActivityRegularizationLayer
    : public LayerBase<Scalar, ActivityRegularizationLayer<Scalar>> {
  double _l2 = 0.0;

public:
  ActivityRegularizationLayer(const double &l2) : _l2(l2) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto activations = inputs.at(0);
    if (_l2 > 0) {
      Scalar l2 = typename Scalar::Value(_l2);
      for (size_t i = 0; i < activations.size(); i++) {
        goal(activations[i] * l2);
      }
    }
    return activations;
  }
};

} // namespace tractor
