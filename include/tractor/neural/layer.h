// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/constraints.h>
#include <tractor/core/enum.h>
#include <tractor/neural/ops.h>
#include <tractor/tensor/ops.h>

#include <random>

namespace tractor {

TRACTOR_ENUM(ActivationType, Linear, TanH, ReLU);

template <class T>
Tensor<T> applyActivation(const Tensor<T> &input_tensor,
                          const ActivationType &activation) {
  switch (activation) {
    case ActivationType::Linear:
      return input_tensor;
      break;
    case ActivationType::TanH:
      return tanh(input_tensor);
      break;
    case ActivationType::ReLU:
      return relu(input_tensor);
      break;
    default:
      throw std::runtime_error("unsupported activation");
      break;
  }
}

struct LayerMode {
  bool training = true;
};

struct NeuralBase {
  virtual ~NeuralBase() {}
  virtual void serialize(
      const std::function<void(NeuralBase *, void *, size_t)> &fnc) {}
};

template <class Scalar>
class Layer : public NeuralBase,
              public std::enable_shared_from_this<Layer<Scalar>> {
 protected:
  std::vector<std::shared_ptr<Layer>> _inputs;

 public:
  const std::vector<std::shared_ptr<Layer>> &inputs() const { return _inputs; }
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) = 0;
  std::shared_ptr<Layer<Scalar>> operator()(
      const std::vector<std::shared_ptr<Layer<Scalar>>> &inputs);
};

template <class Scalar>
class CallLayer : public Layer<Scalar> {
  std::shared_ptr<Layer<Scalar>> _layer;

 public:
  CallLayer(const std::shared_ptr<Layer<Scalar>> &layer,
            const std::vector<std::shared_ptr<Layer<Scalar>>> &inputs) {
    _layer = layer;
    this->_inputs = inputs;
  }
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    return _layer->evaluate(inputs, mode);
  }
  virtual void serialize(
      const std::function<void(NeuralBase *, void *, size_t)> &fnc) override {
    _layer->serialize(fnc);
  }
};

template <class Scalar>
std::shared_ptr<Layer<Scalar>> Layer<Scalar>::operator()(
    const std::vector<std::shared_ptr<Layer<Scalar>>> &inputs) {
  return std::make_shared<CallLayer<Scalar>>(this->shared_from_this(), inputs);
}

template <class Scalar>
class InputLayer : public Layer<Scalar> {
 public:
  InputLayer() {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    throw std::runtime_error("tried to evaluate an input layer");
  }
};

template <class Scalar>
class DenseLayer : public Layer<Scalar> {
  typedef typename BatchScalar<Scalar>::Type WeightScalar;
  size_t _units = 0;
  ActivationType _activation;
  bool _initialized = false;
  Tensor<WeightScalar> _weights;
  Tensor<Scalar> _bias;
  WeightScalar _bias_regularization = 0;
  WeightScalar _weight_regularization = 0;
  WeightScalar _activity_regularization = 0;
  WeightScalar _stdev = 0;
  bool _use_bias = true;
  Tensor<Scalar> _activity_regularization_temp;

  template <class TensorScalar>
  void _randomizeWeights(Tensor<TensorScalar> &tensor,
                         const WeightScalar &stdev) {
    size_t s = tensor.sizeInBytes() / sizeof(WeightScalar);
    static thread_local std::mt19937 gen{std::mt19937::result_type(rand())};
    std::normal_distribution<WeightScalar> dist(0.0, stdev);
    for (size_t i = 0; i < s; i++) {
      ((WeightScalar *)tensor.data())[i] = dist(gen);
    }
  }

 public:
  DenseLayer(size_t units, ActivationType activation = ActivationType::Linear,
             WeightScalar bias_regularization = 0,
             WeightScalar weight_regularization = 0,
             WeightScalar activity_regularization = 0,
             WeightScalar stdev = 0.001, bool use_bias = true)
      : _units(units),
        _activation(activation),
        _bias_regularization(bias_regularization),
        _weight_regularization(weight_regularization),
        _activity_regularization(activity_regularization),
        _stdev(stdev),
        _use_bias(use_bias) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto &input = inputs.at(0);
    // TRACTOR_CHECK_TENSOR_DIMENSIONS("DenseLayer", input, 1);

    if (!_initialized) {
      TRACTOR_DEBUG("dense layer build " << input.shape() << " x " << _units);

      _weights =
          Tensor<WeightScalar>(TensorShape(input.shape().last(), _units));
      _randomizeWeights(_weights, _stdev);
      variable(_weights);

      if (_use_bias) {
        TRACTOR_DEBUG("dense layer create bias");
        _bias = Tensor<Scalar>(TensorShape(_units));
        _randomizeWeights(_bias, _stdev);
        variable(_bias);
      }

      if (_weight_regularization > 0) {
        TRACTOR_DEBUG("dense layer weight regularization "
                      << _weight_regularization);
        goal(_weights * make_tensor(_weights.shape(), _weight_regularization));
      }

      if (_use_bias && _bias_regularization > 0) {
        TRACTOR_DEBUG("dense layer bias regularization "
                      << _bias_regularization);
        auto reg_tens =
            make_tensor(_bias.shape(), Scalar(_bias_regularization));
        TRACTOR_DEBUG("bias regularization " << _bias.shape() << " "
                                             << reg_tens.shape());
        goal(_bias * reg_tens);
      }
    }

    TRACTOR_DEBUG("dense layer emit dense op");
    Tensor<Scalar> activity = matmul(input, _weights);

    if (_use_bias) {
      TRACTOR_DEBUG("dense layer apply bias");
      activity = neural_bias(activity, _bias);
    }

    if (_activity_regularization > 0) {
      TRACTOR_DEBUG("dense layer apply activity regularization");
      goal(activity *
           make_tensor(activity.shape(), Scalar(_activity_regularization)));
    }

    _initialized = true;

    return applyActivation(activity, _activation);
  }
  auto &weights() const { return _weights; }
  auto &bias() const { return _bias; }
  virtual void serialize(
      const std::function<void(NeuralBase *, void *, size_t)> &fnc) override {
    TRACTOR_INFO("serializing dense layer " << _weights.shape());
    if (_use_bias) {
      fnc(this, _bias.data(), _bias.sizeInBytes());
    }
    fnc(this, _weights.data(), _weights.sizeInBytes());
  }
};

template <class Scalar>
class ActivationLayer : public Layer<Scalar> {
  ActivationType _activation = ActivationType::Linear;

 public:
  ActivationLayer(const ActivationType &activation) : _activation(activation) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    return applyActivation(inputs.at(0), _activation);
  }
};

template <class Scalar>
class LambdaLayer : public Layer<Scalar> {
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
class GaussianNoiseLayer : public Layer<Scalar> {
  typedef typename BatchScalar<Scalar>::Type WeightScalar;
  WeightScalar _standard_deviation = 0.0;

 public:
  GaussianNoiseLayer(const WeightScalar &standard_deviation)
      : _standard_deviation(standard_deviation) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto activations = inputs.at(0);
    if (mode.training) {
      activations = add_random_normal(
          activations,
          make_tensor(activations.shape(), WeightScalar(_standard_deviation)));
    }
    return activations;
  }
};

template <class Scalar>
class DropoutLayer : public Layer<Scalar> {
  typedef typename BatchScalar<Scalar>::Type WeightScalar;
  WeightScalar _rate = 0.0;

 public:
  DropoutLayer(const WeightScalar &rate) : _rate(rate) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto activations = inputs.at(0);
    if (mode.training) {
      activations = dropout(
          activations, make_tensor(activations.shape(), WeightScalar(_rate)));
    }
    return activations;
  }
};

template <class Scalar>
class ActivityRegularizationLayer : public Layer<Scalar> {
  double _l2 = 0.0;
  Tensor<Scalar> _l2_temp;
  bool _initialized = false;

 public:
  ActivityRegularizationLayer(const double &l2) : _l2(l2) {}
  virtual Tensor<Scalar> evaluate(const std::vector<Tensor<Scalar>> &inputs,
                                  const LayerMode &mode) override {
    auto &activity = inputs.front();
    if (!_initialized) {
      _initialized = true;
      _l2_temp = make_tensor(activity.shape(), Scalar(_l2));
    }
    goal(activity * _l2_temp);
    return activity;
  }
};

}  // namespace tractor
