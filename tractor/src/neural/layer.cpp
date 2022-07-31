// (c) 2020-2022 Philipp Ruppel

#include <tractor/neural/layer.h>

#include <tractor/core/ops.h>
#include <tractor/core/program.h>
#include <tractor/neural/ops.h>
#include <tractor/tensor/ops.h>

#include <map>
#include <unordered_map>

namespace tractor {

// TODO: generate new type for tensor!;

template <class T>
Tensor<T> applyActivation(const Tensor<T> &input_tensor,
                          const Activation &activation) {
  Tensor<T> tensor = input_tensor;
  size_t tensor_size = tensor.size();
  switch (activation) {
  case Activation::Linear:
    break;
  case Activation::TanH:
    for (size_t i = 0; i < tensor_size; i++) {
      tensor[i] = tanh(tensor[i]);
    }
    break;
  case Activation::ReLU:
    for (size_t i = 0; i < tensor_size; i++) {
      tensor[i] = relu(tensor[i]);
    }
    break;
  default:
    throw std::runtime_error("unsupported activation");
    break;
  }
  return tensor;
}

template Tensor<Var<Batch<double, 4>>>
applyActivation(const Tensor<Var<Batch<double, 4>>> &input_tensor,
                const Activation &activation);

template <class T, size_t N>
std::ostream &operator<<(std::ostream &stream, const TensorArg<T, N> &arg) {
  stream << "tensor[" << arg.dimensions();
  for (size_t i = 0; i < arg.dimensions(); i++) {
    stream << "," << arg.shape()[i];
  }
  stream << "]";
  return stream;
}

template <class T> T batchSum(const T &v) { return v; }
template <class T, size_t N> T batchSum(const Batch<T, N> &v) {
  T ret = T(0);
  for (size_t i = 0; i < N; i++) {
    ret += v[i];
  }
  return ret;
}

template <class Scalar>
Tensor<Scalar>
DenseLayer<Scalar>::evaluate(const std::vector<Tensor<Scalar>> &inputs,
                             const LayerMode &mode) {
  auto &input = inputs.at(0);
  TRACTOR_CHECK_TENSOR_DIMENSIONS(input, 1);

  if (!_initialized) {
    _initialized = true;
    std::cout << "build dense layer " << input.size() << " x " << _units
              << std::endl;

    _weights.resize(input.size(), _units);
    randomize(_weights, _stdev);
    variable(_weights);

    if (_use_bias) {
      _bias.resize(_units);
      randomize(_bias, _stdev);
      variable(_bias);
    }

    if (_weight_regularization != 0) {
      for (size_t row = 0; row < input.size(); row++) {
        for (size_t col = 0; col < _units; col++) {
          goal(_weights(row, col) * _weight_regularization);
        }
      }
    }

    if (_use_bias) {
      if (_bias_regularization != 0) {
        for (size_t i = 0; i < _units; i++) {
          goal(_bias(i) * _bias_regularization);
        }
      }
    }
  }

  Tensor<Scalar> activity = dense_mul_vec_mat(input, _weights);

  if (_use_bias) {
    for (size_t i = 0; i < _units; i++) {
      Scalar bias;
      batch(_bias[i], bias);
      activity[i] += bias;
    }
  }

  if (_activity_regularization != 0) {
    for (size_t i = 0; i < _units; i++) {
      goal(activity(i) * typename Scalar::Value(_activity_regularization));
    }
  }

  return applyActivation(activity, _activation);
}

template <class Scalar>
void DenseLayer<Scalar>::serialize(
    const std::function<void(NeuralBase *, void *, size_t)> &fnc) {
  fnc(this, _bias.data(), _bias.bytes());
  fnc(this, _weights.data(), _weights.bytes());
}

// template class DenseLayer<double>;
// template class DenseLayer<float>;
//
// template class DenseLayer<Batch<double, 4>>;
// template class DenseLayer<Batch<double, 8>>;
// template class DenseLayer<Batch<double, 16>>;
//
// template class DenseLayer<Batch<float, 4>>;
// template class DenseLayer<Batch<float, 8>>;
// template class DenseLayer<Batch<float, 16>>;

// template class DenseLayer<Var<double>>;
// template class DenseLayer<Var<float>>;

template class DenseLayer<Var<Batch<double, 4>>>;
template class DenseLayer<Var<Batch<double, 8>>>;
template class DenseLayer<Var<Batch<double, 16>>>;

// template class DenseLayer<Var<Batch<float, 4>>>;
// template class DenseLayer<Var<Batch<float, 8>>>;
// template class DenseLayer<Var<Batch<float, 16>>>;

} // namespace tractor
