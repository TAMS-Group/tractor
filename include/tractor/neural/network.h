// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/neural/layer.h>

#include <fstream>
#include <unordered_set>

namespace tractor {

template <class Scalar>
class NeuralNetwork : public NeuralBase {
  std::vector<std::shared_ptr<Layer<Scalar>>> _inputs;
  std::vector<std::shared_ptr<Layer<Scalar>>> _outputs;
  static void _findLayerSet(
      const std::shared_ptr<Layer<Scalar>> &layer,
      std::unordered_set<std::shared_ptr<Layer<Scalar>>> &set) {
    if (set.insert(layer).second) {
      for (auto &l : layer->inputs()) {
        _findLayerSet(l, set);
      }
    }
  }
  std::unordered_set<std::shared_ptr<Layer<Scalar>>> _findLayerSet() const {
    std::unordered_set<std::shared_ptr<Layer<Scalar>>> set;
    for (auto &l : _outputs) {
      _findLayerSet(l, set);
    }
    return set;
  }
  std::vector<std::shared_ptr<Layer<Scalar>>> _findLayerList() const {
    std::vector<std::shared_ptr<Layer<Scalar>>> layers;
    for (auto &l : _findLayerSet()) {
      layers.push_back(l);
    }
    return layers;
  }
  static void _evaluate(const std::shared_ptr<Layer<Scalar>> &layer,
                        const LayerMode &mode,
                        std::unordered_map<std::shared_ptr<Layer<Scalar>>,
                                           Tensor<Scalar>> &tensors) {
    if (tensors.find(layer) != tensors.end()) {
      return;
    }
    std::vector<Tensor<Scalar>> args;
    for (auto &input : layer->inputs()) {
      _evaluate(input, mode, tensors);
      args.push_back(tensors[input]);
    }
    tensors[layer] = layer->evaluate(args, mode);
  }

 public:
  NeuralNetwork() {}
  NeuralNetwork(const std::vector<std::shared_ptr<Layer<Scalar>>> &inputs,
                const std::vector<std::shared_ptr<Layer<Scalar>>> &outputs) {
    init(inputs, outputs);
  }
  void init(const std::vector<std::shared_ptr<Layer<Scalar>>> &inputs,
            const std::vector<std::shared_ptr<Layer<Scalar>>> &outputs) {
    _inputs = inputs;
    _outputs = outputs;
  }
  void clear() {
    _inputs.clear();
    _outputs.clear();
  }
  auto &inputs() const { return _inputs; }
  auto &outputs() const { return _outputs; }
  auto layers() const { return _findLayerList(); }
  std::vector<Tensor<Scalar>> predict(
      const std::vector<Tensor<Scalar>> &inputs,
      const LayerMode &mode = LayerMode()) const {
    if (inputs.size() != _inputs.size()) {
      throw std::runtime_error("wrong number of input tensors");
    }
    std::unordered_map<std::shared_ptr<Layer<Scalar>>, Tensor<Scalar>> tensors;
    for (size_t i = 0; i < inputs.size(); i++) {
      tensors[_inputs.at(i)] = inputs.at(i);
    }
    std::vector<Tensor<Scalar>> ret;
    for (auto &layer : _outputs) {
      _evaluate(layer, mode, tensors);
      ret.push_back(tensors[layer]);
    }
    return ret;
  }
  Tensor<Scalar> predict(const Tensor<Scalar> &input,
                         const LayerMode &mode = LayerMode()) const {
    auto ret = predict(std::vector<Tensor<Scalar>>({input}), mode);
    if (ret.size() != 1) {
      throw std::runtime_error(
          "wrong number of output tensors, pass a list of tensors as inputs to "
          "get a list of output tensors");
    }
    return ret.at(0);
  }
  void serialize(
      const std::function<void(NeuralBase *, void *, size_t)> &fnc) override {
    for (auto &layer : _findLayerSet()) {
      TRACTOR_INFO("serializing layer " << typeid(*layer).name());
      layer->serialize(fnc);
    }
  }
  void serializeWeights(std::ostream &stream) {
    TRACTOR_INFO("begin serializing weights");
    auto callback = [&](NeuralBase *layer, void *ptr, size_t s) {
      TRACTOR_INFO("serializing layer " << typeid(*layer).name() << " " << layer
                                        << " " << ptr << " " << s);
      stream.write((const char *)ptr, s);
    };
    TRACTOR_INFO("serializing weights");
    serialize(callback);
  }
  void saveWeights(const std::string &filename) {
    TRACTOR_INFO("opening file " << filename);
    std::ofstream s(filename);
    TRACTOR_INFO("serializing");
    serializeWeights(s);
  }
  void deserializeWeights(std::istream &stream) {
    serialize([&](NeuralBase *layer, void *ptr, size_t s) {
      stream.read((char *)ptr, s);
    });
  }
  void loadWeights(const std::string &filename) {
    std::ifstream s(filename);
    if (!s) {
      throw std::runtime_error("failed to open weight file " + filename);
    }
    deserializeWeights(s);
  }
};

template <class Scalar>
class SequentialNeuralNetwork : public NeuralNetwork<Scalar> {
  std::shared_ptr<Layer<Scalar>> _input, _output;

 public:
  SequentialNeuralNetwork() {}
  SequentialNeuralNetwork(
      const std::vector<std::shared_ptr<Layer<Scalar>>> &layers) {
    for (auto &layer : layers) {
      add(layer);
    }
  }
  void add(const std::shared_ptr<Layer<Scalar>> &layer) {
    if (!_input) {
      _input = _output = std::make_shared<InputLayer<Scalar>>();
    }
    _output = (*layer)({_output});
    this->init({_input}, {_output});
  }
};

}  // namespace tractor
