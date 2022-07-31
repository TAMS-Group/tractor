// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/neural/layer.h>

#include <fstream>
#include <unordered_set>

namespace tractor {

template <class Scalar> class NeuralNetwork : public NeuralBase {
  std::unordered_set<std::shared_ptr<Layer<Scalar>>> _input_set;
  std::unordered_set<std::shared_ptr<Layer<Scalar>>> _layer_set;
  std::vector<std::shared_ptr<Layer<Scalar>>> _inputs;
  std::vector<std::shared_ptr<Layer<Scalar>>> _outputs;
  std::vector<std::shared_ptr<Layer<Scalar>>> _layers;
  struct Op {
    std::shared_ptr<Layer<Scalar>> layer;
    std::vector<std::shared_ptr<Layer<Scalar>>> inputs;
  };
  std::vector<Op> _ops;
  void _findLayers(const Synapse<Scalar> &synapse) {
    if (_layer_set.insert(synapse.layer()).second) {
      _layers.push_back(synapse.layer());
      if (_input_set.find(synapse.layer()) == _input_set.end()) {
        Op op;
        op.layer = synapse.layer();
        for (auto &input : synapse.inputs()) {
          op.inputs.push_back(input.layer());
        }
        _ops.push_back(op);
      }
    }
    for (auto &input : synapse.inputs()) {
      _findLayers(input);
    }
  }

public:
  NeuralNetwork() {}
  NeuralNetwork(const std::initializer_list<Synapse<Scalar>> &inputs,
                const std::initializer_list<Synapse<Scalar>> &outputs) {
    init(inputs, outputs);
  }
  void init(const std::initializer_list<Synapse<Scalar>> &inputs,
            const std::initializer_list<Synapse<Scalar>> &outputs) {
    clear();
    for (auto &input : inputs) {
      _input_set.insert(input.layer());
      _inputs.push_back(input.layer());
    }
    for (auto &output : outputs) {
      _outputs.push_back(output.layer());
      _findLayers(output);
    }
    std::reverse(_layers.begin(), _layers.end());
    std::reverse(_ops.begin(), _ops.end());
  }
  void clear() {
    _input_set.clear();
    _layer_set.clear();
    _inputs.clear();
    _outputs.clear();
    _layers.clear();
    _ops.clear();
  }
  auto &inputs() const { return _inputs; }
  auto &outputs() const { return _outputs; }
  auto &layers() const { return _layers; }
  std::vector<Tensor<Scalar>>
  predict(const std::vector<Tensor<Scalar>> &inputs,
          const LayerMode &mode = LayerMode()) const {
    if (inputs.size() != _inputs.size()) {
      throw std::runtime_error("wrong input count");
    }
    std::unordered_map<std::shared_ptr<Layer<Scalar>>, Tensor<Scalar>> tensors;
    for (size_t i = 0; i < _inputs.size(); i++) {
      tensors[_inputs.at(i)] = inputs.at(i);
    }
    for (auto &op : _ops) {
      std::vector<Tensor<Scalar>> input_tensors;
      for (auto &input : op.inputs) {
        if (tensors.find(input) == tensors.end()) {
          throw std::runtime_error("input not connected");
        }
        input_tensors.push_back(tensors[input]);
      }
      auto output_tensor = op.layer->evaluate(input_tensors, mode);
      tensors[op.layer] = output_tensor;
    }
    std::vector<Tensor<Scalar>> output_tensors;
    for (auto &output : _outputs) {
      if (tensors.find(output) == tensors.end()) {
        throw std::runtime_error("output not connected");
      }
      output_tensors.push_back(tensors[output]);
    }
    return output_tensors;
  }
  Tensor<Scalar> predict(const Tensor<Scalar> &input,
                         const LayerMode &mode = LayerMode()) const {
    return predict(std::vector<Tensor<Scalar>>({input}), mode).at(0);
  }
  void serialize(
      const std::function<void(NeuralBase *, void *, size_t)> &fnc) override {
    for (auto &layer : _layers) {
      if (layer) {
        layer->serialize(fnc);
      }
    }
  }
  void serializeWeights(std::ostream &stream) {
    std::cerr << "begin serializing weights" << std::endl;
    auto callback = [&](NeuralBase *layer, void *ptr, size_t s) {
      std::cerr << "serializing layer " << typeid(*layer).name() << " " << layer
                << std::endl;
      stream.write((const char *)ptr, s);
    };
    std::cerr << "serializing weights" << std::endl;
    serialize(callback);
  }
  void saveWeights(const std::string &filename) {
    std::cerr << "opening file " << filename << std::endl;
    std::ofstream s(filename);
    std::cerr << "serializing" << std::endl;
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
  Synapse<Scalar> _input, _output;

public:
  void add(const Synapse<Scalar> &synapse) {
    _input = _output = synapse;
    this->init({_input}, {_output});
  }
  template <class LayerType>
  auto add(LayerType layer) ->
      typename std::enable_if<std::is_base_of<Layer<Scalar>, LayerType>::value,
                              void>::type {
    _output = layer(_output);
    this->init({_input}, {_output});
  }
};

} // namespace tractor
