// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/tensor/tensor.h>

namespace tractor {

template <class Scalar> class Layer;

template <class Scalar> class Synapse {
  AlignedStdVector<Synapse<Scalar>> _inputs;
  std::shared_ptr<Layer<Scalar>> _layer;

public:
  Synapse() {}
  template <class LayerType, class... InputTypes>
  Synapse(int constructor_tag, const LayerType &layer,
          const InputTypes &...inputs)
      : _layer(std::make_shared<LayerType>(layer)), _inputs({inputs...}) {}
  auto &layer() const { return _layer; }
  auto &inputs() const { return _inputs; }
};

} // namespace tractor
