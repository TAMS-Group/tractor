// 2020-2024 Philipp Ruppel

#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include <tractor/core/log.h>

namespace tractor {

class Buffer {
  std::vector<uint8_t> _data;

 public:
  template <class Begin, class End>
  void assign(const Begin &begin, const End &end) {
    _data.assign(begin, end);
  }

  void append(const Buffer &b) {
    size_t p = _data.size();
    _data.resize(p + b.size());
    std::memcpy(_data.data() + p, b.data(), b.size());
  }

  void clear() { _data.clear(); }

  inline const void *data() const { return _data.data(); }
  inline void *data() { return _data.data(); }

  inline size_t size() const { return _data.size(); }
  void resize(size_t s) { _data.resize(s); }

  void zero(size_t s) {
    _data.clear();
    _data.resize(s, 0);
  }
  void zero() { std::memset(_data.data(), 0, _data.size()); }

  template <class T, class PType>
  inline T &at(const PType &port) {
    if (port.type() != typeid(typename std::decay<T>::type)) {
      throw std::runtime_error("type mismatch");
    }
    if (port.offset() + port.size() < _data.size()) {
      _data.resize(port.offset() + port.size());
    }
    return *(T *)(void *)(_data.data() + port.offset());
  }

  template <class T, class PType>
  inline const T &at(const PType &port) const {
    if (port.type() != typeid(typename std::decay<T>::type)) {
      throw std::runtime_error("type mismatch");
    }
    return *(const T *)(const void *)(_data.data() + port.offset());
  }

  template <class PContainer>
  void gather(const PContainer &container) {
    size_t size = 0;
    for (const auto &port : container) {
      size = std::max(size, port.offset() + port.size());
    }
    // _data.resize(std::max(_data.size(), size));
    _data.resize(size);
    for (const auto &port : container) {
      if (port.binding()) {
        std::memcpy(_data.data() + port.offset(), (const void *)port.binding(),
                    port.size());
      }
    }
  }

  template <class PContainer>
  void scatter(const PContainer &container) const {
    for (const auto &port : container) {
      if (port.binding()) {
        // TRACTOR_DEBUG("scatter " << _data.size() << " " << port.offset()
        //                                 << " " << (void *)port.binding());
        std::memcpy((void *)port.binding(), _data.data() + port.offset(),
                    port.size());
      }
    }
  }

  template <class Vector>
  void fromVector(const Vector &vector) {
    typedef typename std::decay<decltype(vector[0])>::type Scalar;
    _data.resize(vector.size() * sizeof(vector[0]));
    // Scalar *data = (Scalar *)_data.data();
    // for (size_t i = 0; i < vector.size(); i++) {
    //   data[i] = vector[i];
    // }
    std::memcpy(_data.data(), vector.eval().data(), _data.size());
  }

  template <class Vector>
  void toVector(Vector &vector) const {
    typedef typename std::decay<decltype(vector[0])>::type Scalar;
    vector.resize(_data.size() / sizeof(vector[0]));
    std::memcpy(vector.data(), _data.data(), _data.size());
  }
};

}  // namespace tractor
