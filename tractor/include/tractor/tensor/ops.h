// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/core/operator.h>

namespace tractor {

template <class T> inline void batchBackprop(T &da, const T &dx) { da = dx; }
template <class T, size_t S>
inline void batchBackprop(T &da, const Batch<T, S> &dx) {
  T rs = T(0);
  for (size_t i = 0; i < S; i++) {
    rs += dx[i];
  }
  da = rs;
}
TRACTOR_OP(batch, (const S &a, T &x), { x = T(a); })
TRACTOR_D(prepare, batch, (const S &a, const T &x), {})
TRACTOR_D(forward, batch, (const S &da, T &dx), { dx = T(da); })
TRACTOR_D(reverse, batch, (S & da, const T &dx), { batchBackprop(da, dx); })

} // namespace tractor
