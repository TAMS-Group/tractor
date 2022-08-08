// (c) 2020-2022 Philipp Ruppel

#pragma once

#include <tractor/geometry/fast.h>

namespace tractor {

bool interact(const std::string &frame, const std::string &name,
              Vector3<double> &position, double size);

template <class T>
bool interact(const std::string &frame, const std::string &name,
              Var<Vector3<T>> &position, double size) {
  Vector3<double> pos(value(position).x(), value(position).y(),
                      value(position).z());
  if (interact(frame, name, pos, size)) {
    value(position).x() = pos.x();
    value(position).y() = pos.y();
    value(position).z() = pos.z();
  }
}

} // namespace tractor
