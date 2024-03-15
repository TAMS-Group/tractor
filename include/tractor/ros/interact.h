// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/geometry/fast.h>

namespace tractor {

template <class T>
bool interact(const std::string &frame, const std::string &name, Pose<T> &pose,
              const T &size);

template <class T>
bool interact(const std::string &frame, const std::string &name,
              Var<Pose<T>> &pose, const T &size) {
  Pose<T> p = value(pose);
  if (interact(frame, name, p, size)) {
    pose = Var<Pose<T>>(p);
    return true;
  } else {
    return false;
  }
}

template <class T>
bool interact(const std::string &frame, const std::string &name,
              Vector3<T> &position, const T &size);

template <class T>
bool interact(const std::string &frame, const std::string &name,
              Var<Vector3<T>> &position, const T &size) {
  Vector3<T> p = value(position);
  if (interact(frame, name, p, size)) {
    position = Var<Vector3<T>>(p);
    return true;
  } else {
    return false;
  }
}

}  // namespace tractor
