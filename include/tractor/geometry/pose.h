// 2020-2024 Philipp Ruppel

#pragma once

#include <tractor/core/batch.h>
#include <tractor/geometry/quaternion.h>
#include <tractor/geometry/vector3.h>

namespace tractor {

template <class Scalar>
class Pose {
 private:
  Vector3<Scalar> _translation;
  Quaternion<Scalar> _orientation;

 public:
  Pose() {}
  inline Pose(const Vector3<Scalar> &translation,
              const Quaternion<Scalar> &orientation)
      : _translation(translation), _orientation(orientation) {}
  template <class T, class R = decltype(Scalar(std::declval<T>()))>
  explicit Pose(const Pose<T> &other) {
    translation() = Vector3<Scalar>(other.translation());
    orientation() = Quaternion<Scalar>(other.orientation());
  }
  inline auto &translation() const { return _translation; }
  inline auto &translation() { return _translation; }
  inline auto &position() const { return _translation; }
  inline auto &position() { return _translation; }
  inline auto &orientation() const { return _orientation; }
  inline auto &orientation() { return _orientation; }
  inline static auto Identity() {
    Pose ret;
    ret.translation() = Vector3<Scalar>::Zero();
    ret.orientation() = Quaternion<Scalar>::Identity();
    return ret;
  }
  inline void setZero() {
    _translation.setZero();
    _orientation.setZero();
  }
  inline Pose<Scalar> inverse() const {
    Pose<Scalar> ret;
    ret.orientation() = _orientation.inverse();
    ret.translation() = ret.orientation() * -_translation;
    return ret;
  }
};

template <class T>
std::ostream &operator<<(std::ostream &s, const Pose<T> &p) {
  return s << "[" << p.translation() << "," << p.orientation() << "]";
}

template <class T>
void variable(Pose<Var<T>> &v) {
  variable(v.translation());
  variable(v.orientation());
}
template <class T>
void parameter(Pose<Var<T>> &v) {
  parameter(v.translation());
  parameter(v.orientation());
}
template <class T>
void output(Pose<Var<T>> &v) {
  output(v.translation());
  output(v.orientation());
}
template <class T>
void goal(const Pose<Var<T>> &v) {
  goal(v.translation());
  goal(v.orientation());
}

template <class T, size_t S>
inline Pose<T> indexBatch(const Pose<Batch<T, S>> &pose, size_t i) {
  return Pose<T>(indexBatch(pose.translation(), i),
                 indexBatch(pose.orientation(), i));
}

template <class T>
Pose<T> operator*(const Pose<T> &a, const Pose<T> &b) {
  Pose<T> x;
  x.translation() = a.translation() + a.orientation() * b.translation();
  x.orientation() = a.orientation() * b.orientation();
  return x;
}

template <class T>
Vector3<T> operator*(const Pose<T> &a, const Vector3<T> &b) {
  return a.translation() + a.orientation() * b;
}

template <class T>
Pose<T> angle_axis_pose(const T &angle, const Vector3<T> &axis) {
  return Pose<T>(Vector3<T>::Zero(), angle_axis_quat(angle, axis));
}

template <class Pose, class Angle, class Vec3>
Pose pose_angle_axis_pose(const Pose &parent, const Angle &angle,
                          const Vec3 &axis) {
  return parent * angle_axis_pose(angle, axis);
}

template <class T>
Vector3<T> pose_translation(const Pose<T> &pose) {
  return pose.translation();
}

template <class T>
Quaternion<T> pose_orientation(const Pose<T> &pose) {
  return pose.orientation();
}

template <class T>
Pose<T> translation_pose(const Vector3<T> &translation) {
  Pose<T> pose;
  pose.translation() = translation;
  pose.orientation().setIdentity();
  return pose;
}

template <class T>
Pose<T> orientation_pose(const Quaternion<T> &orientation) {
  Pose<T> pose;
  pose.orientation() = orientation;
  pose.translation().setZero();
  return pose;
}

template <class T>
Pose<T> pose_translate(const Pose<T> &parent, const Vector3<T> &translation) {
  Pose<T> pose;
  pose.orientation() = parent.orientation();
  pose.translation() =
      parent.translation() + parent.orientation() * translation;
  return pose;
}

typedef Pose<double> Pose3d;
typedef Pose<float> Pose3f;

}  // namespace tractor
