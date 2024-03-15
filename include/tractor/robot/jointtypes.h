// 2020-2024 Philipp Ruppel

#pragma once

#include <limits>
#include <memory>

#include <tractor/core/constraints.h>
#include <tractor/dynamics/inertia.h>
#include <tractor/geometry/ops.h>

namespace tractor {

template <class Geometry>
class LinkModel;

template <class Geometry>
struct JointVariableOptions {
  double trust_region = -1;
};

template <class Geometry>
class alignas(32) JointModelBase {
  typename Geometry::Pose _origin;
  std::string _name;
  Inertia<Geometry> _inertia;
  std::shared_ptr<const LinkModel<Geometry>> _parent_link, _child_link;

 public:
  void init(const std::shared_ptr<const LinkModel<Geometry>> &parent_link,
            const std::string &name, const typename Geometry::Pose &origin,
            const std::shared_ptr<const LinkModel<Geometry>> &child_link) {
    _parent_link = parent_link;
    _name = name;
    _origin = origin;
    _child_link = child_link;
  }

  auto &name() const { return _name; }
  const typename Geometry::Pose &origin() const { return _origin; }
  virtual ~JointModelBase() {}
  auto &parentLink() const { return _parent_link; }
  auto &childLink() const { return _child_link; }
};
template <class Geometry>
class alignas(32) JointStateBase {
 public:
  virtual ~JointStateBase() {}
  virtual void makeVariables(const JointModelBase<Geometry> &model,
                             const JointVariableOptions<Geometry> &options) {}
  virtual void makeParameters(const JointModelBase<Geometry> &model) {}
  virtual typename Geometry::Pose compute(
      const JointModelBase<Geometry> &model,
      const typename Geometry::Pose &pose) const {
    throw std::runtime_error("nyi");
  }
  virtual void serializePositions(typename Geometry::Scalar *positions) const {}
  virtual void deserializePositions(
      const typename Geometry::Scalar *positions) {}
};

template <class Geometry>
class alignas(32) FixedJointModel : public JointModelBase<Geometry> {
 public:
};
template <class Geometry>
class alignas(32) FixedJointState : public JointStateBase<Geometry> {
 public:
  virtual typename Geometry::Pose compute(
      const JointModelBase<Geometry> &model,
      const typename Geometry::Pose &pose) const override {
    return pose;
  }
};

template <class Geometry>
class alignas(32) JointLimits {
  bool _limited = false;
  typename Geometry::Value _lower = typename Geometry::Value(0);
  typename Geometry::Value _upper = typename Geometry::Value(0);

 public:
  JointLimits() {}
  JointLimits(const typename Geometry::Value &lower,
              const typename Geometry::Value &upper)
      : _limited(true), _lower(lower), _upper(upper) {}
  auto &lower() const { return _lower; }
  auto &upper() const { return _upper; }
  operator bool() const { return _limited; }
};
template <class Geometry>
class alignas(32) ScalarJointModelBase : public JointModelBase<Geometry> {
  JointLimits<Geometry> _limits;

 public:
  auto &limits() const { return _limits; }
  auto &limits() { return _limits; }
};
template <class Geometry>
class alignas(32) ScalarJointStateBase : public JointStateBase<Geometry> {
  typename Geometry::Scalar _position = typename Geometry::Value(0);

 public:
  auto &position() const { return _position; }
  auto &position() { return _position; }
  virtual void makeVariables(
      const JointModelBase<Geometry> &model,
      const JointVariableOptions<Geometry> &options) override {
    bool has_trust_region = (options.trust_region > 0);
    auto &m = dynamic_cast<const ScalarJointModelBase<Geometry> &>(model);
    variable(_position);
    if (m.limits() && has_trust_region) {
      goal(range_trust_region_constraint(
          _position, m.limits().lower(), m.limits().upper(),
          typename Geometry::Value(options.trust_region)));
    }
    if (!m.limits() && has_trust_region) {
      goal(trust_region_constraint(
          _position, typename Geometry::Value(options.trust_region)));
    }
    if (m.limits() && !has_trust_region) {
      goal(range_constraint(_position, m.limits().lower(), m.limits().upper()));
    }
  }
  virtual void makeParameters(const JointModelBase<Geometry> &model) override {
    auto &m = dynamic_cast<const ScalarJointModelBase<Geometry> &>(model);
    parameter(_position);
  }
  virtual void serializePositions(
      typename Geometry::Scalar *positions) const override {
    positions[0] = _position;
  }
  virtual void deserializePositions(
      const typename Geometry::Scalar *positions) override {
    _position = positions[0];
  }
};

template <class Geometry>
class alignas(32) RevoluteJointModel : public ScalarJointModelBase<Geometry> {
  typename Geometry::Vector3 _axis;

 public:
  const typename Geometry::Vector3 &axis() const { return _axis; }
  typename Geometry::Vector3 &axis() { return _axis; }
};
template <class Geometry>
class alignas(32) RevoluteJointState : public ScalarJointStateBase<Geometry> {
 public:
  virtual typename Geometry::Pose compute(
      const JointModelBase<Geometry> &model,
      const typename Geometry::Pose &parent) const override {
    auto &m = dynamic_cast<const RevoluteJointModel<Geometry> &>(model);
    return Geometry::angleAxisPose(parent, this->position(), m.axis());
  }
};

template <class Geometry>
class alignas(32) PrismaticJointModel : public ScalarJointModelBase<Geometry> {
  typename Geometry::Vector3 _axis;

 public:
  auto &axis() const { return _axis; }
  auto &axis() { return _axis; }
};
template <class Geometry>
class alignas(32) PrismaticJointState : public ScalarJointStateBase<Geometry> {
 public:
  virtual typename Geometry::Pose compute(
      const JointModelBase<Geometry> &model,
      const typename Geometry::Pose &parent) const override {
    auto &m = dynamic_cast<const PrismaticJointModel<Geometry> &>(model);
    return Geometry::translationPose(parent, m.axis() * this->position());
  }
};

template <class Geometry>
class alignas(32) PlanarJointModel : public JointModelBase<Geometry> {
 public:
};
template <class Geometry>
class alignas(32) PlanarJointState : public JointStateBase<Geometry> {
  typename Geometry::Scalar _x, _y, _angle;

 public:
  auto &x() const { return _x; }
  auto &y() const { return _y; }
  auto &angle() const { return _angle; }
  auto &x() { return _x; }
  auto &y() { return _y; }
  auto &angle() { return _angle; }
  virtual typename Geometry::Pose compute(
      const JointModelBase<Geometry> &model,
      const typename Geometry::Pose &parent) const override {
    return Geometry::angleAxisPose(
        Geometry::translationPose(parent, _x, _y, typename Geometry::Value(0)),
        _angle, Geometry::UnitZ());
  }
  virtual void makeVariables(
      const JointModelBase<Geometry> &model,
      const JointVariableOptions<Geometry> &options) override {
    variable(_x);
    variable(_y);
    variable(_angle);
    bool has_trust_region = (options.trust_region > 0);
    if (has_trust_region) {
      goal(tractor::trust_region_constraint(
          _x, typename Geometry::Value(options.trust_region)));
      goal(tractor::trust_region_constraint(
          _y, typename Geometry::Value(options.trust_region)));
      goal(tractor::trust_region_constraint(
          _angle, typename Geometry::Value(options.trust_region)));
    }
  }
  virtual void makeParameters(const JointModelBase<Geometry> &model) override {
    parameter(_x);
    parameter(_y);
    parameter(_angle);
  }
  virtual void serializePositions(
      typename Geometry::Scalar *positions) const override {
    positions[0] = _x;
    positions[1] = _y;
    positions[2] = _angle;
  }
  virtual void deserializePositions(
      const typename Geometry::Scalar *positions) override {
    _x = positions[0];
    _y = positions[1];
    _angle = positions[2];
  }
};

template <class Geometry>
class alignas(32) FloatingJointModel : public JointModelBase<Geometry> {
 public:
};
template <class Geometry>
class alignas(32) FloatingJointState : public JointStateBase<Geometry> {
  typename Geometry::Pose _pose;

 public:
  auto &pose() const { return _pose; }
  auto &pose() { return _pose; }
  virtual typename Geometry::Pose compute(
      const JointModelBase<Geometry> &model,
      const typename Geometry::Pose &parent) const {
    return parent * _pose;
  }
  virtual void makeVariables(
      const JointModelBase<Geometry> &model,
      const JointVariableOptions<Geometry> &options) override {
    variable(_pose);
    bool has_trust_region = (options.trust_region > 0);
    if (has_trust_region) {
      throw std::runtime_error("nyi");
    }
  }
  virtual void makeParameters(const JointModelBase<Geometry> &model) override {
    parameter(_pose);
  }
  virtual void serializePositions(
      typename Geometry::Scalar *positions) const override {
    Geometry::unpack(_pose, positions[0], positions[1], positions[2],
                     positions[3], positions[4], positions[5], positions[6]);
  }
  virtual void deserializePositions(
      const typename Geometry::Scalar *positions) override {
    _pose =
        Geometry::pack(positions[0], positions[1], positions[2], positions[3],
                       positions[4], positions[5], positions[6]);
  }
};

template <class Base>
class JointVariant {
  std::shared_ptr<Base> _instance;
  std::shared_ptr<Base> (*_clone)(const std::shared_ptr<Base> &instance) =
      nullptr;

 public:
  JointVariant() {}

  const std::shared_ptr<Base> &pointer() { return _instance; }

  template <class T>
  explicit JointVariant(const T &value) {
    _instance = std::allocate_shared<T, AlignedStdAlloc<T>, const T &>(
        AlignedStdAlloc<T>(), value);
    _clone = [](const std::shared_ptr<Base> &instance) {
      std::shared_ptr<Base> ret =
          std::allocate_shared<T, AlignedStdAlloc<T>, const T &>(
              AlignedStdAlloc<T>(), *std::dynamic_pointer_cast<T>(instance));
      return ret;
    };
  }

  void assign(const JointVariant &other) {
    if (other._instance) {
      _instance = other._clone(other._instance);
      _clone = other._clone;
    } else {
      _instance = nullptr;
      _clone = nullptr;
    }
  }

  JointVariant(const JointVariant &other) { assign(other); }
  JointVariant &operator=(const JointVariant &other) {
    assign(other);
    return *this;
  }

  const Base *operator->() const { return _instance.get(); }
  const Base &operator*() const { return *_instance; }

  Base *operator->() { return _instance.get(); }
  Base &operator*() { return *_instance; }

  const Base *get() const { return _instance.get(); }
  Base *get() { return _instance.get(); }
};

}  // namespace tractor
