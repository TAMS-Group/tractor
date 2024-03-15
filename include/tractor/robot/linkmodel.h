// 2020-2024 Philipp Ruppel

#pragma once

#include <memory>
#include <string>
#include <tractor/dynamics/inertia.h>
#include <vector>

namespace tractor {

template <class Geometry>
class JointModelBase;

template <class Geometry>
class LinkModel {
  std::vector<std::shared_ptr<const JointModelBase<Geometry>>> _child_joints;
  std::shared_ptr<const JointModelBase<Geometry>> _parent_joint;
  std::string _name;
  Inertia<Geometry> _inertia;

 public:
  LinkModel(const std::shared_ptr<const JointModelBase<Geometry>> &parent_joint,
            const std::string &name, const Inertia<Geometry> &inertia) {
    _parent_joint = parent_joint;
    _name = name;
    _inertia = inertia;
  }
  void addChildJoint(
      const std::shared_ptr<const JointModelBase<Geometry>> &child_joint) {
    _child_joints.push_back(child_joint);
  }
  auto &name() const { return _name; }
  auto &inertia() const { return _inertia; }
  auto &parentJoint() const { return _parent_joint; }
  auto &childJoints() const { return _child_joints; }
};

}  // namespace tractor
