// (c) 2020-2022 Philipp Ruppel

#pragma once

#include "dexenv.h"
#include "dexlearn.h"

#include <tams_hand_synergies/hand_synergies.h>

namespace tractor {

template <class ValueSingle, class ValueBatch>
struct DexEnvGrasp2 : tractor::DexEnv<ValueSingle, ValueBatch> {

  typedef tractor::Var<ValueSingle> ScalarSingle;
  typedef tractor::GeometryFast<ScalarSingle> GeometrySingle;

  typedef tractor::Var<ValueBatch> ScalarBatch;
  typedef tractor::GeometryFast<ScalarBatch> GeometryBatch;

  tams_hand_synergies::HandSynergies hand_synergies;

  bool use_synergies = true;

  DexEnvGrasp2() {

    this->_info.name = "grasp2";

    this->_info.frame_count = 20;

    this->_info.collision_avoidance_distance = 0;
    this->_info.collision_avoidance_weight = 0;

    this->_info.slip_avoidance_distance = 0;
    this->_info.slip_avoidance_weight = 0;

    this->_info.friction_cone_penalty = 1;

    this->_info.contact_distance_penalty = 3;
    this->_info.contact_slip_penalty = 0.3;

    this->_info.joint_limit_penalty = 1;

    this->_info.shape_penalty = 3;

    this->_info.contact_point_regularization = 0;
    this->_info.contact_force_regularization = 0;

    this->_info.collision_penalty = 0.5;

    this->_info.end_effectors = {
        "ffdistal", "mfdistal", "thdistal", "rfdistal", "lfdistal", "floor",
    };
  }

  virtual std::vector<ScalarBatch> makePolicyInputVector(
      const std::shared_ptr<tractor::PhysicsSimulator<GeometryBatch>>
          &simulator,
      const std::vector<std::string> &joint_names, size_t frame,
      size_t frame_count) override {

    size_t frequencies = 4;
    std::vector<ScalarBatch> neural_input(frequencies);

    double t = frame * 1.0 / frame_count;

    for (size_t i = 0; i < frequencies; i++) {
      neural_input[i] =
          ValueBatch((0.5 - cos(t * (i + 1) * M_PI) * 0.5) / (i + 1));
    }

    return neural_input;
  }

  virtual tractor::NeuralNetwork<ValueBatch>
  makePolicyNetwork(const std::vector<std::string> &joint_names,
                    size_t end_effector_count,
                    size_t contact_dimensions) override {

    auto hand_synergies = this->hand_synergies;

    tractor::SequentialNeuralNetwork<ValueBatch> policy_net;

    std::unordered_set<std::string> hand_joints;
    for (auto &joint_name : hand_synergies.joints()) {
      hand_joints.insert(joint_name);
    }

    std::vector<std::string> arm_joints;
    for (auto &joint_name : joint_names) {
      if (hand_joints.find(joint_name) == hand_joints.end()) {
        arm_joints.push_back(joint_name);
      }
    }

    size_t output_dimensions;
    if (use_synergies) {
      output_dimensions = arm_joints.size() + hand_synergies.components() +
                          end_effector_count * contact_dimensions;
    } else {
      output_dimensions =
          joint_names.size() + end_effector_count * contact_dimensions;
    }

    // policy_net.add(tractor::GaussianNoiseLayer<ScalarBatch>(0.001));

    policy_net.add(std::make_shared<tractor::DenseLayer<ValueBatch>>(
        output_dimensions, tractor::ActivationType::Linear, 0, 0, 0, 0.001,
        false));

    // policy_net.add(tractor::GaussianNoiseLayer<ScalarBatch>(0.001));

    if (use_synergies) {
      policy_net.add(std::make_shared<tractor::LambdaLayer<ValueBatch>>(
          [hand_synergies, hand_joints, arm_joints, end_effector_count,
           contact_dimensions, joint_names](const Tensor<ValueBatch> &inputt) {
            std::vector<ScalarBatch> input;
            unpack(inputt, input);

            std::vector<ScalarBatch> ret;
            ret.resize(joint_names.size() +
                       end_effector_count * contact_dimensions);

            std::map<std::string, ScalarBatch> joint_map;

            size_t iin = 0;
            for (size_t i = 0; i < arm_joints.size(); i++) {
              joint_map[arm_joints[i]] = input[iin++];
            }
            {
              for (size_t i = 0; i < hand_synergies.joints().size(); i++) {
                joint_map[hand_synergies.joints()[i]] =
                    ValueBatch((double)hand_synergies.center()(i));
              }
            }
            for (size_t j = 0; j < hand_synergies.components(); j++) {
              ScalarBatch f = input[iin++];
              f = f * ValueBatch(2);
              for (size_t i = 0; i < hand_synergies.joints().size(); i++) {
                joint_map[hand_synergies.joints()[i]] +=
                    f * ValueBatch((double)hand_synergies.matrix()(j, i));
              }
            }

            for (size_t i = 0; i < joint_names.size(); i++) {
              ret[i] = joint_map[joint_names[i]];
            }

            for (size_t i = 0; i < end_effector_count * contact_dimensions;
                 i++) {
              ret[i + joint_names.size()] = input[iin++];
            }

            if (iin != input.size()) {
              throw std::runtime_error("failed to unpack synergy tensor");
            }

            return pack_tensor(ret);
          }));
    }

    return policy_net;
  }

  virtual void
  init(tractor::DexLearn<ValueSingle, ValueBatch> &dexlearn) override {
    auto &simulator = *dexlearn.simulator();

    // {
    //   double s = 0.02;
    //   ScalarBatch px = add_random_uniform(this->makeZero(), s * -0.5, s *
    //   0.5); ScalarBatch py = add_random_uniform(this->makeZero(), s * -0.5, s
    //   * 0.5); ScalarBatch pz = ValueBatch(0); simulator.moveBody("object",
    //   GeometryBatch::pack(px, py, pz));
    // }
    //
    // {
    //   auto rot = GeometryBatch::angleAxisOrientation(
    //       add_random_uniform(this->makeZero(), 0, M_PI * 2),
    //       GeometryBatch::importVector3(Eigen::Vector3d(0, 0, 1)));
    //   simulator.rotateBody("object", rot);
    // }
  }

  virtual void
  goals(tractor::DexLearn<ValueSingle, ValueBatch> &dexlearn) override {

    dexlearn.addGoal(std::allocate_shared<tractor::MoveGoal<GeometryBatch>>(
        tractor::AlignedStdAlloc<tractor::MoveGoal<GeometryBatch>>(), "object",
        dexlearn.frameCount() - 1,
        GeometryBatch::pack(ValueBatch(0.0), ValueBatch(0.0), ValueBatch(0.1)),
        ValueBatch(1)));

    dexlearn.addGoal(
        std::allocate_shared<tractor::RelativeOrientationGoal<GeometryBatch>>(
            tractor::AlignedStdAlloc<
                tractor::RelativeOrientationGoal<GeometryBatch>>(),
            "object",
            GeometryBatch::pack(ValueBatch(0.0), ValueBatch(0.0),
                                ValueBatch(0.0)),
            ValueBatch(2)));
  }

  virtual void
  controlRobot(tractor::DexLearn<ValueSingle, ValueBatch> &dexlearn,
               const std::vector<ScalarBatch> &policy_output) override {

    auto &_robot_model = dexlearn.robotModel();
    auto &_group_robot = dexlearn.robotJointGroup();
    auto &joint_names = dexlearn.jointNames();

    auto positions = policy_output;

    std::vector<std::pair<std::string, std::string>> couplings;
    couplings.emplace_back("FFJ1", "FFJ2");
    couplings.emplace_back("RFJ1", "RFJ2");
    couplings.emplace_back("LFJ1", "LFJ2");
    couplings.emplace_back("MFJ1", "MFJ2");
    auto *group = _robot_model->getJointModelGroup(_group_robot);
    for (auto &pair : couplings) {
      int i = group->getVariableGroupIndex(pair.first);
      int j = group->getVariableGroupIndex(pair.second);
      positions[i] = positions[j];
    }

    for (size_t i = 0; i < joint_names.size(); i++) {
      dexlearn.simulator()->setJointPosition(joint_names[i], positions[i]);
    }
  }
};

} // namespace tractor
