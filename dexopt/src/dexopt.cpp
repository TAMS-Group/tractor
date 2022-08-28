// (c) 2020-2022 Philipp Ruppel

//#include <tractor/tractor.h>

#include <tractor/core/eigen.h>

#include "dexlearn.h"

#include "dexenv_grasp.h"
#include "dexenv_grasp_2.h"
#include "dexenv_grasp_3.h"
#include "dexenv_push.h"
#include "dexenv_turn.h"

#include "common.h"
#include "goals.h"
#include "physics5.h"

#include <moveit/move_group_interface/move_group_interface.h>

#include <tf/transform_listener.h>

#include <tractor/core/batch.h>
#include <tractor/core/var.h>
#include <tractor/engines/parallel.h>
#include <tractor/engines/simple.h>
#include <tractor/robot/robotstate.h>
#include <tractor/solvers/gd.h>
#include <tractor/solvers/sq.h>

static const size_t inner_batch_size = 4;
static size_t outer_batch_size = 4;

typedef double ValueSingle;
typedef tractor::Batch<ValueSingle, inner_batch_size> ValueBatch;

typedef tractor::Var<ValueSingle> ScalarSingle;
typedef tractor::GeometryFast<ScalarSingle> GeometrySingle;

typedef tractor::Var<ValueBatch> ScalarBatch;
typedef tractor::GeometryFast<ScalarBatch> GeometryBatch;

std::shared_ptr<tractor::Solver>
makeSolver(const std::shared_ptr<tractor::Engine> &engine,
           const std::string &solvername) {

  if (solvername == "sq") {
    auto s = std::make_shared<tractor::LeastSquaresSolver<ValueSingle>>(engine);
    s->_regularization = 1;
    s->_max_linear_iterations = 100;
    s->_step_scaling = 0.8;
    s->setTimeout(1, false);
    s->setTolerance(0);
    return s;
  }

  if (solvername == "sq2") {
    auto s = std::make_shared<tractor::LeastSquaresSolver<ValueSingle>>(engine);
    s->_regularization = 0.1;
    s->_max_linear_iterations = 10;
    s->_step_scaling = 0.2;
    s->setTimeout(1, false);
    s->setTolerance(0);
    return s;
  }

  throw std::runtime_error("unknown solver " + solvername);
}

int main(int argc, char **argv) {

  tractor::ProfilerThread::start();

  std::vector<std::shared_ptr<tractor::DexEnv<ValueSingle, ValueBatch>>> envs =
      {
          std::make_shared<tractor::DexEnvGrasp<ValueSingle, ValueBatch>>(),
          std::make_shared<tractor::DexEnvTurn<ValueSingle, ValueBatch>>(),
          std::make_shared<tractor::DexEnvPush<ValueSingle, ValueBatch>>(),
          std::make_shared<tractor::DexEnvGrasp2<ValueSingle, ValueBatch>>(),
          std::make_shared<tractor::DexEnvGrasp3<ValueSingle, ValueBatch>>(),
      };

  if (argc < 4) {
    std::cerr << "USAGE: dexopt <env> <command> <solver>" << std::endl;
    return -1;
  }
  std::string envname = argv[1];
  std::string command = argv[2];
  std::string solvername = argv[3];

  // ros::WallDuration training_time(60 * 5);
  ros::WallDuration training_time(60 * 2);
  //  ros::WallDuration training_time(60 * 10);

  std::string filename = "weights-" + envname + "-" + solvername + ".dat";

  std::shared_ptr<tractor::DexEnv<ValueSingle, ValueBatch>> env;
  for (auto &e : envs) {
    if (e->info().name == envname) {
      env = e;
    }
  }
  if (env == nullptr) {
    throw std::runtime_error("unknown env " + envname);
  }

  std::string robot_description = "dexopt_robot_description";

  // ros::init(argc, argv, "dexopt", ros::init_options::NoSigintHandler);
  ros::init(argc, argv, "dexopt");
  ros::NodeHandle node_handle;

  std::string group_robot = "robot";
  std::string group_all = "all";

  RobotTrajectoryPublisher robot_trajectory_publisher;

  ros::AsyncSpinner spinner(4);
  spinner.start();

  robot_model_loader::RobotModelLoader robot_model_loader(robot_description,
                                                          false);
  auto robot_model = robot_model_loader.getModel();

  // auto engine = std::make_shared<tractor::SimpleEngine>();
  auto engine = std::make_shared<tractor::ParallelEngine>();

  ros::Publisher visualization_publisher =
      node_handle.advertise<visualization_msgs::MarkerArray>(
          "/tractor/visualization", 10, true);

  planning_scene::PlanningScene planning_scene(robot_model);

  auto acm2 = planning_scene.getAllowedCollisionMatrix();
  {
    for (auto &a : robot_model->getLinkModelNames()) {
      for (auto &b : robot_model->getLinkModelNames()) {
        acm2.setEntry(a, b, true);
      }
    }

    acm2.setEntry("object", "ffdistal", false);
    acm2.setEntry("object", "mfdistal", false);
    acm2.setEntry("object", "rfdistal", false);
    acm2.setEntry("object", "lfdistal", false);
    acm2.setEntry("object", "thdistal", false);

    acm2.setEntry("object", "ffmiddle", false);
    acm2.setEntry("object", "mfmiddle", false);
    acm2.setEntry("object", "rfmiddle", false);
    acm2.setEntry("object", "lfmiddle", false);
    acm2.setEntry("object", "thmiddle", false);

    acm2.setEntry("object", "ffproximal", false);
    acm2.setEntry("object", "mfproximal", false);
    acm2.setEntry("object", "rfproximal", false);
    acm2.setEntry("object", "lfproximal", false);
    acm2.setEntry("object", "thproximal", false);

    acm2.setEntry("floor", "ffdistal", false);
    acm2.setEntry("floor", "mfdistal", false);
    acm2.setEntry("floor", "rfdistal", false);
    acm2.setEntry("floor", "lfdistal", false);
    acm2.setEntry("floor", "thdistal", false);

    acm2.setEntry("floor", "forearm", false);
    acm2.setEntry("floor", "arm_wrist_3_link", false);
    acm2.setEntry("floor", "arm_wrist_2_link", false);
    acm2.setEntry("floor", "arm_wrist_1_link", false);

    acm2.setEntry("object", "palm", false);

    acm2.setEntry("object", "floor", false);
  }

  auto joint_names =
      robot_model->getJointModelGroup(group_robot)->getVariableNames();

  if (command != "train") {
    outer_batch_size = 1;
  }

  tractor::DexLearn<ValueSingle, ValueBatch> dexlearn(
      engine, robot_model, acm2, group_robot, env, outer_batch_size);

  auto build = [&]() {
    return dexlearn.build([&]() { env->goals(dexlearn); });
  };

  auto saveWeights = [&]() {
    if (!filename.empty()) {
      TRACTOR_INFO("saving weights to " << filename);
      dexlearn.policyNetwork().saveWeights(filename);
      TRACTOR_INFO("weights saved");
    }
  };

  if (command == "train") {
    TRACTOR_INFO("building solver");
    std::shared_ptr<tractor::Solver> solver = makeSolver(engine, solvername);
    solver->compile(build());
    TRACTOR_INFO("training");
    ros::WallTime start_time = ros::WallTime::now();
    std::ofstream logfile("log-" + envname + "-" + solvername + ".txt");
    // Destructor weight_saver([&]() {
    //   TRACTOR_INFO("saving weights");
    //   saveWeights();
    //   TRACTOR_INFO("weights saved");
    // });
    while (true) {
      if (!ros::ok()) {
        TRACTOR_WARN("aborted");
        break;
      }
      auto elapsed_time = ros::WallTime::now() - start_time;
      std::cout << "training time " << elapsed_time << " / " << training_time
                << " "
                << std::round(elapsed_time.toSec() * 100.0 /
                              training_time.toSec())
                << "% finished" << std::endl;
      if (elapsed_time > training_time) {
        std::cerr << "training finished" << std::endl;
        break;
      }
      solver->parameterize();
      solver->gather();
      solver->solve();
      solver->scatter();
      dexlearn.test(true);
      std::cout << "loss " << solver->loss() << std::endl;
      logfile << elapsed_time.toSec() << " " << solver->loss() << std::endl;
      visualization_publisher.publish(dexlearn.visualization());
      robot_trajectory_publisher.publish(robot_model, group_all,
                                         dexlearn.trajectory());
    }
    logfile.close();
    saveWeights();
  }

  if (command == "partrain") {
  }

  if (command == "test") {

    build();

    if (!filename.empty()) {
      TRACTOR_INFO("loading weights from " << filename);
      dexlearn.policyNetwork().loadWeights(filename);
    }

    TRACTOR_INFO("testing");
    while (ros::ok()) {
      dexlearn.test();
      visualization_publisher.publish(dexlearn.visualization());
      robot_trajectory_publisher.publish(robot_model, group_all,
                                         dexlearn.trajectory());
    }
  }

  if (command == "interactive") {

    moveit::core::RobotState robot_state(robot_model);
    robot_state.setToDefaultValues();
    robot_state.update();

    interactive_markers::InteractiveMarkerServer marker_server(
        "interactive_markers");

    InteractivePoseMarker object_marker(marker_server, "object", robot_state,
                                        0.1);

    dexlearn.setInitializer(
        [&](tractor::PhysicsSimulator<GeometryBatch> &simulator) {
          auto pose = GeometryBatch::import(
              /*object_marker.initialPose().inverse() **/ object_marker.pose());

          simulator.setBodyPose("object", pose);
        });

    build();

    if (!filename.empty()) {
      TRACTOR_INFO("loading weights from " << filename);
      dexlearn.policyNetwork().loadWeights(filename);
    }

    TRACTOR_INFO("testing");
    while (ros::ok()) {

      if (!object_marker.poll()) {
        ros::WallDuration(0.01).sleep();
        continue;
      }

      dexlearn.test();
      visualization_publisher.publish(dexlearn.visualization());
      robot_trajectory_publisher.publish(robot_model, group_all,
                                         dexlearn.trajectory());
    }
  }

  if (command == "step") {

    DisplayRobotStatePublisher display_robot_state_pub(
        "/dexopt/display_robot_state");

    JointStatePublisher joint_state_pub("/test_joint_states");

    TRACTOR_INFO("init policy");
    dexlearn.makeSimulator();
    {
      tractor::RobotState<GeometryBatch> robot_state(
          dexlearn.simulator()->model());
      dexlearn.simulator()->model()->computeFK(robot_state.joints(),
                                               robot_state.links());
      dexlearn.simulator()->init(robot_state);
    }

    TRACTOR_INFO("init neural network");
    tractor::LayerMode layer_mode;
    layer_mode.training = false;
    dexlearn.runPolicyNetwork(layer_mode, 0);

    TRACTOR_INFO("loading weights from " << filename);
    dexlearn.policyNetwork().loadWeights(filename);

    TRACTOR_INFO("clear dexviz");
    dexlearn.dexviz().clear();

    moveit::core::RobotState source_robot_state(robot_model);

    size_t iframe = 0;
    auto step_policy = [&]() {
      dexlearn.dexviz().clear();
      dexlearn.simulator()->step();
      auto policy_output_vector =
          dexlearn.runPolicyNetwork(layer_mode, iframe++);
      TRACTOR_INFO("policy output size " << policy_output_vector.size());
      TRACTOR_INFO("joints " << dexlearn.jointNames().size());
      TRACTOR_INFO("eefs " << dexlearn.endEffectors().size());
      env->controlRobot(dexlearn, policy_output_vector);
      dexlearn.applyContacts(policy_output_vector);
      visualization_publisher.publish(dexlearn.visualization());
      toMoveIt(dexlearn.simulator()->state(), source_robot_state);
      display_robot_state_pub.publish(source_robot_state);
      joint_state_pub.publish(source_robot_state);
    };

    TRACTOR_INFO("init env");
    env->init(dexlearn);

    TRACTOR_INFO("run loop");
    while (ros::ok()) {
      step_policy();
      ros::Duration(0.1).sleep();
    }
  }

  if (command == "run") {

    auto arm_command_pub =
        node_handle.advertise<trajectory_msgs::JointTrajectory>(
            "/arm/scaled_pos_joint_traj_controller/command", 1);

    auto hand_command_pub =
        node_handle.advertise<trajectory_msgs::JointTrajectory>(
            "/hand/lh_trajectory_controller/command", 1);

    DisplayRobotStatePublisher display_robot_state_pub(
        "/dexopt/display_robot_state");

    DisplayRobotStatePublisher target_display_robot_state(
        "/display_robot_state");

    JointStatePublisher joint_state_pub("/test_joint_states");

    tf::TransformBroadcaster tf_br;

    tf::TransformListener tf_listener;
    auto getTransform = [&](const std::string &name) {
      TRACTOR_INFO("get transform " << name);
      while (true) {
        tf::StampedTransform transform;
        try {
          tf_listener.lookupTransform("/world", name, ros::Time(0), transform);
          Eigen::Isometry3d pose;
          tf::transformTFToEigen(transform, pose);
          TRACTOR_INFO("transform found " << name);
          return pose;
        } catch (tf::TransformException ex) {
          ROS_ERROR("%s", ex.what());
          ros::Duration(0.5).sleep();
          continue;
        }
      }
    };

    TRACTOR_INFO("init robot move group");
    moveit::planning_interface::MoveGroupInterface target_move_group("arm");

    TRACTOR_INFO("init policy");
    dexlearn.makeSimulator();
    {
      tractor::RobotState<GeometryBatch> robot_state(
          dexlearn.simulator()->model());
      dexlearn.simulator()->model()->computeFK(robot_state.joints(),
                                               robot_state.links());
      dexlearn.simulator()->init(robot_state);
    }

    TRACTOR_INFO("init neural network");
    tractor::LayerMode layer_mode;
    layer_mode.training = false;
    dexlearn.runPolicyNetwork(layer_mode, 0);

    TRACTOR_INFO("loading weights from " << filename);
    dexlearn.policyNetwork().loadWeights(filename);

    TRACTOR_INFO("init env");
    env->init(dexlearn);

    TRACTOR_INFO("clear dexviz");
    dexlearn.dexviz().clear();

    moveit::core::RobotState source_robot_state(robot_model);

    size_t iframe = 0;
    auto step_policy = [&]() {
      dexlearn.dexviz().clear();
      dexlearn.simulator()->step();
      auto policy_output_vector =
          dexlearn.runPolicyNetwork(layer_mode, iframe++);
      TRACTOR_INFO("policy output size " << policy_output_vector.size());
      TRACTOR_INFO("joints " << dexlearn.jointNames().size());
      TRACTOR_INFO("eefs " << dexlearn.endEffectors().size());
      env->controlRobot(dexlearn, policy_output_vector);
      dexlearn.applyContacts(policy_output_vector);
      visualization_publisher.publish(dexlearn.visualization());
      toMoveIt(dexlearn.simulator()->state(), source_robot_state);
      display_robot_state_pub.publish(source_robot_state);
      joint_state_pub.publish(source_robot_state);
    };

    TRACTOR_INFO("plan to first state");
    step_policy();

    auto source_object_pose =
        source_robot_state.getGlobalLinkTransform("object");
    source_object_pose.linear() = Eigen::Matrix3d::Identity();

    Eigen::Isometry3d target_object_pose = getTransform("object");
    target_object_pose.linear() = Eigen::Matrix3d::Identity();

    TRACTOR_INFO(__LINE__);

    auto mapForearmPose = [&]() {
      Eigen::Isometry3d goal_pose(
          (target_object_pose *
           Eigen::AngleAxisd(M_PI * 1.0, Eigen::Vector3d::UnitZ()) *
           Eigen::Affine3d(Eigen::Scaling(Eigen::Vector3d(1, -1, 1))) *
           source_object_pose.inverse() *
           source_robot_state.getGlobalLinkTransform("forearm") *
           Eigen::Affine3d(Eigen::Scaling(Eigen::Vector3d(-1, 1, 1))))
              .matrix());
      {
        geometry_msgs::TransformStamped transform;
        transform.header.stamp = ros::Time::now();
        transform.header.frame_id = "world";
        transform.child_frame_id = "goal";
        tf::transformEigenToMsg(goal_pose, transform.transform);
        tf_br.sendTransform(transform);
      }
      return goal_pose;
    };

    // {
    //   bool ok = target_move_group.setPoseTarget(mapForearmPose(),
    //   "lh_forearm"); TRACTOR_INFO("set pose target " << (int)ok); if (!ok) {
    //     ROS_ERROR_STREAM("set pose target failed");
    //     return -1;
    //   }
    //   target_display_robot_state.publish();
    // }

    {
      // robot_state::RobotState target_robot_state =
      //     *target_move_group.getCurrentState();

      robot_state::RobotState target_robot_state(
          target_move_group.getRobotModel());
      target_robot_state.setToDefaultValues();
      TRACTOR_ASSERT(target_robot_state.setToDefaultValues(
          target_robot_state.getJointModelGroup("arm"), "ready"));

      while (true) {
        TRACTOR_INFO("solving start state ik");
        bool ok = target_robot_state.setFromIK(
            target_robot_state.getRobotModel()->getJointModelGroup("arm"),
            mapForearmPose(), "lh_forearm");
        if (ok) {
          break;
        }
      }

      target_display_robot_state.publish(target_robot_state);

      if (!target_move_group.setJointValueTarget(target_robot_state)) {
        TRACTOR_FATAL("failed to set joint value target");
        ros::Duration(1).sleep();
        throw std::runtime_error("failed to set joint value target");
      }
    }

    TRACTOR_INFO(__LINE__);

    {
      moveit::planning_interface::MoveGroupInterface::Plan plan;

      {
        auto ok = target_move_group.plan(plan);
        TRACTOR_INFO("plan result " << ok);
        if (!ok) {
          TRACTOR_FATAL("plan failed");
          return -1;
        }
      }

      TRACTOR_INFO("press key to execute");
      getchar();

      {
        auto ok = target_move_group.execute(plan);
        TRACTOR_INFO("execute result " << ok);
        if (!ok) {
          TRACTOR_FATAL("execute failed");
          return -1;
        }
      }
    }

    TRACTOR_INFO(__LINE__);

    const static std::vector<std::string> source_hand_joint_names = {
        "FFJ4", "FFJ3", "FFJ2", "FFJ1", "LFJ5", "LFJ4", "LFJ3", "LFJ2",
        "LFJ1", "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2",
        "RFJ1", "THJ5", "THJ4", "THJ3", "THJ2", "THJ1", "WRJ2", "WRJ1",
    };

    const static std::vector<std::string> target_hand_joint_names = []() {
      std::vector<std::string> ret;
      for (auto &n : source_hand_joint_names) {
        ret.push_back("lh_" + n);
      }
      return ret;
    }();

    const static std::vector<std::string> arm_joint_names = {
        "arm_shoulder_pan_joint", "arm_shoulder_lift_joint",
        "arm_elbow_joint",        "arm_wrist_1_joint",
        "arm_wrist_2_joint",      "arm_wrist_3_joint",
    };

    auto sendCommands = [](const robot_state::RobotState &robot_state,
                           const std::vector<std::string> &joint_names,
                           ros::Publisher &publisher,
                           const ros::Duration &time_from_start) {
      trajectory_msgs::JointTrajectory traj;
      traj.points.emplace_back();
      traj.points.front().time_from_start = time_from_start;
      for (auto &joint_name : joint_names) {
        traj.joint_names.push_back(joint_name);
        traj.points.front().positions.push_back(
            robot_state.getJointPositions(joint_name)[0]);
      }
      publisher.publish(traj);
    };

    TRACTOR_INFO(__LINE__);

    robot_state::RobotState target_robot_state =
        *target_move_group.getCurrentState();

    auto updateTargetRobotState = [&]() {
      {
        bool ok = target_robot_state.setFromIK(
            target_robot_state.getJointModelGroup("arm"), mapForearmPose(),
            "lh_forearm");
        TRACTOR_INFO("ik ok " << (int)ok);
      }

      for (auto &joint_name : source_hand_joint_names) {
        double p = source_robot_state.getJointPositions(joint_name)[0];
        if (joint_name == "THJ3") {
          p *= 1;
          TRACTOR_INFO("THJ3");
        }
        target_robot_state.setJointPositions("lh_" + joint_name, &p);
      }
    };

    {
      ros::Duration move_to_start_duration(1);
      updateTargetRobotState();
      sendCommands(target_robot_state, target_hand_joint_names,
                   hand_command_pub, move_to_start_duration);
      move_to_start_duration.sleep();
    }

    double fps = 3;
    ros::Rate control_rate(fps);
    ros::Duration time_from_start(1.5 / fps);

    TRACTOR_INFO("start main loop");
    for (size_t i = 0; ros::ok(); i++) {

      TRACTOR_INFO("loop");

      // TRACTOR_INFO("press key to continue");
      // getchar();

      updateTargetRobotState();

      control_rate.sleep();

      sendCommands(target_robot_state, target_hand_joint_names,
                   hand_command_pub, time_from_start);
      sendCommands(target_robot_state, arm_joint_names, arm_command_pub,
                   time_from_start);

      step_policy();
    }
  }
}
