// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>

#include <tractor/ros/interact.h>
#include <tractor/ros/message.h>
#include <tractor/ros/publish.h>
#include <tractor/ros/visualize.h>

#include <ros/ros.h>

namespace tractor {

template <class Scalar>
static void pythonizeROSTyped(py::module main_module, py::module type_module) {
  main_module.def("interact",
                  [](const std::string &frame, const std::string &name,
                     Var<Vector3<Scalar>> &point, double size) {
                    return interact(frame, name, point, Scalar(size));
                  });

  main_module.def("interact",
                  [](const std::string &frame, const std::string &name,
                     Var<Pose<Scalar>> &pose, double size) {
                    return interact(frame, name, pose, Scalar(size));
                  });
}

TRACTOR_PYTHON_TYPED(pythonizeROSTyped);

static void pythonizeROS(py::module m) {
  m.def("visualize_points",
        py::overload_cast<const std::string &, double, const Eigen::Vector4d &,
                          const std::vector<Eigen::Vector3d> &>(
            &visualizePoints));

  m.def("visualize_points",
        py::overload_cast<const std::string &, double,
                          const std::vector<Eigen::Vector4d> &,
                          const std::vector<Eigen::Vector3d> &>(
            &visualizePoints));

  m.def(
      "visualize_lines",
      py::overload_cast<const std::string &, double, const Eigen::Vector4d &,
                        const std::vector<Eigen::Vector3d> &>(&visualizeLines));

  m.def(
      "visualize_lines",
      py::overload_cast<const std::string &, double,
                        const std::vector<Eigen::Vector4d> &,
                        const std::vector<Eigen::Vector3d> &>(&visualizeLines));

  // m.def(
  //     "visualize_mesh",
  //     py::overload_cast<const std::string &, const Eigen::Vector4d &,
  //                       const std::vector<Eigen::Vector3d>
  //                       &>(&visualizeMesh));

  // m.def(
  //     "visualize_mesh",
  //     py::overload_cast<const std::string &,
  //                       const std::vector<Eigen::Vector4d> &,
  //                       const std::vector<Eigen::Vector3d>
  //                       &>(&visualizeMesh));

  m.def("visualize_mesh",
        [](const std::string &name, const py::array_t<float> &colors,
           const py::array_t<float> &vertices) {
          TRACTOR_ASSERT(colors.ndim() == 2);
          TRACTOR_ASSERT(vertices.ndim() == 2);
          TRACTOR_ASSERT(colors.shape(0) == vertices.shape(0));
          TRACTOR_ASSERT(colors.shape(1) == 4);
          TRACTOR_ASSERT(vertices.shape(1) == 3);

          size_t count = vertices.shape(0);

          auto vertex_data = vertices.unchecked<2>();
          auto color_data = colors.unchecked<2>();

          std::vector<Eigen::Vector3d> vertex_vector(count);
          std::vector<Eigen::Vector4d> color_vector(count);

          for (size_t i = 0; i < count; i++) {
            vertex_vector[i].x() = vertex_data(i, 0);
            vertex_vector[i].y() = vertex_data(i, 1);
            vertex_vector[i].z() = vertex_data(i, 2);

            color_vector[i].x() = color_data(i, 0);
            color_vector[i].y() = color_data(i, 1);
            color_vector[i].z() = color_data(i, 2);
            color_vector[i].w() = color_data(i, 3);
          }

          visualizeMesh(name, color_vector, vertex_vector);
        });

  m.def("clear_visualization", &clearVisualization);

  m.def("ros_ok", []() { return ros::ok(); });

  auto init_ros = [](const std::string &name, bool sigint_handler = false) {
    TRACTOR_DEBUG("init_ros " << name);
    auto args =
        py::module::import("sys").attr("argv").cast<std::vector<std::string>>();
    std::vector<char *> argv;
    for (auto &a : args) {
      TRACTOR_DEBUG("arg " << a);
      argv.push_back((char *)a.c_str());
    }
    int argc = args.size();
    int flags = ros::init_options::NoRosout;
    if (!sigint_handler) {
      flags |= ros::init_options::NoSigintHandler;
    }
    ros::init(argc, argv.data(), name, flags);
    static ros::NodeHandle node_handle("~");
    static ros::AsyncSpinner spinner(4);
    clearVisualization();
  };
  m.def("init_ros", init_ros);
  m.def("init_ros", [init_ros](const std::string &name) { init_ros(name); });

  m.def("publish", [](const std::string &topic, const py::object &message) {
    auto bytes_io = py::module::import("io").attr("BytesIO")();
    message.attr("serialize")(bytes_io);
    std::string serialized_data =
        bytes_io.attr("getvalue")().cast<std::string>();
    std::string type = message.attr("_type").cast<std::string>();
    std::string hash = message.attr("_md5sum").cast<std::string>();
    std::string definition = message.attr("_full_text").cast<std::string>();
    // TRACTOR_DEBUG("publish " << topic << " " << type << " " << hash << " "
    //                          << definition << " " << serialized_data);
    publish(topic, Message(MessageType::instance(type, hash, definition),
                           serialized_data.data(), serialized_data.size()));
  });

  m.def("advertise", [](const std::string &topic, const py::object &type) {
    std::string name = type.attr("_type").cast<std::string>();
    std::string hash = type.attr("_md5sum").cast<std::string>();
    std::string definition = type.attr("_full_text").cast<std::string>();
    // TRACTOR_DEBUG("advertise " << topic << " type:" << name << " hash:" <<
    // hash
    //                            << " def:" << definition);
    advertise(topic, hash, name, definition);
  });
}

TRACTOR_PYTHON_GLOBAL(pythonizeROS);

}  // namespace tractor
