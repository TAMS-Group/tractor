// (c) 2022 Philipp Ruppel

#include <tractor/python/common.h>
#include <tractor/ros/interact.h>
#include <tractor/ros/message.h>
#include <tractor/ros/publish.h>

#include <ros/ros.h>

namespace tractor {

template <class Scalar>
static void pythonizeROSTyped(py::module &main_module,
                              py::module &type_module) {

  main_module.def("interact",
                  [](const std::string &frame, const std::string &name,
                     Var<Vector3<Scalar>> &point,
                     double size) { interact(frame, name, point, size); });
}

TRACTOR_PYTHON_TYPED(pythonizeROSTyped);

static void pythonizeROS(py::module &m) {

  m.def("init_ros", [](const std::string &name) {
    TRACTOR_DEBUG("init_ros " << name);
    auto args =
        py::module::import("sys").attr("argv").cast<std::vector<std::string>>();
    std::vector<char *> argv;
    for (auto &a : args) {
      TRACTOR_DEBUG("arg " << a);
      argv.push_back((char *)a.c_str());
    }
    int argc = args.size();
    ros::init(argc, argv.data(), name,
              ros::init_options::NoSigintHandler | ros::init_options::NoRosout);
    static ros::NodeHandle node_handle("~");
    static ros::AsyncSpinner spinner(4);
  });

  m.def("publish", [](const std::string &topic, const py::object &message) {
    auto bytes_io = py::module::import("io").attr("BytesIO")();
    message.attr("serialize")(bytes_io);
    std::string serialized_data =
        bytes_io.attr("getvalue")().cast<std::string>();
    std::string type = message.attr("_type").cast<std::string>();
    std::string hash = message.attr("_md5sum").cast<std::string>();
    std::string definition = message.attr("_full_text").cast<std::string>();
    TRACTOR_DEBUG("publish " << topic << " " << type << " " << hash << " "
                             << definition << " " << serialized_data);
    publish(topic, Message(MessageType::instance(type, hash, definition),
                           serialized_data.data(), serialized_data.size()));
  });
}

TRACTOR_PYTHON_GLOBAL(pythonizeROS);

} // namespace tractor
