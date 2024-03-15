// 2020-2024 Philipp Ruppel

#include <tractor/collision/loader.h>

#include <tractor/collision/base.h>
#include <tractor/collision/shape.h>

#include <tractor/core/error.h>
#include <tractor/core/log.h>

#include <tractor/geometry/fast.h>
#include <tractor/geometry/eigen.h>

#include <geometric_shapes/mesh_operations.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#include <bullet/HACD/hacdHACD.h>

#include <geometric_shapes/shape_operations.h>

namespace tractor {

void _loadCollisionLink(CollisionRobot *collision_robot,
                        const moveit::core::LinkModel *link_model,
                        const Eigen::Isometry3d &link_transform,
                        std::shared_ptr<CollisionLink> collision_link) {
  auto new_collision_link =
      std::make_shared<CollisionLink>(link_model->getName());
  collision_robot->addLink(new_collision_link);

  if (!collision_link) {
    collision_link = new_collision_link;
  }

  auto &shapes = link_model->getShapes();
  auto &origins = link_model->getCollisionOriginTransforms();
  for (size_t shape_index = 0; shape_index < shapes.size(); shape_index++) {
    auto &shape = shapes[shape_index];
    TRACTOR_DEBUG("link " << link_model->getName() << " shape "
                          << typeid(*shape).name());
    auto &shape_origin = origins[shape_index];
    Eigen::Affine3d shape_pose = Eigen::Affine3d(link_transform) * shape_origin;
    Pose3d shape_pose_x = convertEigenToPose<GeometryFast<double>>(shape_pose);

    TRACTOR_INFO("shape type " << shape->type);

    if (auto *sphere = dynamic_cast<const shapes::Sphere *>(shape.get())) {
      TRACTOR_INFO("sphere " << sphere->radius << " " << shape_origin.matrix()
                             << " " << shape_pose.matrix());
      collision_link->addShape(collision_robot->engine()->createSphere(
          link_model->getName(), shape_pose_x, sphere->radius));
      continue;
    }

    if (auto *cylinder = dynamic_cast<const shapes::Cylinder *>(shape.get())) {
      TRACTOR_INFO("cylinder " << cylinder->length << " " << cylinder->radius
                               << " " << shape_origin.matrix() << " "
                               << shape_pose.matrix());
      collision_link->addShape(collision_robot->engine()->createCylinder(
          link_model->getName(), shape_pose_x, cylinder->length,
          cylinder->radius));
      continue;
    }

    if (auto *box = dynamic_cast<const shapes::Box *>(shape.get())) {
      TRACTOR_INFO("box " << box->size[0] << " " << box->size[1] << " "
                          << box->size[2] << " " << shape_origin.matrix() << " "
                          << shape_pose.matrix());
      collision_link->addShape(collision_robot->engine()->createBox(
          link_model->getName(), shape_pose_x,
          Vec3d(box->size[0], box->size[1], box->size[2])));
      continue;
    }

    std::shared_ptr<shapes::Mesh> mesh;
    if (!dynamic_cast<const shapes::Mesh *>(shape.get())) {
      mesh.reset(shapes::createMeshFromShape(shape.get()));
    } else {
      mesh.reset(dynamic_cast<shapes::Mesh *>(shape->clone()));
    }
    for (size_t i = 0; i < mesh->vertex_count; i++) {
      Eigen::Vector3d &v = ((Eigen::Vector3d *)mesh->vertices)[i];
      v = shape_pose * v;
    }

    if (mesh->vertex_count == 0 || mesh->triangle_count == 0) {
      continue;
    }

    if (shapes::computeShapeExtents(mesh.get()).norm() < 1.0) {
      TRACTOR_INFO("adding convex hull for link " << link_model->getName()
                                                  << " shape " << shape_index);

      auto collision_shape = collision_robot->engine()->createConvexMesh(
          link_model->getName(), mesh.get());
      collision_link->addShape(
          std::dynamic_pointer_cast<const CollisionShape>(collision_shape));

    } else {
      TRACTOR_INFO("computing convex decomposition for link "
                   << link_model->getName() << " shape " << shape_index);

      HACD::HACD hacd;
      hacd.SetConcavity(0.01);
      hacd.SetNClusters(1);
      hacd.SetAddExtraDistPoints(false);
      hacd.SetAddNeighboursDistPoints(false);
      hacd.SetAddFacesPoints(false);

      std::vector<HACD::Vec3<double>> hacd_points;
      std::vector<HACD::Vec3<long>> hacd_triangles;

      std::map<std::array<double, 3>, size_t> vertex_map;
      auto add_vertex = [&](double x, double y, double z) {
        std::array<double, 3> p = {x, y, z};
        auto it = vertex_map.find(p);
        if (it != vertex_map.end()) {
          return it->second;
        }
        size_t i = hacd_points.size();
        hacd_points.emplace_back(x, y, z);
        vertex_map[p] = i;
        return i;
      };
      for (size_t itri = 0; itri < mesh->triangle_count; itri++) {
        std::array<size_t, 3> tri;
        for (size_t iedge = 0; iedge < 3; iedge++) {
          size_t ivert = mesh->triangles[itri * 3 + iedge];
          tri[iedge] = add_vertex(mesh->vertices[ivert * 3 + 0],
                                  mesh->vertices[ivert * 3 + 1],
                                  mesh->vertices[ivert * 3 + 2]);
        }
        hacd_triangles.emplace_back(tri[0], tri[1], tri[2]);
      }

      hacd.SetPoints(hacd_points.data());
      hacd.SetNPoints(hacd_points.size());

      hacd.SetTriangles(hacd_triangles.data());
      hacd.SetNTriangles(hacd_triangles.size());

      TRACTOR_INFO(hacd_points.size() << " " << hacd_triangles.size());

      bool ok = hacd.Compute();
      if (!ok) {
        throw std::runtime_error("convex decomposition failed");
      }

      TRACTOR_SUCCESS("convex decomposition finished with "
                      << hacd.GetNClusters() << " components for link "
                      << link_model->getName() << " shape " << shape_index);

      for (size_t i_cluster = 0; i_cluster < hacd.GetNClusters(); i_cluster++) {
        std::vector<HACD::Vec3<double>> cluster_hacd_vertices(
            hacd.GetNPointsCH(i_cluster));

        std::vector<HACD::Vec3<long>> cluster_hacd_triangles(
            hacd.GetNTrianglesCH(i_cluster));

        hacd.GetCH(i_cluster, cluster_hacd_vertices.data(),
                   cluster_hacd_triangles.data());

        EigenSTL::vector_Vector3d cluster_mesh_vertices;
        for (auto &v : cluster_hacd_vertices) {
          cluster_mesh_vertices.emplace_back(v.X(), v.Y(), v.Z());
        }

        std::vector<unsigned int> cluster_mesh_indices;
        for (auto &t : cluster_hacd_triangles) {
          cluster_mesh_indices.emplace_back(t.X());
          cluster_mesh_indices.emplace_back(t.Y());
          cluster_mesh_indices.emplace_back(t.Z());
        }

        auto cluster_mesh =
            std::unique_ptr<shapes::Mesh>(shapes::createMeshFromVertices(
                cluster_mesh_vertices, cluster_mesh_indices));

        collision_link->addShape(
            std::dynamic_pointer_cast<const CollisionShape>(
                collision_robot->engine()->createConvexMesh(
                    link_model->getName(), cluster_mesh.get())));
      }
    }
  }

  for (auto *child_joint : link_model->getChildJointModels()) {
    auto *child_link = child_joint->getChildLinkModel();
    if (child_joint->getType() == moveit::core::JointModel::FIXED) {
      _loadCollisionLink(
          collision_robot, child_link,
          Eigen::Isometry3d(
              (link_transform * child_link->getJointOriginTransform())
                  .matrix()),
          collision_link);
    } else {
      _loadCollisionLink(collision_robot, child_link,
                         Eigen::Isometry3d::Identity(), nullptr);
    }
  }
}

void loadCollisionRobot(const std::shared_ptr<const CollisionEngine> &engine,
                        const moveit::core::RobotModel &moveit_model,
                        CollisionRobot *collision_robot) {
  TRACTOR_ASSERT(collision_robot->links().empty());

  _loadCollisionLink(collision_robot,
                     moveit_model.getRootJoint()->getChildLinkModel(),
                     Eigen::Isometry3d::Identity(), nullptr);
}

}  // namespace tractor
