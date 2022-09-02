// (c) 2020-2022 Philipp Ruppel

#include <tractor/collision/base.h>

#include <tractor/core/log.h>
#include <tractor/geometry/eigen.h>
#include <tractor/geometry/plane.h>

#include <geometric_shapes/mesh_operations.h>

#include <random>

namespace tractor {

struct SurfaceSampler {
  struct Triangle {
    Vec3d pa, pb, pc;
    Vec3d normal;
    double area = 0;
    double area_sum = 0;
    Triangle(const Vector3<double> &pa, const Vector3<double> &pb,
             const Vector3<double> &pc)
        : pa(pa), pb(pb), pc(pc) {
      auto crs = cross(pb - pa, pc - pa);
      normal = normalized(crs);
      area = 0.5 * norm(crs);
    }
  };
  std::vector<Triangle> triangles;
  std::vector<double> area_sum;
  double total_area = 0;
  SurfaceSampler() {}
  SurfaceSampler(const shapes::Mesh *mesh) {
    auto getVertex = [&](size_t i) {
      return Vec3d(mesh->vertices[i * 3 + 0], mesh->vertices[i * 3 + 1],
                   mesh->vertices[i * 3 + 2]);
    };
    for (size_t itri = 0; itri < mesh->triangle_count; itri++) {
      Triangle tri(getVertex(mesh->triangles[itri * 3 + 0]),
                   getVertex(mesh->triangles[itri * 3 + 1]),
                   getVertex(mesh->triangles[itri * 3 + 2]));
      triangles.push_back(tri);
      total_area += tri.area;
    }
    {
      double sum = 0;
      for (auto &tri : triangles) {
        sum += tri.area;
        area_sum.push_back(sum);
      }
    }
  }
  void sample(Vec3d &point, Vec3d &normal) const {
    if (area_sum.empty()) {
      throw std::runtime_error("shape is empty");
    }
    static thread_local std::default_random_engine rng{std::random_device()()};
    std::uniform_real_distribution<double> sum_dist(0, total_area);
    auto it = std::upper_bound(area_sum.begin(), area_sum.end(), sum_dist(rng));
    if (it == area_sum.end()) {
      TRACTOR_WARN("failed to sample surface point");
      it = area_sum.begin();
    }
    auto &tri = triangles[it - area_sum.begin()];
    std::uniform_real_distribution<double> uv_dist(0, 1);
    double u = uv_dist(rng);
    double v = uv_dist(rng);
    if (u + v > 1) {
      u = 1 - u;
      v = 1 - v;
    }
    point = ((tri.pb - tri.pa) * u + (tri.pc - tri.pa) * v + tri.pa);
    normal = tri.normal;
  }
};

void ConvexCollisionMesh::sample(Vec3d &point, Vec3d &normal) const {
  surface_sampler->sample(point, normal);
}

void ConvexCollisionMesh::initConvexMesh(const std::string &name,
                                         const shapes::Mesh *mesh) {
  _name = name;
  surface_sampler = std::make_shared<SurfaceSampler>(mesh);
  for (size_t i = 0; i < mesh->vertex_count; i++) {
    _vertices.emplace_back(mesh->vertices[i * 3 + 0], mesh->vertices[i * 3 + 1],
                           mesh->vertices[i * 3 + 2]);
  }
}

} // namespace tractor
