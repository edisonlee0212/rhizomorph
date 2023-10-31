//
// Created by lllll on 2/23/2022.
//
#ifdef RAYTRACERFACILITY
#include "PARSensorGroup.hpp"
#include "RayTracerLayer.hpp"
#include "DefaultResources.hpp"
#include "Graphics.hpp"

using namespace EcoSysLab;
void EcoSysLab::PARSensorGroup::CalculateIllumination(
    const RayProperties &rayProperties, int seed, float pushNormalDistance) {
  if (m_samplers.empty())
    return;
  CudaModule::EstimateIlluminationRayTracing(
      Application::GetLayer<RayTracerLayer>()->m_environmentProperties,
      rayProperties, m_samplers, seed, pushNormalDistance);
}
void PARSensorGroup::OnInspect() {
  ImGui::Text("Sampler size: %llu", m_samplers.size());
  if (ImGui::TreeNode("Grid settings")) {
    static auto minRange = glm::vec3(-25, 0, -25);
    static auto maxRange = glm::vec3(25, 3, 25);
    static float step = 3.0f;
    if (ImGui::DragFloat3("Min", &minRange.x, 0.1f)) {
      minRange = (glm::min)(minRange, maxRange);
    }
    if (ImGui::DragFloat3("Max", &maxRange.x, 0.1f)) {
      maxRange = (glm::max)(minRange, maxRange);
    }
    if (ImGui::DragFloat("Step", &step, 0.01f)) {
      step = glm::clamp(step, 0.1f, 10.0f);
    }
    if (ImGui::Button("Instantiate")) {
      const int sx = (int)((maxRange.x - minRange.x + step) / step);
      const int sy = (int)((maxRange.y - minRange.y + step) / step);
      const int sz = (int)((maxRange.z - minRange.z + step) / step);
      auto voxelSize = sx * sy * sz;
      m_samplers.resize(voxelSize);
      std::vector<std::shared_future<void>> results;
      Jobs::ParallelFor(
          voxelSize,
          [&](unsigned i) {
            float z = (i % sz) * step + minRange.z;
            float y = ((i / sz) % sy) * step + minRange.y;
            float x = ((i / sz / sy) % sx) * step + minRange.x;
            glm::vec3 start = {x, y, z};
            m_samplers[i].m_a.m_position = m_samplers[i].m_b.m_position = m_samplers[i].m_c.m_position = start;
            m_samplers[i].m_frontFace = true;
						m_samplers[i].m_backFace = false;
						m_samplers[i].m_a.m_normal = m_samplers[i].m_b.m_normal = m_samplers[i].m_c.m_normal = glm::vec3(0, 1, 0);
          },
          results);
      for (const auto &i : results)
        i.wait();
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNode("Estimation")) {
    static RayProperties rayProperties = {8, 1000};
    rayProperties.OnInspect();
    if (ImGui::Button("Run!"))
      CalculateIllumination(rayProperties, 0, 0.0f);
    ImGui::TreePop();
  }
  static bool draw = true;
  ImGui::Checkbox("Render field", &draw);
  if (draw && !m_samplers.empty()) {
    static float lineWidth = 0.05f;
    static float lineLengthFactor = 3.0f;
    static float pointSize = 0.1f;
    static std::vector<glm::vec3> starts;
    static std::vector<glm::vec3> ends;
    static std::vector<glm::mat4> matrices;
    static glm::vec4 color = {0.0f, 1.0f, 0.0f, 0.5f};
    static glm::vec4 pointColor = {1.0f, 0.0f, 0.0f, 0.75f};
    starts.resize(m_samplers.size());
    ends.resize(m_samplers.size());
    matrices.resize(m_samplers.size());
    ImGui::DragFloat("Vector width", &lineWidth, 0.01f);
    ImGui::DragFloat("Vector length factor", &lineLengthFactor, 0.01f);
    ImGui::ColorEdit4("Vector Color", &color.x);
    ImGui::DragFloat("Point Size", &pointSize, 0.01f);
    ImGui::ColorEdit4("Point Color", &pointColor.x);
    std::vector<std::shared_future<void>> results;
    Jobs::ParallelFor(
        m_samplers.size(),
        [&](unsigned i) {
          const auto start = m_samplers[i].m_a.m_position;
          starts[i] = start;
          ends[i] = start + m_samplers[i].m_direction * lineLengthFactor * m_samplers[i].m_energy;
          matrices[i] =
              glm::translate(start) * glm::scale(glm::vec3(pointSize));
        },
        results);
    for (const auto &i : results)
      i.wait();
    Gizmos::DrawGizmoRays(color, starts, ends, lineWidth);
    Gizmos::DrawGizmoMeshInstanced(DefaultResources::Primitives::Cube,
                                     pointColor, matrices);
  }
}
void PARSensorGroup::Serialize(YAML::Emitter &out) {
  if (!m_samplers.empty())
  {
    out << YAML::Key << "m_samplers" << YAML::Value
        << YAML::Binary((const unsigned char *)m_samplers.data(), m_samplers.size() * sizeof(IlluminationSampler<glm::vec3>));
  }
}
void PARSensorGroup::Deserialize(const YAML::Node &in) {
  if (in["m_samplers"])
  {
    auto binaryList = in["m_samplers"].as<YAML::Binary>();
    m_samplers.resize(binaryList.size() / sizeof(IlluminationSampler<glm::vec3>));
    std::memcpy(m_samplers.data(), binaryList.data(), binaryList.size());
  }
}
#endif