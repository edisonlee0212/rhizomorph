//
// Created by lllll on 3/13/2022.
//

#include "StemData.hpp"
#include "DefaultResources.hpp"
#include "Graphics.hpp"
#include "SorghumLayer.hpp"
using namespace EcoSysLab;
void StemData::OnInspect() {
  if (ImGui::TreeNodeEx("Curves", ImGuiTreeNodeFlags_DefaultOpen)) {
    for (int i = 0; i < m_curves.size(); i++) {
      ImGui::Text(("Curve" + std::to_string(i)).c_str());
      ImGui::InputFloat3("CP0", &m_curves[i].m_p0.x);
      ImGui::InputFloat3("CP1", &m_curves[i].m_p1.x);
      ImGui::InputFloat3("CP2", &m_curves[i].m_p2.x);
      ImGui::InputFloat3("CP3", &m_curves[i].m_p3.x);
    }
    ImGui::TreePop();
  }
  static bool renderNodes = false;
  static float nodeSize = 0.1f;
  static glm::vec4 renderColor = glm::vec4(1.0f);
  ImGui::Checkbox("Render nodes", &renderNodes);
  if (renderNodes) {
    if (ImGui::TreeNodeEx("Render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::DragFloat("Size", &nodeSize, 0.01f, 0.01f, 1.0f);
      ImGui::ColorEdit4("Color", &renderColor.x);
      ImGui::TreePop();
    }
    std::vector<glm::mat4> matrices;
    matrices.resize(m_nodes.size());
    for (int i = 0; i < m_nodes.size(); i++) {
      matrices[i] =
          glm::translate(m_nodes[i].m_position) * glm::scale(glm::vec3(1.0f));
    }
    Gizmos::DrawGizmoMeshInstanced(
        DefaultResources::Primitives::Sphere, renderColor, matrices,
        GetScene()->GetDataComponent<GlobalTransform>(GetOwner()).m_value,
        nodeSize);
  }
}
void StemData::OnDestroy() {
  m_curves.clear();
  m_nodes.clear();
  m_segments.clear();
  m_vertices.clear();
  m_triangles.clear();
  m_vertexColor = glm::vec4(0, 1, 0, 1);
}
void StemData::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_left" << YAML::Value << m_left;
  out << YAML::Key << "m_curves" << YAML::BeginSeq;
  for (const auto &i : m_curves) {
    out << YAML::BeginMap;
    out << YAML::Key << "m_p0" << YAML::Value << i.m_p0;
    out << YAML::Key << "m_p1" << YAML::Value << i.m_p1;
    out << YAML::Key << "m_p2" << YAML::Value << i.m_p2;
    out << YAML::Key << "m_p3" << YAML::Value << i.m_p3;
    out << YAML::EndMap;
  }
  out << YAML::EndSeq;

  if (!m_nodes.empty()) {
    out << YAML::Key << "m_nodes" << YAML::Value
        << YAML::Binary((const unsigned char *)m_nodes.data(),
                        m_nodes.size() * sizeof(SplineNode));
  }
}
void EcoSysLab::StemData::Deserialize(const YAML::Node &in) {

  m_left = in["m_left"].as<glm::vec3>();
  if (in["m_curves"]) {
    m_curves.clear();
    for (const auto &i : in["m_curves"]) {
      m_curves.push_back(
          BezierCurve(i["m_p0"].as<glm::vec3>(), i["m_p1"].as<glm::vec3>(),
                      i["m_p2"].as<glm::vec3>(), i["m_p3"].as<glm::vec3>()));
    }
  }

  if (in["m_nodes"]) {
    YAML::Binary nodes = in["m_nodes"].as<YAML::Binary>();
    m_nodes.resize(nodes.size() / sizeof(SplineNode));
    std::memcpy(m_nodes.data(), nodes.data(), nodes.size());
  }
}
void StemData::GenerateStemGeometry() {
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (!sorghumLayer)
    return;

  m_vertices.clear();
  m_triangles.clear();
  m_segments.clear();

  for (int i = 1; i < m_nodes.size(); i++) {
    auto &prev = m_nodes.at(i - 1);
    auto &curr = m_nodes.at(i);
    float distance = glm::distance(prev.m_position, curr.m_position);
    BezierCurve curve = BezierCurve(
        prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis,
        curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);
    for (float div = (i == 1 ? 0.0f : 0.5f); div <= 1.0f; div += 0.5f) {
      auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;
      auto up = glm::normalize(glm::cross(m_left, front));
      m_segments.emplace_back(
          curve.GetPoint(div), up, front,
          prev.m_stemWidth * (1.0f - div) + curr.m_stemWidth * div,
          prev.m_leafWidth * (1.0f - div) + curr.m_leafWidth * div,
          prev.m_theta * (1.0f - div) + curr.m_theta * div, curr.m_isLeaf, 1.0f,
          1.0f);
    }
  }
  const int vertexIndex = m_vertices.size();
  Vertex archetype{};
  m_vertexColor = glm::vec4(0, 0, 0, 1);
  archetype.m_color = m_vertexColor;
  const float xStep = 1.0f / sorghumLayer->m_horizontalSubdivisionStep / 2.0f;
  auto segmentSize = m_segments.size();
  const float yStemStep = 0.5f / segmentSize;
  for (int i = 0; i < segmentSize; i++) {
    auto &segment = m_segments.at(i);
    if (i <= segmentSize / 3) {
      archetype.m_color = glm::vec4(1, 0, 0, 1);
    } else if (i <= segmentSize * 2 / 3) {
      archetype.m_color = glm::vec4(0, 1, 0, 1);
    } else {
      archetype.m_color = glm::vec4(0, 0, 1, 1);
    }
    const float angleStep =
        segment.m_theta / sorghumLayer->m_horizontalSubdivisionStep;
    const int vertsCount = sorghumLayer->m_horizontalSubdivisionStep * 2 + 1;
    for (int j = 0; j < vertsCount; j++) {
      const auto position = segment.GetPoint(
          (j - sorghumLayer->m_horizontalSubdivisionStep) * angleStep);
      archetype.m_position = glm::vec3(position.x, position.y, position.z);
      float yPos = yStemStep * i;
      archetype.m_texCoord = glm::vec2(j * xStep, yPos);
      m_vertices.push_back(archetype);
    }
    if (i != 0) {
      for (int j = 0; j < vertsCount - 1; j++) {
        // Down triangle
        m_triangles.emplace_back(vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                 vertexIndex + (i - 1) * vertsCount + j + 1,
                                 vertexIndex + (i - 1) * vertsCount + j);
        // Up triangle
        m_triangles.emplace_back(vertexIndex + (i - 1) * vertsCount + j + 1,
                                 vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                 vertexIndex + ((i - 1) + 1) * vertsCount + j +
                                     1);
      }
    }
  }
}
void StemData::FormStem(const SorghumStatePair &sorghumStatePair,
                        bool skeleton) {
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  float length = sorghumStatePair.GetStemLength();
  auto direction = sorghumStatePair.GetStemDirection();
  int nodeAmount = (int)glm::max(
      4.0f, length / sorghumLayer->m_verticalSubdivisionMaxUnitLength);
  float unitLength = length / nodeAmount;

  m_nodes.clear();
  for (int i = 0; i <= nodeAmount; i++) {
    float stemWidth =
        glm::mix(sorghumStatePair.m_left.m_stem.m_widthAlongStem.GetValue(
                     (float)i / nodeAmount),
                 sorghumStatePair.m_right.m_stem.m_widthAlongStem.GetValue(
                     (float)i / nodeAmount),
                 sorghumStatePair.m_a);
    if (skeleton)
      stemWidth = sorghumLayer->m_skeletonWidth;
    glm::vec3 position;
    switch ((StateMode)sorghumStatePair.m_mode) {
    case StateMode::Default:
      position = glm::normalize(direction) * unitLength * static_cast<float>(i);
      break;
    case StateMode::CubicBezier:
      position = glm::mix(
          sorghumStatePair.m_left.m_stem.m_spline.EvaluatePointFromCurves(
              (float)i / nodeAmount),
          sorghumStatePair.m_right.m_stem.m_spline.EvaluatePointFromCurves(
              (float)i / nodeAmount),
          sorghumStatePair.m_a);
      direction = glm::mix(
          sorghumStatePair.m_left.m_stem.m_spline.EvaluateAxisFromCurves(
              (float)i / nodeAmount),
          sorghumStatePair.m_right.m_stem.m_spline.EvaluateAxisFromCurves(
              (float)i / nodeAmount),
          sorghumStatePair.m_a);
      break;
    }
    m_nodes.emplace_back(position, 180.0f, stemWidth, stemWidth, 0.0f,
                         -direction, false, (float)i / nodeAmount);
  }
  m_left = glm::rotate(glm::vec3(1, 0, 0),
                       glm::radians(glm::linearRand(0.0f, 0.0f)), direction);
  GenerateStemGeometry();
}
void StemData::Copy(const std::shared_ptr<StemData> &target) {
  *this = *target;
}
