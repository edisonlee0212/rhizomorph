//
// Created by lllll on 3/13/2022.
//
#include "LeafData.hpp"
#include "DefaultResources.hpp"
#include "Graphics.hpp"
#include "ProceduralSorghum.hpp"
#include "SorghumLayer.hpp"
using namespace EcoSysLab;
void LeafData::OnInspect() {
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
  static float nodeSize = 0.03f;
  static glm::vec4 renderColor = glm::vec4(1.0f);
  ImGui::Checkbox("Render nodes", &renderNodes);
  if (renderNodes) {
    if (ImGui::TreeNodeEx("Render settings", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::DragFloat("Size", &nodeSize, 0.001f, 0.001f, 1.0f);
      ImGui::ColorEdit4("Color", &renderColor.x);
      ImGui::TreePop();
    }
    std::vector<glm::mat4> matrices;
    matrices.resize(m_nodes.size());
    for (int i = 0; i < m_nodes.size(); i++) {
      matrices[i] =
          glm::translate(m_nodes[i].m_position) * glm::scale(glm::vec3(1.0f));
    }
    auto owner = GetOwner();
    auto scene = GetScene();
    Gizmos::DrawGizmoMeshInstanced(
        DefaultResources::Primitives::Sphere, renderColor, matrices,
        scene->GetDataComponent<GlobalTransform>(owner).m_value, nodeSize);
  }
}
void LeafData::OnDestroy() {
  m_curves.clear();
  m_nodes.clear();
  m_segments.clear();
  m_vertices.clear();
  m_triangles.clear();
  m_bottomFaceTriangles.clear();
  m_bottomFaceVertices.clear();
  m_vertexColor = glm::vec4(0, 1, 0, 1);
}
void LeafData::Serialize(YAML::Emitter &out) {
  out << YAML::Key << "m_left" << YAML::Value << m_left;
  out << YAML::Key << "m_index" << YAML::Value << m_index;
  out << YAML::Key << "m_leafSheath" << YAML::Value << m_leafSheath;
  out << YAML::Key << "m_leafTip" << YAML::Value << m_leafTip;
  out << YAML::Key << "m_branchingAngle" << YAML::Value << m_branchingAngle;
  out << YAML::Key << "m_rollAngle" << YAML::Value << m_rollAngle;

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
void LeafData::Deserialize(const YAML::Node &in) {
  if (in["m_left"])
    m_left = in["m_left"].as<glm::vec3>();
  if (in["m_index"])
    m_index = in["m_index"].as<int>();
  if (in["m_leafSheath"])
    m_leafSheath = in["m_leafSheath"].as<glm::vec3>();
  if (in["m_leafTip"])
    m_leafTip = in["m_leafTip"].as<glm::vec3>();
  if (in["m_branchingAngle"])
    m_branchingAngle = in["m_branchingAngle"].as<float>();
  if (in["m_rollAngle"])
    m_rollAngle = in["m_rollAngle"].as<float>();

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
void LeafData::GenerateLeafGeometry(const SorghumStatePair &sorghumStatePair,
                                    bool isBottomFace, float thickness) {
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  if (!sorghumLayer)
    return;

  auto scene = GetScene();

  ProceduralLeafState actualLeft, actualRight;
  float actualA;
  LeafStateHelper(actualLeft, actualRight, actualA, sorghumStatePair, m_index);

  auto *vertices = &m_vertices;
  auto *triangles = &m_triangles;
  if (isBottomFace) {
    vertices = &m_bottomFaceVertices;
    triangles = &m_bottomFaceTriangles;
  }

  vertices->clear();
  triangles->clear();
  m_segments.clear();

  if (m_nodes.empty())
    return;

  float leftFreq = glm::mix(actualLeft.m_wavinessFrequency.x,
                            actualRight.m_wavinessFrequency.x, actualA);
  float rightFreq = glm::mix(actualLeft.m_wavinessFrequency.y,
                             actualRight.m_wavinessFrequency.y, actualA);

  for (int i = 1; i < m_nodes.size(); i++) {
    auto &prev = m_nodes.at(i - 1);
    auto &curr = m_nodes.at(i);
    if (isBottomFace && !prev.m_isLeaf) {
      continue;
    }
    float distance = glm::distance(prev.m_position, curr.m_position);
    BezierCurve curve = BezierCurve(
        prev.m_position, prev.m_position + distance / 5.0f * prev.m_axis,
        curr.m_position - distance / 5.0f * curr.m_axis, curr.m_position);

    for (float div = (i == 1 ? 0.0f : 0.5f); div <= 1.0f; div += 0.5f) {
      float leftPeriod =
          glm::mix(actualLeft.m_wavinessPeriodStart.x,
                   actualRight.m_wavinessPeriodStart.x, actualA) +
          glm::mix(prev.m_range, curr.m_range, div) * leftFreq;
      float rightPeriod =
          glm::mix(actualLeft.m_wavinessPeriodStart.y,
                   actualRight.m_wavinessPeriodStart.y, actualA) +
          glm::mix(prev.m_range, curr.m_range, div) * rightFreq;

      auto front = prev.m_axis * (1.0f - div) + curr.m_axis * div;
      auto up = glm::normalize(glm::cross(m_left, front));
      auto waviness = glm::mix(prev.m_waviness, curr.m_waviness, div);
      m_segments.emplace_back(curve.GetPoint(div), up, front,
                              glm::mix(prev.m_stemWidth, curr.m_stemWidth, div),
                              glm::mix(prev.m_leafWidth, curr.m_leafWidth, div),
                              glm::mix(prev.m_theta, curr.m_theta, div),
                              curr.m_isLeaf, glm::sin(leftPeriod) * waviness,
                              glm::sin(rightPeriod) * waviness);
    }
  }

  const int vertexIndex = vertices->size();
  Vertex archetype{};
#pragma region Semantic mask color
  auto index = m_index + 1;
  m_vertexColor = glm::vec4((index % 3) * 0.5f, ((index / 3) % 3) * 0.5f,
                            ((index / 9) % 3) * 0.5f, 1.0f);
#pragma endregion
  archetype.m_color = m_vertexColor;

  const float xStep = 1.0f / sorghumLayer->m_horizontalSubdivisionStep / 2.0f;
  auto segmentSize = m_segments.size();
  const float yLeafStep = 0.5f / segmentSize;

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
      auto position = segment.GetPoint(
          (j - sorghumLayer->m_horizontalSubdivisionStep) * angleStep);
      auto normal = segment.GetNormal(
          (j - sorghumLayer->m_horizontalSubdivisionStep) * angleStep);
      if (i != 0 && isBottomFace && j != 0 && j != vertsCount - 1) {
        position -= normal * thickness;
      }
      archetype.m_position = glm::vec3(position.x, position.y, position.z);
      float yPos = 0.5f + yLeafStep * i;
      archetype.m_texCoord = glm::vec2(j * xStep, yPos);
      vertices->push_back(archetype);
    }
    if (i != 0) {
      for (int j = 0; j < vertsCount - 1; j++) {
        // Down triangle
        triangles->emplace_back(vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                vertexIndex + (i - 1) * vertsCount + j + 1,
                                vertexIndex + (i - 1) * vertsCount + j);
        // Up triangle
        triangles->emplace_back(vertexIndex + (i - 1) * vertsCount + j + 1,
                                vertexIndex + ((i - 1) + 1) * vertsCount + j,
                                vertexIndex + ((i - 1) + 1) * vertsCount + j +
                                    1);
      }
    }
  }
}
void LeafData::FormLeaf(const SorghumStatePair &sorghumStatePair, bool skeleton,
                        bool doubleFace) {
  auto scene = GetScene();

  ProceduralLeafState actualLeft, actualRight;
  float actualA;
  LeafStateHelper(actualLeft, actualRight, actualA, sorghumStatePair, m_index);

  float stemLength = sorghumStatePair.GetStemLength();
  auto stemDirection = sorghumStatePair.GetStemDirection();
  auto sorghumLayer = Application::GetLayer<SorghumLayer>();
  m_nodes.clear();
  auto startingPoint = glm::mix(actualLeft.m_startingPoint,
                                actualRight.m_startingPoint, actualA);
  float stemWidth = glm::mix(
      sorghumStatePair.m_left.m_stem.m_widthAlongStem.GetValue(startingPoint),
      sorghumStatePair.m_right.m_stem.m_widthAlongStem.GetValue(startingPoint),
      sorghumStatePair.m_a);
  float backDistance = 0.05f;
  if (startingPoint < backDistance)
    backDistance = startingPoint;
  float sheathPoint = startingPoint - backDistance;

  m_leafTip = m_leafSheath = sorghumStatePair.GetStemPoint(startingPoint);
  glm::vec3 direction;
  float leafLength;
  BezierSpline middleSpline;
  switch ((StateMode)sorghumStatePair.m_mode) {
  case StateMode::Default:
    m_rollAngle =
        glm::mix(actualLeft.m_rollAngle, actualRight.m_rollAngle, actualA);
    while (m_rollAngle > 360.0f)
      m_rollAngle -= 360.0f;
    while (m_rollAngle < 0.0f)
      m_rollAngle += 360.0f;
    m_left = glm::rotate(glm::vec3(0, 0, -1), glm::radians(m_rollAngle),
                         glm::vec3(0, 1, 0));
    m_branchingAngle = glm::mix(actualLeft.m_branchingAngle,
                                actualRight.m_branchingAngle, actualA);
    direction =
        glm::rotate(glm::vec3(0, 1, 0), glm::radians(m_branchingAngle), m_left);
    leafLength = glm::mix(actualLeft.m_length, actualRight.m_length, actualA);
    break;
  case StateMode::CubicBezier:
    assert(!actualLeft.m_spline.m_curves.empty() &&
           !actualRight.m_spline.m_curves.empty());
    assert(actualLeft.m_spline.m_curves.size() ==
           actualRight.m_spline.m_curves.size());
    middleSpline.m_curves.resize(actualLeft.m_spline.m_curves.size());
    leafLength = 0.0f;
    for (int i = 0; i < actualLeft.m_spline.m_curves.size(); i++) {
      middleSpline.m_curves[i].m_p0 =
          glm::mix(actualLeft.m_spline.m_curves[i].m_p0,
                   actualRight.m_spline.m_curves[i].m_p0, sorghumStatePair.m_a);
      middleSpline.m_curves[i].m_p1 =
          glm::mix(actualLeft.m_spline.m_curves[i].m_p1,
                   actualRight.m_spline.m_curves[i].m_p1, sorghumStatePair.m_a);
      middleSpline.m_curves[i].m_p2 =
          glm::mix(actualLeft.m_spline.m_curves[i].m_p2,
                   actualRight.m_spline.m_curves[i].m_p2, sorghumStatePair.m_a);
      middleSpline.m_curves[i].m_p3 =
          glm::mix(actualLeft.m_spline.m_curves[i].m_p3,
                   actualRight.m_spline.m_curves[i].m_p3, sorghumStatePair.m_a);
      leafLength += glm::distance(middleSpline.m_curves[i].m_p0,
                                  middleSpline.m_curves[i].m_p3);
    }
    m_left = glm::cross(glm::vec3(0, 1, 0),
                        middleSpline.EvaluateAxisFromCurves(0.0f));
    direction = middleSpline.EvaluateAxisFromCurves(0.0f);
    break;
  }
  if (leafLength == 0.0f)
    return;

  bool modelToRoot = true;
  if (modelToRoot) {
    float rootToSheath = startingPoint - backDistance;
    if (rootToSheath > 0) {
      int nodeForRootToSheath =
          glm::min(2.0f, stemLength * rootToSheath /
                             sorghumLayer->m_verticalSubdivisionMaxUnitLength);
      for (int i = 0; i < nodeForRootToSheath; i++) {
        float currentPoint = (float)i / nodeForRootToSheath * rootToSheath;
        glm::vec3 actualDirection = stemDirection;
        m_nodes.emplace_back(
            sorghumStatePair.GetStemPoint(currentPoint), 180.0f, stemWidth,
            (skeleton ? sorghumLayer->m_skeletonWidth : stemWidth), 0.0f,
            -actualDirection, false, 0.0f);
      }
    }
  }

  int nodeForSheath =
      glm::max(2.0f, stemLength * backDistance /
                         sorghumLayer->m_verticalSubdivisionMaxUnitLength);
  for (int i = 0; i <= nodeForSheath; i++) {
    float currentPoint = (float)i / nodeForSheath * backDistance;
    glm::vec3 actualDirection =
        glm::mix(stemDirection, direction, (float)i / nodeForSheath);
    m_nodes.emplace_back(
        sorghumStatePair.GetStemPoint(sheathPoint + currentPoint),
        (skeleton ? 180.0f : 180.0f - 90.0f * (float)i / nodeForSheath),
        stemWidth + 0.002f,
        (skeleton ? sorghumLayer->m_skeletonWidth
                  : stemWidth + 0.002f * (float)i / nodeForSheath),
        0.0f, -actualDirection, false, 0.0f);
  }

  int nodeAmount = glm::max(
      4.0f, leafLength / sorghumLayer->m_verticalSubdivisionMaxUnitLength);
  float unitLength = leafLength / nodeAmount;

  int nodeToFullExpand =
      0.1f * leafLength / sorghumLayer->m_verticalSubdivisionMaxUnitLength;

  for (int i = 1; i <= nodeAmount; i++) {
    const float factor = (float)i / nodeAmount;
    glm::vec3 currentDirection;
    switch ((StateMode)sorghumStatePair.m_mode) {
    case StateMode::Default: {
      float rotateAngle =
          glm::mix(actualLeft.m_bendingAlongLeaf.GetValue(factor),
                   actualRight.m_bendingAlongLeaf.GetValue(factor), actualA);
      currentDirection =
          glm::rotate(direction, glm::radians(rotateAngle), m_left);
      m_leafTip += currentDirection * unitLength;
    } break;
    case StateMode::CubicBezier:
      currentDirection = middleSpline.EvaluateAxisFromCurves(factor);
      m_leafTip = middleSpline.EvaluatePointFromCurves(factor);
      break;
    }
    float expandAngle =
        glm::mix(actualLeft.m_curlingAlongLeaf.GetValue(factor),
                 actualRight.m_curlingAlongLeaf.GetValue(factor), actualA);
    float collarFactor = glm::min(1.0f, (float)i / nodeToFullExpand);
    float wavinessAlongLeaf =
        glm::mix(actualLeft.m_wavinessAlongLeaf.GetValue(factor),
                 actualRight.m_wavinessAlongLeaf.GetValue(factor), actualA);
    float width = glm::mix(
        stemWidth + 0.002f,
        glm::mix(actualLeft.m_widthAlongLeaf.GetValue(factor),
                 actualRight.m_widthAlongLeaf.GetValue(factor), actualA),
        collarFactor);
    float angle = 90.0f - (90.0f - expandAngle) * glm::pow(collarFactor, 2.0f);
    m_nodes.emplace_back(m_leafTip, (skeleton ? 180.0f : angle),
                         stemWidth + 0.002f,
                         (skeleton ? sorghumLayer->m_skeletonWidth : width),
                         wavinessAlongLeaf, -currentDirection, true, factor);
  }
  GenerateLeafGeometry(sorghumStatePair);
  if (!skeleton && doubleFace)
    GenerateLeafGeometry(sorghumStatePair, true);
}
void LeafData::Copy(const std::shared_ptr<LeafData> &target) {
  *this = *target;
}
void LeafData::LeafStateHelper(ProceduralLeafState &left,
                               ProceduralLeafState &right, float &a,
                               const SorghumStatePair &sorghumStatePair,
                               int leafIndex) {
  int previousLeafSize = sorghumStatePair.m_left.m_leaves.size();
  int nextLeafSize = sorghumStatePair.m_right.m_leaves.size();
  if (leafIndex < previousLeafSize) {
    left = sorghumStatePair.m_left.m_leaves[leafIndex];
    if (left.m_dead)
      left.m_length = 0;
    if (leafIndex < nextLeafSize) {
      if (sorghumStatePair.m_right.m_leaves[leafIndex].m_dead ||
          sorghumStatePair.m_right.m_leaves[leafIndex].m_length == 0)
        right = left;
      else {
        right = sorghumStatePair.m_right.m_leaves[leafIndex];
      }
    } else {
      right = sorghumStatePair.m_left.m_leaves[leafIndex];
    }
    a = sorghumStatePair.m_a;
    return;
  }

  int completedLeafSize = sorghumStatePair.m_left.m_leaves.size() +
                          glm::floor((sorghumStatePair.m_right.m_leaves.size() -
                                      sorghumStatePair.m_left.m_leaves.size()) *
                                     sorghumStatePair.m_a);
  a = glm::clamp(sorghumStatePair.m_a * (nextLeafSize - previousLeafSize) -
                     (completedLeafSize - previousLeafSize),
                 0.0f, 1.0f);
  left = right = sorghumStatePair.m_right.m_leaves[leafIndex];
  if (leafIndex >= completedLeafSize) {
    left.m_length = 0.0f;
    left.m_widthAlongLeaf.m_minValue = left.m_widthAlongLeaf.m_maxValue = 0.0f;
    left.m_wavinessAlongLeaf.m_minValue = left.m_wavinessAlongLeaf.m_maxValue =
        0.0f;
    for (auto &i : left.m_spline.m_curves) {
      i.m_p0 = i.m_p1 = i.m_p2 = i.m_p3 =
          right.m_spline.EvaluatePointFromCurves(0.0f);
    }
  } else {
    left = right;
  }
}
