//
// Created by lllll on 3/13/2022.
//

#include "PanicleData.hpp"
#include "IVolume.hpp"
using namespace EcoSysLab;
void PanicleData::OnInspect() {

}
void PanicleData::OnDestroy() {
  m_vertices.clear();
  m_triangles.clear();
}
void PanicleData::Serialize(YAML::Emitter &out) {
  ISerializable::Serialize(out);
}
void PanicleData::Deserialize(const YAML::Node &in) {
  ISerializable::Deserialize(in);
}
void PanicleData::FormPanicle(const SorghumStatePair &sorghumStatePair) {
  m_vertices.clear();
  m_triangles.clear();
  auto pinnacleSize = glm::mix(sorghumStatePair.m_left.m_panicle.m_panicleSize, sorghumStatePair.m_right.m_panicle.m_panicleSize, sorghumStatePair.m_a);
  auto seedAmount = glm::mix(sorghumStatePair.m_left.m_panicle.m_seedAmount, sorghumStatePair.m_right.m_panicle.m_seedAmount, sorghumStatePair.m_a);
  auto seedRadius = glm::mix(sorghumStatePair.m_left.m_panicle.m_seedRadius, sorghumStatePair.m_right.m_panicle.m_seedRadius, sorghumStatePair.m_a);
  std::vector<glm::vec3> icosahedronVertices;
  std::vector<glm::uvec3> icosahedronTriangles;
  SphereMeshGenerator::Icosahedron(icosahedronVertices, icosahedronTriangles);
  int offset = 0;
  UniEngine::Vertex archetype = {};
  SphericalVolume volume;
  volume.m_radius = pinnacleSize;
  for (int seedIndex = 0;
       seedIndex < seedAmount;
       seedIndex++) {
    glm::vec3 positionOffset = volume.GetRandomPoint();
    for (const auto position : icosahedronVertices) {
      archetype.m_position =
          position * seedRadius + glm::vec3(0, pinnacleSize.y, 0) +
          positionOffset + sorghumStatePair.GetStemPoint(1.0f);
      m_vertices.push_back(archetype);
    }
    for (const auto triangle : icosahedronTriangles) {
      glm::uvec3 actualTriangle = triangle + glm::uvec3(offset);
      m_triangles.push_back(actualTriangle);
    }
    offset += icosahedronVertices.size();
  }
}
