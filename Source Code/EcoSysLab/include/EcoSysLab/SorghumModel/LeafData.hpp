#pragma once
#include "ProceduralSorghum.hpp"
#include <SorghumStateGenerator.hpp>
#include <Curve.hpp>
#include <LeafSegment.hpp>
#include "Spline.hpp"
using namespace UniEngine;
namespace EcoSysLab {
class LeafData : public IPrivateComponent {
  void LeafStateHelper(ProceduralLeafState& left, ProceduralLeafState& right, float& a, const SorghumStatePair &sorghumStatePair, int leafIndex);

  void GenerateLeafGeometry(const SorghumStatePair & sorghumStatePair, bool isBottomFace = false, float thickness = 0.001f);
public:
  glm::vec3 m_leafSheath;
  glm::vec3 m_leafTip;
  float m_branchingAngle;
  float m_rollAngle;
  int m_index = 0;

  //The "normal" direction of the leaf.
  glm::vec3 m_left;
  //Spline representation from Mathieu's skeleton
  std::vector<BezierCurve> m_curves;

  //Geometry generation
  std::vector<SplineNode> m_nodes;
  std::vector<LeafSegment> m_segments;
  std::vector<Vertex> m_vertices;
  std::vector<glm::uvec3> m_triangles;
  std::vector<Vertex> m_bottomFaceVertices;
  std::vector<glm::uvec3> m_bottomFaceTriangles;

  glm::vec4 m_vertexColor = glm::vec4(0, 1, 0, 1);

  void FormLeaf(const SorghumStatePair & sorghumStatePair, bool skeleton = false, bool doubleFace = false);
  void Copy(const std::shared_ptr<LeafData> &target);
  void OnInspect() override;
  void OnDestroy() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace EcoSysLab