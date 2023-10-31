#pragma once
#include "ProceduralSorghum.hpp"
#include <SorghumStateGenerator.hpp>

using namespace UniEngine;
namespace EcoSysLab {
class PanicleData : public IPrivateComponent {
public:
  std::vector<Vertex> m_vertices;
  std::vector<glm::uvec3> m_triangles;
  void FormPanicle(const SorghumStatePair & sorghumStatePair);
  void OnInspect() override;
  void OnDestroy() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
};
} // namespace EcoSysLab