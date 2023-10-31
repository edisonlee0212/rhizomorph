#pragma once
#include "ProceduralSorghum.hpp"
#include <SorghumStateGenerator.hpp>
using namespace UniEngine;
namespace EcoSysLab {
enum class SorghumMode{
  ProceduralSorghum,
  SorghumStateGenerator
};

class SorghumData : public IPrivateComponent {
  float m_currentTime = 1.0f;
  unsigned m_recordedVersion = 0;
  friend class SorghumLayer;
  bool m_segmentedMask = false;
public:
  int m_mode = (int)SorghumMode::ProceduralSorghum;
  glm::vec3 m_gravityDirection = glm::vec3(0, -1, 0);
  bool m_meshGenerated = false;
  AssetRef m_descriptor;
  int m_seed = 0;
  bool m_skeleton = false;
  bool m_seperated = true;
  bool m_includeStem = true;
  bool m_bottomFace = false;

  void OnCreate() override;
  void OnDestroy() override;
  void OnInspect() override;
  void SetTime(float time);
  void ExportModel(const std::string &filename,
                   const bool &includeFoliage = true) const;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void FormPlant();
  void ApplyGeometry();

  void SetEnableSegmentedMask(bool value);
};
} // namespace PlantFactory
