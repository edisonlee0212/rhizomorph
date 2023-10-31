#pragma once

using namespace UniEngine;
namespace EcoSysLab {

class RectangularSorghumFieldPattern {
public:
  glm::ivec2 m_size = glm::ivec2(4, 4);
  glm::vec2 m_distances = glm::vec2(2, 2);
  glm::vec3 m_rotationVariation = glm::vec3(0, 0, 0);
  void GenerateField(std::vector<std::vector<glm::mat4>> &matricesList);
};

  class SorghumField : public IAsset {
  friend class SorghumLayer;

public:
  bool m_seperated = false;
  bool m_includeStem = true;

  int m_sizeLimit = 2000;
  float m_sorghumSize = 1.0f;
  std::vector<std::pair<AssetRef, glm::mat4>> m_newSorghums;
  virtual void GenerateMatrices(){};
  Entity InstantiateField();

  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
};

class RectangularSorghumField : public SorghumField {
  friend class SorghumLayer;
  AssetRef m_sorghumStateGenerator;
  glm::vec2 m_distance = glm::vec2(3.0f);
  glm::vec2 m_distanceVariance = glm::vec2(0.5f);
  glm::vec3 m_rotationVariance = glm::vec3(0.0f);
  glm::ivec2 m_size = glm::ivec2(10, 10);

public:
  void GenerateMatrices() override;

  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
};

class PositionsField : public SorghumField {
  friend class SorghumLayer;
public:
  AssetRef m_sorghumStateGenerator;
  float m_factor = 1.0f;
  std::vector<glm::dvec2> m_positions;
  glm::vec3 m_rotationVariance = glm::vec3(0.0f);

  glm::dvec2 m_sampleX = glm::dvec2(0.0);
  glm::dvec2 m_sampleY = glm::dvec2(0.0);

  glm::dvec2 m_xRange = glm::vec2(0, 0);
  glm::dvec2 m_yRange = glm::vec2(0, 0);
  void GenerateMatrices() override;
  std::pair<Entity, Entity> InstantiateAroundIndex(unsigned i, float radius, glm::dvec2& offset, float positionVariance = 0.0f);
  void ImportFromFile(const std::filesystem::path &path);
  void OnInspect() override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
};

template <typename T>
inline void SaveListAsBinary(const std::string &name,
                             const std::vector<T> &target, YAML::Emitter &out) {
  if (!target.empty()) {
    out << YAML::Key << name << YAML::Value
        << YAML::Binary((const unsigned char *)target.data(),
                        target.size() * sizeof(T));
  }
}
template <typename T>
inline void LoadListFromBinary(const std::string &name, std::vector<T> &target,
                               const YAML::Node &in) {
  if (in[name]) {
    auto binaryList = in[name].as<YAML::Binary>();
    target.resize(binaryList.size() / sizeof(T));
    std::memcpy(target.data(), binaryList.data(), binaryList.size());
  }
}
} // namespace EcoSysLab