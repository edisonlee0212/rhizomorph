#pragma once

using namespace UniEngine;
namespace EcoSysLab {

class CBTFGroup : public IAsset{
public:
  std::vector<AssetRef> m_doubleCBTFs;
  void OnInspect() override;
  void CollectAssetRef(std::vector<AssetRef> &list) override;
  void Serialize(YAML::Emitter &out) override;
  void Deserialize(const YAML::Node &in) override;
  AssetRef GetRandom() const;
};
}