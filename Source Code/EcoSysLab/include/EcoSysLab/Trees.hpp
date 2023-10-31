#pragma once
#include "Tree.hpp"
using namespace UniEngine;
namespace EcoSysLab {
    class Trees : public IAsset{
    public:
        std::vector<Transform> m_globalTransforms;
        AssetRef m_treeDescriptor;

        TreeGrowthSettings m_treeGrowthSettings;

        void OnInspect() override;

        void OnCreate() override;

        void CollectAssetRef(std::vector<AssetRef> &list) override;

        void Serialize(YAML::Emitter &out) override;

        void Deserialize(const YAML::Node &in) override;
    };
}