#pragma once

#include "ecosyslab_export.h"
#include "PlantStructure.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
    struct TreeGraphNode {
        glm::vec3 m_start;
        float m_length;
        float m_thickness;
        int m_id;
        int m_parentId;
        bool m_fromApicalBud = false;
        glm::quat m_globalRotation;
        glm::vec3 m_position;
        std::weak_ptr<TreeGraphNode> m_parent;
        std::vector<std::shared_ptr<TreeGraphNode>> m_children;
    };

    class TreeGraph : public IAsset {
        void CollectChild(const std::shared_ptr<TreeGraphNode>& node, std::vector<std::vector<std::shared_ptr<TreeGraphNode>>>& graphNodes, int currentLayer) const;
    public:
        bool m_enableInstantiateLengthLimit = false;
        float m_instantiateLengthLimit = 8.0f;
        std::shared_ptr<TreeGraphNode> m_root;
        std::string m_name;
        int m_layerSize;
        void CollectAssetRef(std::vector<AssetRef>& list) override;

        void Serialize(YAML::Emitter& out) override;

        void Deserialize(const YAML::Node& in) override;

        void OnInspect() override;
    };

    class TreeGraphV2 : public IAsset {
        void CollectChild(const std::shared_ptr<TreeGraphNode>& node, std::vector<std::vector<std::shared_ptr<TreeGraphNode>>>& graphNodes, int currentLayer) const;
    public:
        bool m_enableInstantiateLengthLimit = false;
        float m_instantiateLengthLimit = 8.0f;
        std::shared_ptr<TreeGraphNode> m_root;
        std::string m_name;
        int m_layerSize;
        void CollectAssetRef(std::vector<AssetRef>& list) override;

        void Serialize(YAML::Emitter& out) override;

        void Deserialize(const YAML::Node& in) override;

        void OnInspect() override;
    };
}