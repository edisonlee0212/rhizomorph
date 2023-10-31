#include "TreeGraph.hpp"

using namespace EcoSysLab;

void TreeGraph::Serialize(YAML::Emitter& out) {
    out << YAML::Key << "name" << YAML::Value << m_name;
    out << YAML::Key << "layersize" << YAML::Value << m_layerSize;
    out << YAML::Key << "layers" << YAML::Value << YAML::BeginMap;
    std::vector<std::vector<std::shared_ptr<TreeGraphNode>>> graphNodes;
    graphNodes.resize(m_layerSize);
    CollectChild(m_root, graphNodes, 0);
    for (int layerIndex = 0; layerIndex < m_layerSize; layerIndex++) {
        out << YAML::Key << std::to_string(layerIndex) << YAML::Value << YAML::BeginMap;
        {
            auto& layer = graphNodes[layerIndex];
            out << YAML::Key << "internodesize" << YAML::Value << layer.size();
            for (int nodeIndex = 0; nodeIndex < layer.size(); nodeIndex++) {
                auto node = layer[nodeIndex];
                out << YAML::Key << std::to_string(nodeIndex) << YAML::Value << YAML::BeginMap;
                {
                    out << YAML::Key << "id" << YAML::Value << node->m_id;
                    out << YAML::Key << "parent" << YAML::Value << node->m_parentId;
                    out << YAML::Key << "quat" << YAML::Value << YAML::BeginSeq;
                    for (int i = 0; i < 4; i++) {
                        out << YAML::BeginMap;
                        out << std::to_string(node->m_globalRotation[i]);
                        out << YAML::EndMap;
                    }
                    out << YAML::EndSeq;

                    out << YAML::Key << "position" << YAML::Value << YAML::BeginSeq;
                    for (int i = 0; i < 3; i++) {
                        out << YAML::BeginMap;
                        out << std::to_string(node->m_position[i]);
                        out << YAML::EndMap;
                    }
                    out << YAML::EndSeq;

                    out << YAML::Key << "thickness" << YAML::Value << node->m_thickness;
                    out << YAML::Key << "length" << YAML::Value << node->m_length;
                }
                out << YAML::EndMap;
            }
        }
        out << YAML::EndMap;
    }

    out << YAML::EndMap;
}


void TreeGraph::CollectAssetRef(std::vector<AssetRef>& list)
{
}

struct GraphNode {
    int m_id;
    int m_parent;
    glm::vec3 m_endPosition;
    float m_radius;
};

void TreeGraph::Deserialize(const YAML::Node& in) {
    m_saved = true;
    m_name = in["name"].as<std::string>();
    m_layerSize = in["layersize"].as<int>();
    auto layers = in["layers"];
    auto rootLayer = layers["0"];
    std::unordered_map<int, std::shared_ptr<TreeGraphNode>> previousNodes;
    m_root = std::make_shared<TreeGraphNode>();
    m_root->m_start = glm::vec3(0, 0, 0);
    int rootIndex = 0;
    auto rootNode = rootLayer["0"];
    m_root->m_length = rootNode["length"].as<float>();
    m_root->m_thickness = rootNode["thickness"].as<float>();
    m_root->m_id = rootNode["id"].as<int>();
    m_root->m_parentId = -1;
    m_root->m_fromApicalBud = true;
    int index = 0;
    for (const auto& component : rootNode["quat"]) {
        m_root->m_globalRotation[index] = component.as<float>();
        index++;
    }
    index = 0;
    for (const auto& component : rootNode["position"]) {
        m_root->m_position[index] = component.as<float>();
        index++;
    }
    previousNodes[m_root->m_id] = m_root;
    for (int layerIndex = 1; layerIndex < m_layerSize; layerIndex++) {
        auto layer = layers[std::to_string(layerIndex)];
        auto internodeSize = layer["internodesize"].as<int>();
        for (int nodeIndex = 0; nodeIndex < internodeSize; nodeIndex++) {
            auto node = layer[std::to_string(nodeIndex)];
            auto parentNodeId = node["parent"].as<int>();
            if (parentNodeId == -1) parentNodeId = 0;
            auto& parentNode = previousNodes[parentNodeId];
            auto newNode = std::make_shared<TreeGraphNode>();
            newNode->m_id = node["id"].as<int>();
            newNode->m_start = parentNode->m_start + parentNode->m_length *
                (glm::normalize(parentNode->m_globalRotation) *
                    glm::vec3(0, 0, -1));
            newNode->m_thickness = node["thickness"].as<float>();
            newNode->m_length = node["length"].as<float>();
            newNode->m_parentId = parentNodeId;
            if (newNode->m_parentId == 0) newNode->m_parentId = -1;
            index = 0;
            for (const auto& component : node["quat"]) {
                newNode->m_globalRotation[index] = component.as<float>();
                index++;
            }
            index = 0;
            for (const auto& component : node["position"]) {
                newNode->m_position[index] = component.as<float>();
                index++;
            }
            if (parentNode->m_children.empty()) newNode->m_fromApicalBud = true;
            previousNodes[newNode->m_id] = newNode;
            parentNode->m_children.push_back(newNode);
            newNode->m_parent = parentNode;
        }
    }
}

void TreeGraph::OnInspect()
{
    ImGui::Checkbox("Length limit", &m_enableInstantiateLengthLimit);
    ImGui::DragFloat("Length limit", &m_instantiateLengthLimit, 0.1f);
}

void TreeGraph::CollectChild(const std::shared_ptr<TreeGraphNode>& node,
    std::vector<std::vector<std::shared_ptr<TreeGraphNode>>>& graphNodes,
    int currentLayer) const {
    graphNodes[currentLayer].push_back(node);
    for (const auto& i : node->m_children) {
        CollectChild(i, graphNodes, currentLayer + 1);
    }
}


void TreeGraphV2::Serialize(YAML::Emitter& out) {
}


void TreeGraphV2::CollectAssetRef(std::vector<AssetRef>& list)
{
}


void TreeGraphV2::Deserialize(const YAML::Node& in) {
    m_saved = true;
    m_name = GetTitle();
    int id = 0;
    std::vector<GraphNode> nodes;
    while (in[std::to_string(id)]) {
        auto& inNode = in[std::to_string(id)];
        nodes.emplace_back();
        auto& node = nodes.back();
        node.m_id = id;
        node.m_parent = inNode["parent"].as<int>();
        int index = 0;
        for (const auto& component : inNode["position"]) {
            node.m_endPosition[index] = component.as<float>();
            index++;
        }
        index = 0;
        node.m_radius = inNode["radius"].as<float>();
        id++;
    }
    std::unordered_map<int, std::shared_ptr<TreeGraphNode>> previousNodes;
    m_root = std::make_shared<TreeGraphNode>();
    m_root->m_start = glm::vec3(0, 0, 0);
    m_root->m_thickness = nodes[0].m_radius;
    m_root->m_id = 0;
    m_root->m_parentId = -1;
    m_root->m_fromApicalBud = true;
    m_root->m_position = nodes[0].m_endPosition;
    m_root->m_length = glm::length(nodes[0].m_endPosition);
    auto direction = glm::normalize(nodes[0].m_endPosition);
    m_root->m_globalRotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z,
        direction.x)
    );
    previousNodes[0] = m_root;
    for (id = 1; id < nodes.size(); id++) {
        auto& node = nodes[id];
        auto parentNodeId = node.m_parent;
        auto& parentNode = previousNodes[parentNodeId];
        auto newNode = std::make_shared<TreeGraphNode>();
        newNode->m_id = id;
        newNode->m_start = parentNode->m_position;
        newNode->m_thickness = node.m_radius;
        newNode->m_parentId = parentNodeId;
        newNode->m_position = node.m_endPosition;
        newNode->m_length = glm::length(newNode->m_position - newNode->m_start);
        auto direction = glm::normalize(newNode->m_position - newNode->m_start);
        newNode->m_globalRotation = glm::quatLookAt(direction, glm::vec3(direction.y, direction.z,
            direction.x)
        );
        if (parentNode->m_children.empty()) newNode->m_fromApicalBud = true;
        previousNodes[id] = newNode;
        parentNode->m_children.push_back(newNode);
        newNode->m_parent = parentNode;
    }
}

void TreeGraphV2::OnInspect()
{
    ImGui::Checkbox("Length limit", &m_enableInstantiateLengthLimit);
    ImGui::DragFloat("Length limit", &m_instantiateLengthLimit, 0.1f);
}

void TreeGraphV2::CollectChild(const std::shared_ptr<TreeGraphNode>& node,
    std::vector<std::vector<std::shared_ptr<TreeGraphNode>>>& graphNodes,
    int currentLayer) const {
    graphNodes[currentLayer].push_back(node);
    for (const auto& i : node->m_children) {
        CollectChild(i, graphNodes, currentLayer + 1);
    }
}
