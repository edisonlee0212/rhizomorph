//
// Created by lllll on 10/25/2022.
//

#include "Trees.hpp"
#include "Graphics.hpp"
#include "EditorLayer.hpp"
#include "Application.hpp"
#include "Climate.hpp"
#include "Tree.hpp"
#include "EcoSysLabLayer.hpp"
using namespace EcoSysLab;

void Trees::OnInspect() {
    static glm::ivec2 gridSize = {20, 20};
    static glm::vec2 gridDistance = {5, 5};
    static bool setParent = true;
    static bool enableHistory = true;
    static int historyIteration = 30;
    ImGui::Checkbox("Enable history", &enableHistory);
    if (enableHistory) ImGui::DragInt("History iteration", &historyIteration, 1, 1, 999);
    ImGui::DragInt2("Grid size", &gridSize.x, 1, 0, 100);
    ImGui::DragFloat2("Grid distance", &gridDistance.x, 0.1f, 0.0f, 100.0f);
    if (ImGui::Button("Reset Grid")) {
        m_globalTransforms.clear();
        const auto ecoSysLabLayer = Application::GetLayer<EcoSysLabLayer>();
        const auto soil = ecoSysLabLayer->m_soilHolder.Get<Soil>();
        const auto soilDescriptor = soil->m_soilDescriptor.Get<SoilDescriptor>();
        bool heightSet = false;
        if (soilDescriptor)
        {
	        if (const auto heightField = soilDescriptor->m_heightField.Get<HeightField>()) {
                heightSet = true;
                for (int i = 0; i < gridSize.x; i++) {
                    for (int j = 0; j < gridSize.y; j++) {
                        m_globalTransforms.emplace_back();
                        m_globalTransforms.back().SetPosition(glm::vec3(i * gridDistance.x, heightField->GetValue({ i * gridDistance.x, j * gridDistance.y }) - 0.05f, j * gridDistance.y));
                    }
                }
            }
        }
        if (!heightSet) {
            for (int i = 0; i < gridSize.x; i++) {
                for (int j = 0; j < gridSize.y; j++) {
                    m_globalTransforms.emplace_back();
                    m_globalTransforms.back().SetPosition(glm::vec3(i * gridDistance.x, 0.0f, j * gridDistance.y));
                }
            }
        }
    }
    Editor::DragAndDropButton<TreeDescriptor>(m_treeDescriptor, "TreeDescriptor", true);
    if (!m_globalTransforms.empty() && m_treeDescriptor.Get<TreeDescriptor>()) {
        auto &parameters = m_treeDescriptor.Get<TreeDescriptor>()->m_shootGrowthParameters;
        const auto ecoSysLabLayer = Application::GetLayer<EcoSysLabLayer>();
        ImGui::DragInt("Flow max node size", &m_treeGrowthSettings.m_flowNodeLimit, 1, 1, 100);
        ImGui::Checkbox("Auto balance vigor", &m_treeGrowthSettings.m_autoBalance);
        ImGui::Checkbox("Receive light", &m_treeGrowthSettings.m_collectLight);
        ImGui::Checkbox("Receive water", &m_treeGrowthSettings.m_collectWater);
        ImGui::Checkbox("Receive nitrite", &m_treeGrowthSettings.m_collectNitrite);
        ImGui::Checkbox("Enable Branch collision detection", &m_treeGrowthSettings.m_enableBranchCollisionDetection);
        ImGui::Checkbox("Enable Root collision detection", &m_treeGrowthSettings.m_enableRootCollisionDetection);

        if (ImGui::Button("Instantiate trees")) {
            auto scene = Application::GetActiveScene();
            Entity parent;
            if(setParent){
                parent = scene->CreateEntity("Trees (" + std::to_string(m_globalTransforms.size()) + ") - " + GetTitle());
            }
            int i = 0;
            for(const auto& gt : m_globalTransforms){
                auto treeEntity = scene->CreateEntity("Tree No." + std::to_string(i));
                i++;
                scene->SetDataComponent(treeEntity, gt);
                auto tree = scene->GetOrSetPrivateComponent<Tree>(treeEntity).lock();
                tree->m_treeModel.m_treeGrowthSettings = m_treeGrowthSettings;
                tree->m_treeDescriptor = m_treeDescriptor;
                tree->m_enableHistory = enableHistory;
                tree->m_historyIteration = historyIteration;
                tree->m_treeModel.RefShootSkeleton().m_data.m_treeIlluminationEstimator.m_settings = ecoSysLabLayer->m_shadowEstimationSettings;
                if(setParent) scene->SetParent(treeEntity, parent);
            }
        }
    }

    if (!m_globalTransforms.empty() && ImGui::Button("Clear")) {
        m_globalTransforms.clear();
    }
}

void Trees::OnCreate() {

}

void SerializeTreeGrowthSettings(const std::string& name, const TreeGrowthSettings& treeGrowthSettings, YAML::Emitter& out)
{
    out << YAML::Key << name << YAML::BeginMap;
    out << YAML::Key << "m_flowNodeLimit" << YAML::Value << treeGrowthSettings.m_flowNodeLimit;
    out << YAML::Key << "m_autoBalance" << YAML::Value << treeGrowthSettings.m_autoBalance;
    out << YAML::Key << "m_collectLight" << YAML::Value << treeGrowthSettings.m_collectLight;
    out << YAML::Key << "m_collectWater" << YAML::Value << treeGrowthSettings.m_collectWater;
    out << YAML::Key << "m_collectNitrite" << YAML::Value << treeGrowthSettings.m_collectNitrite;
    out << YAML::Key << "m_enableRootCollisionDetection" << YAML::Value << treeGrowthSettings.m_enableRootCollisionDetection;
    out << YAML::Key << "m_enableBranchCollisionDetection" << YAML::Value << treeGrowthSettings.m_enableBranchCollisionDetection;
    out << YAML::EndMap;
}
void DeserializeTreeGrowthSettings(const std::string& name, TreeGrowthSettings& treeGrowthSettings, const YAML::Node& in) {
    if (in[name]) {
        auto& param = in[name];
        if (param["m_flowNodeLimit"]) treeGrowthSettings.m_flowNodeLimit = param["m_flowNodeLimit"].as<int>();
        if (param["m_autoBalance"]) treeGrowthSettings.m_autoBalance = param["m_autoBalance"].as<bool>();
        if (param["m_collectLight"]) treeGrowthSettings.m_collectLight = param["m_collectLight"].as<bool>();
        if (param["m_collectWater"]) treeGrowthSettings.m_collectWater = param["m_collectWater"].as<bool>();
        if (param["m_collectNitrite"]) treeGrowthSettings.m_collectNitrite = param["m_collectNitrite"].as<bool>();
        if (param["m_enableRootCollisionDetection"]) treeGrowthSettings.m_enableRootCollisionDetection = param["m_enableRootCollisionDetection"].as<bool>();
        if (param["m_enableBranchCollisionDetection"]) treeGrowthSettings.m_enableBranchCollisionDetection = param["m_enableBranchCollisionDetection"].as<bool>();
    }
}


void Trees::CollectAssetRef(std::vector<AssetRef> &list) {
    list.push_back(m_treeDescriptor);
}

void Trees::Serialize(YAML::Emitter &out) {
    m_treeDescriptor.Save("m_treeDescriptor", out);

    SerializeTreeGrowthSettings("m_treeGrowthSettings", m_treeGrowthSettings, out);
}

void Trees::Deserialize(const YAML::Node &in) {
    m_treeDescriptor.Load("m_treeDescriptor", in);

    DeserializeTreeGrowthSettings("m_treeGrowthSettings", m_treeGrowthSettings, in);
}
