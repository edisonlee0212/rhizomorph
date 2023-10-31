// PlantFactory.cpp : This file contains the 'main' function. Program execution
// begins and ends there.
//
#include <Application.hpp>

#ifdef RAYTRACERFACILITY

#include <CUDAModule.hpp>
#include <RayTracerLayer.hpp>

#endif

#include "ProjectManager.hpp"
#include "PhysicsLayer.hpp"
#include "PostProcessing.hpp"
#include "ClassRegistry.hpp"
#include "TreeModel.hpp"
#include "Tree.hpp"
#include "Soil.hpp"
#include "Climate.hpp"
#include "Trees.hpp"
#include "EcoSysLabLayer.hpp"
#include "RadialBoundingVolume.hpp"
#include "HeightField.hpp"
#include "ObjectRotator.hpp"
#include "SorghumLayer.hpp"
#include "TreePointCloud.hpp"
#ifdef RAYTRACERFACILITY
using namespace RayTracerFacility;
#endif
using namespace EcoSysLab;

void EngineSetup();


int main() {
    ClassRegistry::RegisterPrivateComponent<Tree>("Tree");
    ClassRegistry::RegisterPrivateComponent<TreePointCloud>("TreePointCloud");
    ClassRegistry::RegisterPrivateComponent<Soil>("Soil");
    ClassRegistry::RegisterPrivateComponent<Climate>("Climate");
    ClassRegistry::RegisterPrivateComponent<ObjectRotator>("ObjectRotator");
    ClassRegistry::RegisterAsset<Trees>("Trees", {".trees"});
    ClassRegistry::RegisterAsset<TreeDescriptor>("TreeDescriptor", {".td"});
    ClassRegistry::RegisterAsset<SoilDescriptor>("SoilDescriptor", { ".sd" });
    ClassRegistry::RegisterAsset<ClimateDescriptor>("ClimateDescriptor", { ".cd" });
    ClassRegistry::RegisterAsset<RadialBoundingVolume>("RadialBoundingVolume", { ".rbv" });

    ClassRegistry::RegisterAsset<HeightField>("HeightField", { ".hf" });

    ClassRegistry::RegisterAsset<NoiseSoilLayerDescriptor>("NoiseSoilLayerDescriptor", { ".nsld" });




	EngineSetup();

    ApplicationConfigs applicationConfigs;
    applicationConfigs.m_applicationName = "EcoSysLab";
    Application::Create(applicationConfigs);

    Application::PushLayer<EcoSysLabLayer>();
    Application::PushLayer<SorghumLayer>();
#ifdef RAYTRACERFACILITY
    Application::PushLayer<RayTracerLayer>();
#endif

    // adjust default camera speed
    auto editorLayer = Application::GetLayer<EditorLayer>();
    editorLayer->m_velocity = 2.f;
    editorLayer->m_defaultSceneCameraPosition = glm::vec3(1.124, 0.218, 14.089);
    // override default scene camera position etc.
    editorLayer->m_showCameraWindow = false;
    editorLayer->m_showSceneWindow = true;
    editorLayer->m_showEntityExplorerWindow = true;
    editorLayer->m_showEntityInspectorWindow = true;

    auto consoleLayer = Application::GetLayer<ConsoleLayer>();
    consoleLayer->m_showConsoleWindow = true;

    auto rayTracerLayer = Application::GetLayer<RayTracerLayer>();
    rayTracerLayer->m_showCameraWindow = false;
    rayTracerLayer->m_showSceneWindow = false;
    rayTracerLayer->m_showRayTracerWindow = false;
#pragma region Engine Loop
    Application::Start();
#pragma endregion
    Application::End();
}

void EngineSetup() {
    ProjectManager::SetScenePostLoadActions([=]() {
        auto scene = Application::GetActiveScene();
#pragma region Engine Setup
        Transform transform;
        transform.SetEulerRotation(glm::radians(glm::vec3(150, 30, 0)));
#pragma region Preparations
        Application::Time().SetTimeStep(0.016f);
        transform = Transform();
        transform.SetPosition(glm::vec3(0, 2, 35));
        transform.SetEulerRotation(glm::radians(glm::vec3(15, 0, 0)));
        auto mainCamera = Application::GetActiveScene()->m_mainCamera.Get<UniEngine::Camera>();
        if (mainCamera) {
            auto postProcessing =
                    scene->GetOrSetPrivateComponent<PostProcessing>(mainCamera->GetOwner()).lock();
            auto ssao = postProcessing->GetLayer<SSAO>().lock();
            ssao->m_kernelRadius = 0.1;
            scene->SetDataComponent(mainCamera->GetOwner(), transform);
            mainCamera->m_useClearColor = true;
            mainCamera->m_clearColor = glm::vec3(0.5f);
        }
#pragma endregion
#pragma endregion

    });
}
