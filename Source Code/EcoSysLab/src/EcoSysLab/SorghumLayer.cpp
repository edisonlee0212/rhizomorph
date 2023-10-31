#ifdef RAYTRACERFACILITY
#include "BTFMeshRenderer.hpp"
#include "RayTracerLayer.hpp"
#include <TriangleIlluminationEstimator.hpp>
#endif
#include "ClassRegistry.hpp"
#include "DefaultResources.hpp"
#include "Graphics.hpp"
#include "LeafData.hpp"
#include "PanicleData.hpp"
#include "SkyIlluminance.hpp"
#include "StemData.hpp"
#include <SorghumData.hpp>
#include <SorghumLayer.hpp>
#ifdef RAYTRACERFACILITY
#include "CBTFGroup.hpp"
#include "DoubleCBTF.hpp"
#include "PARSensorGroup.hpp"
using namespace RayTracerFacility;
#endif
using namespace EcoSysLab;
using namespace UniEngine;

void SorghumLayer::OnCreate() {
  ClassRegistry::RegisterDataComponent<PanicleTag>("PanicleTag");
  ClassRegistry::RegisterDataComponent<StemTag>("StemTag");
  ClassRegistry::RegisterDataComponent<LeafTag>("LeafTag");
  ClassRegistry::RegisterDataComponent<SorghumTag>("SorghumTag");

  ClassRegistry::RegisterDataComponent<LeafGeometryTag>("LeafGeometryTag");
  ClassRegistry::RegisterDataComponent<LeafBottomFaceGeometryTag>(
      "LeafBottomFaceGeometryTag");
  ClassRegistry::RegisterDataComponent<PanicleGeometryTag>(
      "PanicleGeometryTag");
  ClassRegistry::RegisterDataComponent<StemGeometryTag>("StemGeometryTag");

  ClassRegistry::RegisterPrivateComponent<SorghumData>("SorghumData");
  ClassRegistry::RegisterPrivateComponent<LeafData>("LeafData");
  ClassRegistry::RegisterPrivateComponent<StemData>("StemData");
  ClassRegistry::RegisterPrivateComponent<PanicleData>("PanicleData");

  ClassRegistry::RegisterAsset<ProceduralSorghum>("ProceduralSorghum",
                                                  {".proceduralsorghum"});
  ClassRegistry::RegisterAsset<SorghumStateGenerator>(
      "SorghumStateGenerator", {".sorghumstategenerator"});
  ClassRegistry::RegisterAsset<SorghumField>("SorghumField", {".sorghumfield"});
#ifdef RAYTRACERFACILITY
  ClassRegistry::RegisterAsset<PARSensorGroup>("PARSensorGroup",
                                               {".parsensorgroup"});
  ClassRegistry::RegisterAsset<CBTFGroup>("CBTFGroup", {".cbtfg"});
  ClassRegistry::RegisterAsset<DoubleCBTF>("DoubleCBTF", {".dcbtf"});
#endif
  ClassRegistry::RegisterAsset<SkyIlluminance>("SkyIlluminance",
                                               {".skyilluminance"});
  ClassRegistry::RegisterAsset<RectangularSorghumField>(
      "RectangularSorghumField", {".rectsorghumfield"});
  ClassRegistry::RegisterAsset<PositionsField>("PositionsField",
                                               {".possorghumfield"});

  auto texture2D = std::make_shared<Texture2D>();
  texture2D->Import(std::filesystem::absolute(
      std::filesystem::path("./EcoSysLabResources/Textures") /
      "ProceduralSorghum.png"));
  Editor::AssetIcons()["ProceduralSorghum"] = texture2D;
  texture2D = std::make_shared<Texture2D>();
  texture2D->Import(std::filesystem::absolute(
      std::filesystem::path("./EcoSysLabResources/Textures") /
      "SorghumStateGenerator.png"));
  Editor::AssetIcons()["SorghumStateGenerator"] = texture2D;
  texture2D = std::make_shared<Texture2D>();
  texture2D->Import(std::filesystem::absolute(
      std::filesystem::path("./EcoSysLabResources/Textures") /
      "PositionsField.png"));
  Editor::AssetIcons()["PositionsField"] = texture2D;

  texture2D->Import(std::filesystem::absolute(
      std::filesystem::path("./EcoSysLabResources/Textures") /
      "GeneralDataPipeline.png"));
  Editor::AssetIcons()["GeneralDataCapture"] = texture2D;

  m_leafArchetype = Entities::CreateEntityArchetype("Leaf", LeafTag());
  m_leafQuery = Entities::CreateEntityQuery();
  m_leafQuery.SetAllFilters(LeafTag());

  m_stemArchetype = Entities::CreateEntityArchetype("Stem", StemTag());
  m_stemQuery = Entities::CreateEntityQuery();
  m_stemQuery.SetAllFilters(StemTag());

  m_panicleArchetype = Entities::CreateEntityArchetype("Panicle", PanicleTag());
  m_panicleQuery = Entities::CreateEntityQuery();
  m_panicleQuery.SetAllFilters(PanicleTag());

  m_sorghumArchetype = Entities::CreateEntityArchetype("Sorghum", SorghumTag());
  m_sorghumQuery = Entities::CreateEntityQuery();
  m_sorghumQuery.SetAllFilters(SorghumTag());

  m_leafGeometryArchetype =
      Entities::CreateEntityArchetype("Leaf Geometry", LeafGeometryTag());
  m_leafGeometryQuery = Entities::CreateEntityQuery();
  m_leafGeometryQuery.SetAllFilters(LeafGeometryTag());

  m_leafBottomFaceGeometryArchetype = Entities::CreateEntityArchetype(
      "Leaf Bottom Face Geometry", LeafBottomFaceGeometryTag());
  m_leafBottomFaceGeometryQuery = Entities::CreateEntityQuery();
  m_leafBottomFaceGeometryQuery.SetAllFilters(LeafBottomFaceGeometryTag());

  m_panicleGeometryArchetype =
      Entities::CreateEntityArchetype("Panicle Geometry", PanicleGeometryTag());
  m_panicleGeometryQuery = Entities::CreateEntityQuery();
  m_panicleGeometryQuery.SetAllFilters(PanicleGeometryTag());

  m_stemGeometryArchetype =
      Entities::CreateEntityArchetype("Stem Geometry", StemGeometryTag());
  m_stemGeometryQuery = Entities::CreateEntityQuery();
  m_stemGeometryQuery.SetAllFilters(StemGeometryTag());

  if (!m_leafAlbedoTexture.Get<Texture2D>()) {
    auto albedo = ProjectManager::CreateTemporaryAsset<Texture2D>();
    albedo->Import(std::filesystem::absolute(
        std::filesystem::path("./EcoSysLabResources/Textures") /
        "leafSurface.png"));
    m_leafAlbedoTexture.Set(albedo);
  }

  if (!m_leafMaterial.Get<Material>()) {
    auto material = ProjectManager::CreateTemporaryAsset<Material>();
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    m_leafMaterial = material;
    material->m_albedoTexture = m_leafAlbedoTexture;
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    material->m_drawSettings.m_cullFace = false;
    material->m_materialProperties.m_albedoColor =
        glm::vec3(113.0f / 255, 169.0f / 255, 44.0f / 255);
    material->m_materialProperties.m_roughness = 0.8f;
    material->m_materialProperties.m_metallic = 0.1f;
  }

  if (!m_leafBottomFaceMaterial.Get<Material>()) {
    auto material = ProjectManager::CreateTemporaryAsset<Material>();
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    m_leafBottomFaceMaterial = material;
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    material->m_drawSettings.m_cullFace = false;
    material->m_materialProperties.m_albedoColor =
        glm::vec3(113.0f / 255, 169.0f / 255, 44.0f / 255);
    material->m_materialProperties.m_roughness = 0.8f;
    material->m_materialProperties.m_metallic = 0.1f;
  }

  if (!m_panicleMaterial.Get<Material>()) {
    auto material = ProjectManager::CreateTemporaryAsset<Material>();
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    m_panicleMaterial = material;
    material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
    material->m_drawSettings.m_cullFaceMode = OpenGLCullFace::Back;
    material->m_materialProperties.m_albedoColor =
        glm::vec3(255.0 / 255, 210.0 / 255, 0.0 / 255);
    material->m_materialProperties.m_roughness = 0.5f;
    material->m_materialProperties.m_metallic = 0.0f;
  }

  for (auto &i : m_segmentedLeafMaterials) {
    if (!i.Get<Material>()) {
      auto material = ProjectManager::CreateTemporaryAsset<Material>();
      material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
      i = material;
      material->m_drawSettings.m_cullFace = false;
      material->m_materialProperties.m_albedoColor =
          glm::linearRand(glm::vec3(0.0f), glm::vec3(1.0f));
      material->m_materialProperties.m_roughness = 1.0f;
      material->m_materialProperties.m_metallic = 0.0f;
    }
  }
}

Entity SorghumLayer::CreateSorghum() {
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto scene = GetScene();
  const Entity entity = scene->CreateEntity(m_sorghumArchetype, "Sorghum");
  auto sorghumData =
      scene->GetOrSetPrivateComponent<SorghumData>(entity).lock();
  scene->SetEntityName(entity, "Sorghum");
#ifdef RAYTRACERFACILITY
  scene->GetOrSetPrivateComponent<TriangleIlluminationEstimator>(entity);
#endif
  return entity;
}
Entity SorghumLayer::CreateSorghumStem(const Entity &plantEntity) {
  auto scene = GetScene();
  const Entity entity = scene->CreateEntity(m_stemArchetype);
  scene->SetEntityName(entity, "Stem");
  scene->SetParent(entity, plantEntity);
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto stemData = scene->GetOrSetPrivateComponent<StemData>(entity).lock();

  StemTag tag;
  scene->SetDataComponent(entity, tag);
  scene->SetDataComponent(entity, transform);
  return entity;
}
Entity SorghumLayer::CreateSorghumLeaf(const Entity &plantEntity,
                                       int leafIndex) {
  auto scene = GetScene();
  const Entity entity = scene->CreateEntity(m_leafArchetype);
  scene->SetEntityName(entity, "Leaf");
  scene->SetParent(entity, plantEntity);
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto leafData = scene->GetOrSetPrivateComponent<LeafData>(entity).lock();
  leafData->m_index = leafIndex;
  scene->SetDataComponent(entity, transform);
  return entity;
}
Entity SorghumLayer::CreateSorghumPanicle(const Entity &plantEntity) {
  auto scene = GetScene();
  const Entity entity = scene->CreateEntity(m_panicleArchetype);
  scene->SetEntityName(entity, "Panicle");
  scene->SetParent(entity, plantEntity);
  Transform transform;
  transform.SetScale(glm::vec3(1.0f));
  auto panicleData =
      scene->GetOrSetPrivateComponent<PanicleData>(entity).lock();
  scene->SetDataComponent(entity, transform);
  return entity;
}

void SorghumLayer::GenerateMeshForAllSorghums() {
  std::vector<Entity> plants;
  auto scene = GetScene();
  scene->GetEntityArray(m_sorghumQuery, plants);
  for (auto &plant : plants) {
    if (scene->HasPrivateComponent<SorghumData>(plant)) {
      auto sorghumData =
          scene->GetOrSetPrivateComponent<SorghumData>(plant).lock();
      sorghumData->FormPlant();
      sorghumData->ApplyGeometry();
    }
  }
}

void SorghumLayer::OnInspect() {
  auto scene = GetScene();
  if (ImGui::Begin("Sorghum Layer")) {
#ifdef RAYTRACERFACILITY
    if (ImGui::TreeNodeEx("Illumination Estimation")) {
      ImGui::DragInt("Seed", &m_seed);
      ImGui::DragFloat("Push distance along normal", &m_pushDistance, 0.0001f,
                       -1.0f, 1.0f, "%.5f");
      m_rayProperties.OnInspect();

      if (ImGui::Button("Calculate illumination")) {
        CalculateIlluminationFrameByFrame();
      }
      if (ImGui::Button("Calculate illumination instantly")) {
        CalculateIllumination();
      }
      ImGui::TreePop();
    }
    Editor::DragAndDropButton<CBTFGroup>(m_leafCBTFGroup,
                                             "Leaf CBTFGroup");

    ImGui::Checkbox("Enable BTF", &m_enableCompressedBTF);
#endif
    ImGui::Separator();
    ImGui::Checkbox("Auto regenerate sorghum", &m_autoRefreshSorghums);
    ImGui::Checkbox("Bottom Face", &m_enableBottomFace);
    if (ImGui::Button("Generate mesh for all sorghums")) {
      GenerateMeshForAllSorghums();
    }
    if (ImGui::DragFloat("Vertical subdivision max unit length",
                         &m_verticalSubdivisionMaxUnitLength, 0.001f, 0.001f,
                         1.0f, "%.4f")) {
      m_verticalSubdivisionMaxUnitLength =
          glm::max(0.0001f, m_verticalSubdivisionMaxUnitLength);
    }

    if (ImGui::DragInt("Horizontal subdivision step",
                       &m_horizontalSubdivisionStep)) {
      m_horizontalSubdivisionStep = glm::max(2, m_horizontalSubdivisionStep);
    }

    if (ImGui::DragFloat("Skeleton width", &m_skeletonWidth, 0.001f, 0.001f,
                         1.0f, "%.4f")) {
      m_skeletonWidth = glm::max(0.0001f, m_skeletonWidth);
    }
    ImGui::ColorEdit3("Skeleton color", &m_skeletonColor.x);

    if (Editor::DragAndDropButton<Texture2D>(m_leafAlbedoTexture,
                                             "Replace Leaf Albedo Texture")) {
      auto tex = m_leafAlbedoTexture.Get<Texture2D>();
      if (tex) {
        m_leafMaterial.Get<Material>()->m_albedoTexture = m_leafAlbedoTexture;
        std::vector<Entity> sorghumEntities;
        scene->GetEntityArray(m_sorghumQuery, sorghumEntities, false);
        scene->GetEntityArray(m_leafQuery, sorghumEntities, false);
        scene->GetEntityArray(m_stemQuery, sorghumEntities, false);
        for (const auto &i : sorghumEntities) {
          if (scene->HasPrivateComponent<MeshRenderer>(i)) {
            scene->GetOrSetPrivateComponent<MeshRenderer>(i)
                .lock()
                ->m_material.Get<Material>()
                ->m_albedoTexture = m_leafAlbedoTexture;
          }
        }
      }
    }

    if (Editor::DragAndDropButton<Texture2D>(m_leafNormalTexture,
                                             "Replace Leaf Normal Texture")) {
      auto tex = m_leafNormalTexture.Get<Texture2D>();
      if (tex) {
        m_leafMaterial.Get<Material>()->m_normalTexture = m_leafNormalTexture;
        std::vector<Entity> sorghumEntities;
        scene->GetEntityArray(m_sorghumQuery, sorghumEntities, false);
        scene->GetEntityArray(m_leafQuery, sorghumEntities, false);
        scene->GetEntityArray(m_stemQuery, sorghumEntities, false);
        for (const auto &i : sorghumEntities) {
          if (scene->HasPrivateComponent<MeshRenderer>(i)) {
            scene->GetOrSetPrivateComponent<MeshRenderer>(i)
                .lock()
                ->m_material.Get<Material>()
                ->m_normalTexture = m_leafNormalTexture;
          }
        }
      }
    }

    FileUtils::SaveFile("Export OBJ for all sorghums", "3D Model", {".obj"},
                        [this](const std::filesystem::path &path) {
                          ExportAllSorghumsModel(path.string());
                        });

    static bool opened = false;
#ifdef RAYTRACERFACILITY
    if (m_processing && !opened) {
      ImGui::OpenPopup("Illumination Estimation");
      opened = true;
    }
    if (ImGui::BeginPopupModal("Illumination Estimation", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("Progress: ");
      float fraction = 1.0f - static_cast<float>(m_processingIndex) /
                                  m_processingEntities.size();
      std::string text =
          std::to_string(static_cast<int>(fraction * 100.0f)) + "% - " +
          std::to_string(m_processingEntities.size() - m_processingIndex) +
          "/" + std::to_string(m_processingEntities.size());
      ImGui::ProgressBar(fraction, ImVec2(240, 0), text.c_str());
      ImGui::SetItemDefaultFocus();
      ImGui::Text(("Estimation time for 1 plant: " +
                   std::to_string(m_perPlantCalculationTime) + " seconds")
                      .c_str());
      if (ImGui::Button("Cancel") || m_processing == false) {
        m_processing = false;
        opened = false;
        ImGui::CloseCurrentPopup();
      }
      ImGui::EndPopup();
    }
#endif
  }
  ImGui::End();
}

void SorghumLayer::ExportSorghum(const Entity &sorghum, std::ofstream &of,
                                 unsigned &startIndex) {
  auto scene = Application::GetActiveScene();
  const std::string start = "#Sorghum\n";
  of.write(start.c_str(), start.size());
  of.flush();
  const auto position =
      scene->GetDataComponent<GlobalTransform>(sorghum).GetPosition();

  const auto stemMesh = scene->GetOrSetPrivateComponent<MeshRenderer>(sorghum)
                            .lock()
                            ->m_mesh.Get<Mesh>();
  ObjExportHelper(position, stemMesh, of, startIndex);

  scene->ForEachChild(sorghum, [&](Entity child) {
    if (!scene->HasPrivateComponent<MeshRenderer>(child))
      return;
    const auto leafMesh = scene->GetOrSetPrivateComponent<MeshRenderer>(child)
                              .lock()
                              ->m_mesh.Get<Mesh>();
    ObjExportHelper(position, leafMesh, of, startIndex);
  });
}

void SorghumLayer::ObjExportHelper(glm::vec3 position,
                                   std::shared_ptr<Mesh> mesh,
                                   std::ofstream &of, unsigned &startIndex) {
  if (mesh && !mesh->UnsafeGetTriangles().empty()) {
    std::string header =
        "#Vertices: " + std::to_string(mesh->GetVerticesAmount()) +
        ", tris: " + std::to_string(mesh->GetTriangleAmount());
    header += "\n";
    of.write(header.c_str(), header.size());
    of.flush();
    std::string o = "o ";
    o += "[" + std::to_string(position.x) + "," + std::to_string(position.z) +
         "]" + "\n";
    of.write(o.c_str(), o.size());
    of.flush();
    std::string data;
#pragma region Data collection

    for (auto i = 0; i < mesh->UnsafeGetVertices().size(); i++) {
      auto &vertexPosition = mesh->UnsafeGetVertices().at(i).m_position;
      auto &color = mesh->UnsafeGetVertices().at(i).m_color;
      data += "v " + std::to_string(vertexPosition.x + position.x) + " " +
              std::to_string(vertexPosition.y + position.y) + " " +
              std::to_string(vertexPosition.z + position.z) + " " +
              std::to_string(color.x) + " " + std::to_string(color.y) + " " +
              std::to_string(color.z) + "\n";
    }
    for (const auto &vertex : mesh->UnsafeGetVertices()) {
      data += "vn " + std::to_string(vertex.m_normal.x) + " " +
              std::to_string(vertex.m_normal.y) + " " +
              std::to_string(vertex.m_normal.z) + "\n";
    }

    for (const auto &vertex : mesh->UnsafeGetVertices()) {
      data += "vt " + std::to_string(vertex.m_texCoord.x) + " " +
              std::to_string(vertex.m_texCoord.y) + "\n";
    }
    // data += "s off\n";
    data += "# List of indices for faces vertices, with (x, y, z).\n";
    auto &triangles = mesh->UnsafeGetTriangles();
    for (auto i = 0; i < mesh->GetTriangleAmount(); i++) {
      const auto triangle = triangles[i];
      const auto f1 = triangle.x + startIndex;
      const auto f2 = triangle.y + startIndex;
      const auto f3 = triangle.z + startIndex;
      data += "f " + std::to_string(f1) + "/" + std::to_string(f1) + "/" +
              std::to_string(f1) + " " + std::to_string(f2) + "/" +
              std::to_string(f2) + "/" + std::to_string(f2) + " " +
              std::to_string(f3) + "/" + std::to_string(f3) + "/" +
              std::to_string(f3) + "\n";
    }
    startIndex += mesh->GetVerticesAmount();
#pragma endregion
    of.write(data.c_str(), data.size());
    of.flush();
  }
}

void SorghumLayer::ExportAllSorghumsModel(const std::string &filename) {
  std::ofstream of;
  of.open(filename, std::ofstream::out | std::ofstream::trunc);
  if (of.is_open()) {
    std::string start = "#Sorghum field, by Bosheng Li";
    start += "\n";
    of.write(start.c_str(), start.size());
    of.flush();
    auto scene = GetScene();
    unsigned startIndex = 1;
    std::vector<Entity> sorghums;
    scene->GetEntityArray(m_sorghumQuery, sorghums);
    for (const auto &plant : sorghums) {
      ExportSorghum(plant, of, startIndex);
    }
    of.close();
    UNIENGINE_LOG("Sorghums saved as " + filename);
  } else {
    UNIENGINE_ERROR("Can't open file!");
  }
}


#ifdef RAYTRACERFACILITY
void SorghumLayer::CalculateIlluminationFrameByFrame() {
  auto scene = GetScene();
  const auto *owners = scene->UnsafeGetPrivateComponentOwnersList<
      TriangleIlluminationEstimator>();
  if (!owners)
    return;
  m_processingEntities.clear();

  m_processingEntities.insert(m_processingEntities.begin(), owners->begin(),
                              owners->end());
  m_processingIndex = m_processingEntities.size();
  m_processing = true;
}
void SorghumLayer::CalculateIllumination() {
  auto scene = GetScene();
  const auto *owners = scene->UnsafeGetPrivateComponentOwnersList<
      TriangleIlluminationEstimator>();
  if (!owners)
    return;
  m_processingEntities.clear();

  m_processingEntities.insert(m_processingEntities.begin(), owners->begin(),
                              owners->end());
  m_processingIndex = m_processingEntities.size();
  while (m_processing) {
    m_processingIndex--;
    if (m_processingIndex == -1) {
      m_processing = false;
    } else {
      const float timer = Application::Time().CurrentTime();
      auto estimator =
          scene
              ->GetOrSetPrivateComponent<TriangleIlluminationEstimator>(
                  m_processingEntities[m_processingIndex])
              .lock();
			estimator->PrepareLightProbeGroup();
      estimator->SampleLightProbeGroup(m_rayProperties, m_seed,
                                                     m_pushDistance);
    }
  }
}
#endif
void SorghumLayer::Update() {
  auto scene = GetScene();
#ifdef RAYTRACERFACILITY
  if (m_processing) {
    m_processingIndex--;
    if (m_processingIndex == -1) {
      m_processing = false;
    } else {
      const float timer = Application::Time().CurrentTime();
      auto estimator =
          scene
              ->GetOrSetPrivateComponent<TriangleIlluminationEstimator>(
                  m_processingEntities[m_processingIndex])
              .lock();
				estimator->PrepareLightProbeGroup();
				estimator->SampleLightProbeGroup(m_rayProperties, m_seed,
																				 m_pushDistance);
      m_perPlantCalculationTime = Application::Time().CurrentTime() - timer;
    }
  }
#endif
}

Entity SorghumLayer::CreateSorghum(
    const std::shared_ptr<ProceduralSorghum> &descriptor) {
  if (!descriptor) {
    UNIENGINE_ERROR("ProceduralSorghum empty!");
    return {};
  }
  auto scene = GetScene();
  Entity sorghum = CreateSorghum();
  auto sorghumData =
      scene->GetOrSetPrivateComponent<SorghumData>(sorghum).lock();
  sorghumData->m_mode = (int)SorghumMode::ProceduralSorghum;
  sorghumData->m_descriptor = descriptor;
  sorghumData->SetTime(1.0f);
  sorghumData->FormPlant();
  sorghumData->ApplyGeometry();
  return sorghum;
}
Entity SorghumLayer::CreateSorghum(
    const std::shared_ptr<SorghumStateGenerator> &descriptor) {
  if (!descriptor) {
    UNIENGINE_ERROR("SorghumStateGenerator empty!");
    return {};
  }
  auto scene = GetScene();
  Entity sorghum = CreateSorghum();
  auto sorghumData =
      scene->GetOrSetPrivateComponent<SorghumData>(sorghum).lock();
  sorghumData->m_mode = (int)SorghumMode::SorghumStateGenerator;
  sorghumData->m_descriptor = descriptor;
  sorghumData->m_bottomFace = m_enableBottomFace;
  sorghumData->SetTime(1.0f);
  sorghumData->FormPlant();
  sorghumData->ApplyGeometry();
  return sorghum;
}

void SorghumLayer::LateUpdate() {
  if (m_autoRefreshSorghums) {
    auto scene = GetScene();
    std::vector<Entity> plants;
    scene->GetEntityArray(m_sorghumQuery, plants);
    for (auto &plant : plants) {
      if (scene->HasPrivateComponent<SorghumData>(plant)) {
        auto sorghumData =
            scene->GetOrSetPrivateComponent<SorghumData>(plant).lock();
        auto proceduralSorghum =
            sorghumData->m_descriptor.Get<ProceduralSorghum>();
        if (proceduralSorghum &&
            proceduralSorghum->GetVersion() != sorghumData->m_recordedVersion) {
          sorghumData->FormPlant();
          sorghumData->ApplyGeometry();
          continue;
        }
        auto sorghumStateGenerator =
            sorghumData->m_descriptor.Get<SorghumStateGenerator>();
        if (sorghumStateGenerator && sorghumStateGenerator->GetVersion() !=
                                         sorghumData->m_recordedVersion) {
          sorghumData->FormPlant();
          sorghumData->ApplyGeometry();
        }
      }
    }
  }
}
