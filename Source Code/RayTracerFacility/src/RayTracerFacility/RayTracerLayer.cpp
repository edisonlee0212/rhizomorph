#include "BTFMeshRenderer.hpp"
#include <RayTracerLayer.hpp>
#include <ProjectManager.hpp>
#include <RayTracer.hpp>
#include "Windows.hpp"
#include "EditorLayer.hpp"
#include "RayTracerCamera.hpp"
#include "TriangleIlluminationEstimator.hpp"
#include "PointCloudScanner.hpp"
#include "ClassRegistry.hpp"
#include "StrandsRenderer.hpp"
#include "CompressedBTF.hpp"

using namespace RayTracerFacility;

std::shared_ptr<RayTracerCamera> RayTracerLayer::m_rayTracerCamera;

void RayTracerLayer::UpdateMeshesStorage(std::unordered_map<uint64_t, RayTracedMaterial>& materialStorage,
	std::unordered_map<uint64_t, RayTracedGeometry>& geometryStorage,
	std::unordered_map<uint64_t, RayTracedInstance>& instanceStorage,
	bool& rebuildInstances, bool& updateShaderBindingTable) const {
	for (auto& i : instanceStorage) i.second.m_removeFlag = true;
	for (auto& i : geometryStorage) i.second.m_removeFlag = true;
	for (auto& i : materialStorage) i.second.m_removeFlag = true;
	auto scene = GetScene();
	if (const auto* rayTracedEntities =
		scene->UnsafeGetPrivateComponentOwnersList<StrandsRenderer>();
		rayTracedEntities && m_renderStrandsRenderer) {
		for (auto entity : *rayTracedEntities) {
			if (!scene->IsEntityEnabled(entity))
				continue;
			auto strandsRendererRenderer =
				scene->GetOrSetPrivateComponent<StrandsRenderer>(entity).lock();
			if (!strandsRendererRenderer->IsEnabled())
				continue;
			auto strands = strandsRendererRenderer->m_strands.Get<Strands>();
			auto material = strandsRendererRenderer->m_material.Get<Material>();
			if (!material || !strands || strands->UnsafeGetPoints().empty() || strands->UnsafeGetSegments().empty())
				continue;
			auto globalTransform = scene->GetDataComponent<GlobalTransform>(entity).m_value;
			bool needInstanceUpdate = false;
			bool needMaterialUpdate = false;

			auto entityHandle = scene->GetEntityHandle(entity);
			auto geometryHandle = strands->GetHandle();
			auto materialHandle = material->GetHandle();
			auto& rayTracedInstance = instanceStorage[strandsRendererRenderer->GetHandle().GetValue()];
			auto& rayTracedGeometry = geometryStorage[geometryHandle];
			auto& rayTracedMaterial = materialStorage[materialHandle];
			rayTracedInstance.m_removeFlag = false;
			rayTracedMaterial.m_removeFlag = false;
			rayTracedGeometry.m_removeFlag = false;

			if (rayTracedInstance.m_entityHandle != entityHandle
				|| rayTracedInstance.m_privateComponentHandle != strandsRendererRenderer->GetHandle().GetValue()
				|| rayTracedInstance.m_version != strandsRendererRenderer->GetVersion()
				|| globalTransform != rayTracedInstance.m_globalTransform) {
				needInstanceUpdate = true;
			}
			if (rayTracedGeometry.m_handle == 0 || rayTracedGeometry.m_version != strands->GetVersion()) {
				rayTracedGeometry.m_updateFlag = true;
				needInstanceUpdate = true;
				rayTracedGeometry.m_rendererType = RendererType::Curve;
				rayTracedGeometry.m_curveSegments = &strands->UnsafeGetSegments();
				rayTracedGeometry.m_curvePoints = &strands->UnsafeGetPoints();
				//rayTracedGeometry.m_strandU = &strands->UnsafeGetStrandU();
				//rayTracedGeometry.m_strandIndices = &strands->UnsafeGetStrandIndices();
				//rayTracedGeometry.m_strandInfos = &strands->UnsafeGetStrandInfos();

				rayTracedGeometry.m_version = strands->GetVersion();
				switch (strands->GetSplineMode()) {
				case Strands::SplineMode::Linear:
					rayTracedGeometry.m_geometryType = GeometryType::Linear;
					break;
				case Strands::SplineMode::Quadratic:
					rayTracedGeometry.m_geometryType = GeometryType::QuadraticBSpline;
					break;
				case Strands::SplineMode::Cubic:
					rayTracedGeometry.m_geometryType = GeometryType::CubicBSpline;
					break;
				}
				rayTracedGeometry.m_handle = geometryHandle;
			}
			if (CheckMaterial(rayTracedMaterial, material)) needInstanceUpdate = true;
			if (needInstanceUpdate) {
				rayTracedInstance.m_entityHandle = entityHandle;
				rayTracedInstance.m_privateComponentHandle = strandsRendererRenderer->GetHandle().GetValue();
				rayTracedInstance.m_version = strandsRendererRenderer->GetVersion();
				rayTracedInstance.m_globalTransform = globalTransform;
				rayTracedInstance.m_geometryMapKey = geometryHandle;
				rayTracedInstance.m_materialMapKey = materialHandle;
			}
			updateShaderBindingTable = updateShaderBindingTable || needMaterialUpdate;
			rebuildInstances = rebuildInstances || needInstanceUpdate;
		}
	}
	if (const auto* rayTracedEntities =
		scene->UnsafeGetPrivateComponentOwnersList<MeshRenderer>();
		rayTracedEntities && m_renderMeshRenderer) {
		for (auto entity : *rayTracedEntities) {
			if (!scene->IsEntityEnabled(entity))
				continue;
			auto meshRenderer =
				scene->GetOrSetPrivateComponent<MeshRenderer>(entity).lock();
			if (!meshRenderer->IsEnabled())
				continue;
			auto mesh = meshRenderer->m_mesh.Get<Mesh>();
			auto material = meshRenderer->m_material.Get<Material>();
			if (!material || !mesh || mesh->UnsafeGetVertices().empty())
				continue;
			auto globalTransform = scene->GetDataComponent<GlobalTransform>(entity).m_value;
			bool needInstanceUpdate = false;
			bool needMaterialUpdate = false;

			auto entityHandle = scene->GetEntityHandle(entity);
			auto geometryHandle = mesh->GetHandle();
			auto materialHandle = material->GetHandle();
			auto& rayTracedInstance = instanceStorage[meshRenderer->GetHandle().GetValue()];
			auto& rayTracedGeometry = geometryStorage[geometryHandle];
			auto& rayTracedMaterial = materialStorage[materialHandle];
			rayTracedInstance.m_removeFlag = false;
			rayTracedMaterial.m_removeFlag = false;
			rayTracedGeometry.m_removeFlag = false;

			if (rayTracedInstance.m_entityHandle != entityHandle
				|| rayTracedInstance.m_privateComponentHandle != meshRenderer->GetHandle().GetValue()
				|| rayTracedInstance.m_version != meshRenderer->GetVersion()
				|| globalTransform != rayTracedInstance.m_globalTransform) {
				needInstanceUpdate = true;
			}
			if (rayTracedGeometry.m_handle == 0 || rayTracedGeometry.m_version != mesh->GetVersion()) {
				rayTracedGeometry.m_updateFlag = true;
				needInstanceUpdate = true;
				rayTracedGeometry.m_rendererType = RendererType::Default;
				rayTracedGeometry.m_triangles = &mesh->UnsafeGetTriangles();
				rayTracedGeometry.m_vertices = &mesh->UnsafeGetVertices();
				rayTracedGeometry.m_version = mesh->GetVersion();
				rayTracedGeometry.m_geometryType = GeometryType::Triangle;
				rayTracedGeometry.m_handle = geometryHandle;
			}
			if (CheckMaterial(rayTracedMaterial, material)) needInstanceUpdate = true;
			if (needInstanceUpdate) {
				rayTracedInstance.m_entityHandle = entityHandle;
				rayTracedInstance.m_privateComponentHandle = meshRenderer->GetHandle().GetValue();
				rayTracedInstance.m_version = meshRenderer->GetVersion();
				rayTracedInstance.m_globalTransform = globalTransform;
				rayTracedInstance.m_geometryMapKey = geometryHandle;
				rayTracedInstance.m_materialMapKey = materialHandle;
			}
			updateShaderBindingTable = updateShaderBindingTable || needMaterialUpdate;
			rebuildInstances = rebuildInstances || needInstanceUpdate;
		}
	}
	if (const auto* rayTracedEntities =
		scene->UnsafeGetPrivateComponentOwnersList<SkinnedMeshRenderer>();
		rayTracedEntities && m_renderSkinnedMeshRenderer) {
		for (auto entity : *rayTracedEntities) {
			if (!scene->IsEntityEnabled(entity))
				continue;
			auto skinnedMeshRenderer =
				scene->GetOrSetPrivateComponent<SkinnedMeshRenderer>(entity).lock();
			if (!skinnedMeshRenderer->IsEnabled())
				continue;
			auto mesh = skinnedMeshRenderer->m_skinnedMesh.Get<SkinnedMesh>();
			auto material = skinnedMeshRenderer->m_material.Get<Material>();
			if (!material || !mesh || mesh->UnsafeGetSkinnedVertices().empty() ||
				skinnedMeshRenderer->m_finalResults->m_value.empty())
				continue;
			auto globalTransform =
				skinnedMeshRenderer->RagDoll()
				? glm::mat4(1.0f)
				: scene->GetDataComponent<GlobalTransform>(entity).m_value;
			bool needInstanceUpdate = false;
			bool needMaterialUpdate = false;

			auto entityHandle = scene->GetEntityHandle(entity);
			auto geometryHandle = skinnedMeshRenderer->GetHandle().GetValue();
			auto materialHandle = material->GetHandle();
			auto& rayTracedInstance = instanceStorage[geometryHandle];
			auto& rayTracedGeometry = geometryStorage[geometryHandle];
			auto& rayTracedMaterial = materialStorage[materialHandle];
			rayTracedInstance.m_removeFlag = false;
			rayTracedMaterial.m_removeFlag = false;
			rayTracedGeometry.m_removeFlag = false;

			if (rayTracedInstance.m_entityHandle != entityHandle
				|| rayTracedInstance.m_privateComponentHandle != skinnedMeshRenderer->GetHandle().GetValue()
				|| rayTracedInstance.m_version != skinnedMeshRenderer->GetVersion()
				|| globalTransform != rayTracedInstance.m_globalTransform) {
				needInstanceUpdate = true;
			}

			if (rayTracedGeometry.m_handle == 0
				|| rayTracedInstance.m_version != skinnedMeshRenderer->GetVersion()
				|| rayTracedGeometry.m_version != mesh->GetVersion()
				|| rayTracedInstance.m_dataVersion != skinnedMeshRenderer->m_finalResults->GetVersion()
				|| skinnedMeshRenderer->m_animator.Get<Animator>()->AnimatedCurrentFrame()) {
				rayTracedGeometry.m_updateFlag = true;
				needInstanceUpdate = true;
				rayTracedGeometry.m_geometryType = GeometryType::Triangle;
				rayTracedGeometry.m_rendererType = RendererType::Skinned;
				rayTracedGeometry.m_triangles = &mesh->UnsafeGetTriangles();
				rayTracedGeometry.m_skinnedVertices = &mesh->UnsafeGetSkinnedVertices();
				rayTracedGeometry.m_boneMatrices =
					&skinnedMeshRenderer->m_finalResults->m_value;
				rayTracedGeometry.m_version = mesh->GetVersion();
				rayTracedInstance.m_dataVersion = skinnedMeshRenderer->m_finalResults->GetVersion();
				rayTracedGeometry.m_handle = geometryHandle;
			}
			if (CheckMaterial(rayTracedMaterial, material)) needInstanceUpdate = true;
			if (needInstanceUpdate) {
				rayTracedInstance.m_entityHandle = entityHandle;
				rayTracedInstance.m_privateComponentHandle = skinnedMeshRenderer->GetHandle().GetValue();
				rayTracedInstance.m_version = skinnedMeshRenderer->GetVersion();
				rayTracedInstance.m_globalTransform = globalTransform;
				rayTracedInstance.m_geometryMapKey = geometryHandle;
				rayTracedInstance.m_materialMapKey = materialHandle;
			}
			updateShaderBindingTable = updateShaderBindingTable || needMaterialUpdate;
			rebuildInstances = rebuildInstances || needInstanceUpdate;
		}
	}
	if (const auto* rayTracedEntities =
		scene->UnsafeGetPrivateComponentOwnersList<Particles>();
		rayTracedEntities && m_renderParticles) {
		for (auto entity : *rayTracedEntities) {
			if (!scene->IsEntityEnabled(entity))
				continue;
			auto particles =
				scene->GetOrSetPrivateComponent<Particles>(entity).lock();
			if (!particles->IsEnabled())
				continue;
			auto mesh = particles->m_mesh.Get<Mesh>();
			auto material = particles->m_material.Get<Material>();
			auto matrices = particles->m_matrices;
			if (!material || !mesh || mesh->UnsafeGetVertices().empty() || matrices->RefMatrices().empty())
				continue;
			auto globalTransform = scene->GetDataComponent<GlobalTransform>(entity).m_value;
			bool needInstanceUpdate = false;
			bool needMaterialUpdate = false;

			auto entityHandle = scene->GetEntityHandle(entity);
			auto geometryHandle = particles->GetHandle().GetValue();
			auto materialHandle = material->GetHandle();
			auto& rayTracedInstance = instanceStorage[geometryHandle];
			auto& rayTracedGeometry = geometryStorage[geometryHandle];
			auto& rayTracedMaterial = materialStorage[materialHandle];
			rayTracedInstance.m_removeFlag = false;
			rayTracedMaterial.m_removeFlag = false;
			rayTracedGeometry.m_removeFlag = false;

			if (rayTracedInstance.m_entityHandle != entityHandle
				|| rayTracedInstance.m_privateComponentHandle != particles->GetHandle().GetValue()
				|| rayTracedInstance.m_version != particles->GetVersion()
				|| rayTracedInstance.m_dataVersion != particles->m_matrices->GetVersion()
				|| globalTransform != rayTracedInstance.m_globalTransform) {
				needInstanceUpdate = true;
			}
			if (needInstanceUpdate || rayTracedGeometry.m_handle == 0
				|| rayTracedInstance.m_version != particles->GetVersion()
				|| rayTracedGeometry.m_version != mesh->GetVersion()) {
				rayTracedGeometry.m_updateFlag = true;
				needInstanceUpdate = true;
				rayTracedGeometry.m_geometryType = GeometryType::Triangle;
				rayTracedGeometry.m_rendererType = RendererType::Instanced;
				rayTracedGeometry.m_triangles = &mesh->UnsafeGetTriangles();
				rayTracedGeometry.m_vertices = &mesh->UnsafeGetVertices();
				rayTracedGeometry.m_instanceMatrices = &matrices->RefMatrices();
				rayTracedGeometry.m_instanceColors = &matrices->RefColors();
				rayTracedGeometry.m_version = mesh->GetVersion();
				rayTracedGeometry.m_handle = geometryHandle;
				rayTracedInstance.m_dataVersion = particles->m_matrices->GetVersion();
			}
			if (CheckMaterial(rayTracedMaterial, material)) needInstanceUpdate = true;
			if (needInstanceUpdate) {
				rayTracedInstance.m_entityHandle = entityHandle;
				rayTracedInstance.m_privateComponentHandle = particles->GetHandle().GetValue();
				rayTracedInstance.m_version = particles->GetVersion();
				rayTracedInstance.m_globalTransform = globalTransform;
				rayTracedInstance.m_geometryMapKey = geometryHandle;
				rayTracedInstance.m_materialMapKey = materialHandle;
			}
			updateShaderBindingTable = updateShaderBindingTable || needMaterialUpdate;
			rebuildInstances = rebuildInstances || needInstanceUpdate;
		}
	}
	if (const auto* rayTracedEntities =
		scene->UnsafeGetPrivateComponentOwnersList<BTFMeshRenderer>();
		rayTracedEntities && m_renderBTFMeshRenderer) {
		for (auto entity : *rayTracedEntities) {
			if (!scene->IsEntityEnabled(entity))
				continue;
			auto meshRenderer =
				scene->GetOrSetPrivateComponent<BTFMeshRenderer>(entity).lock();
			if (!meshRenderer->IsEnabled())
				continue;
			auto mesh = meshRenderer->m_mesh.Get<Mesh>();
			auto material = meshRenderer->m_btf.Get<CompressedBTF>();
			if (!material || !material->m_bTFBase.m_hasData || !mesh || mesh->UnsafeGetVertices().empty())
				continue;
			auto globalTransform = scene->GetDataComponent<GlobalTransform>(entity).m_value;
			bool needInstanceUpdate = false;
			bool needMaterialUpdate = false;

			auto entityHandle = scene->GetEntityHandle(entity);
			auto geometryHandle = mesh->GetHandle();
			auto materialHandle = material->GetHandle();
			auto& rayTracedInstance = instanceStorage[meshRenderer->GetHandle().GetValue()];
			auto& rayTracedGeometry = geometryStorage[geometryHandle];
			auto& rayTracedMaterial = materialStorage[materialHandle];
			rayTracedInstance.m_removeFlag = false;
			rayTracedMaterial.m_removeFlag = false;
			rayTracedGeometry.m_removeFlag = false;

			if (rayTracedInstance.m_entityHandle != entityHandle
				|| rayTracedInstance.m_privateComponentHandle != meshRenderer->GetHandle().GetValue()
				|| rayTracedInstance.m_version != meshRenderer->GetVersion()
				|| globalTransform != rayTracedInstance.m_globalTransform) {
				needInstanceUpdate = true;
			}
			if (rayTracedGeometry.m_handle == 0 || rayTracedGeometry.m_version != mesh->GetVersion()) {
				rayTracedGeometry.m_updateFlag = true;
				needInstanceUpdate = true;
				rayTracedGeometry.m_rendererType = RendererType::Default;
				rayTracedGeometry.m_triangles = &mesh->UnsafeGetTriangles();
				rayTracedGeometry.m_vertices = &mesh->UnsafeGetVertices();
				rayTracedGeometry.m_version = mesh->GetVersion();
				rayTracedGeometry.m_geometryType = GeometryType::Triangle;
				rayTracedGeometry.m_handle = geometryHandle;
			}
			if (CheckCompressedBTF(rayTracedMaterial, material)) needInstanceUpdate = true;
			if (needInstanceUpdate) {
				rayTracedInstance.m_entityHandle = entityHandle;
				rayTracedInstance.m_privateComponentHandle = meshRenderer->GetHandle().GetValue();
				rayTracedInstance.m_version = meshRenderer->GetVersion();
				rayTracedInstance.m_globalTransform = globalTransform;
				rayTracedInstance.m_geometryMapKey = geometryHandle;
				rayTracedInstance.m_materialMapKey = materialHandle;
			}
			updateShaderBindingTable = updateShaderBindingTable || needMaterialUpdate;
			rebuildInstances = rebuildInstances || needInstanceUpdate;
		}
	}

	for (auto& i : instanceStorage) if (i.second.m_removeFlag) rebuildInstances = true;
}

void RayTracerLayer::UpdateScene() {
	bool rebuildAccelerationStructure = false;
	bool updateShaderBindingTable = false;
	auto& instanceStorage = CudaModule::GetRayTracer()->m_instances;
	auto& materialStorage = CudaModule::GetRayTracer()->m_materials;
	auto& geometryStorage = CudaModule::GetRayTracer()->m_geometries;
	UpdateMeshesStorage(materialStorage, geometryStorage, instanceStorage, rebuildAccelerationStructure,
		updateShaderBindingTable);
	unsigned int envMapId = 0;
	auto& envSettings = GetScene()->m_environmentSettings;
	if (envSettings.m_environmentType == UniEngine::EnvironmentType::EnvironmentalMap) {
		auto environmentalMap = envSettings.m_environmentalMap.Get<EnvironmentalMap>();
		if (environmentalMap) {
			auto cubeMap = environmentalMap->GetCubemap().Get<Cubemap>();
			if (cubeMap) envMapId = cubeMap->Texture()->Id();
		}
	}
	else if (envSettings.m_backgroundColor != m_environmentProperties.m_color) {
		m_environmentProperties.m_color = envSettings.m_backgroundColor;
		updateShaderBindingTable = true;
	}
	if (m_environmentProperties.m_skylightIntensity != envSettings.m_ambientLightIntensity) {
		m_environmentProperties.m_skylightIntensity = envSettings.m_ambientLightIntensity;
		updateShaderBindingTable = true;
	}
	if (m_environmentProperties.m_gamma != envSettings.m_environmentGamma) {
		m_environmentProperties.m_gamma = envSettings.m_environmentGamma;
		updateShaderBindingTable = true;
	}
	if (m_environmentProperties.m_environmentalMapId != envMapId) {
		m_environmentProperties.m_environmentalMapId = envMapId;
		updateShaderBindingTable = true;
	}

	CudaModule::GetRayTracer()->m_requireUpdate = false;
	if (rebuildAccelerationStructure &&
		(!instanceStorage.empty())) {
		CudaModule::GetRayTracer()->BuildIAS();
		CudaModule::GetRayTracer()->m_requireUpdate = true;
	}
	else if (updateShaderBindingTable) {
		CudaModule::GetRayTracer()->m_requireUpdate = true;
	}
}

void RayTracerLayer::OnCreate() {
	CudaModule::Init();
	ClassRegistry::RegisterPrivateComponent<BTFMeshRenderer>(
		"BTFMeshRenderer");
	ClassRegistry::RegisterPrivateComponent<TriangleIlluminationEstimator>(
		"TriangleIlluminationEstimator");
	ClassRegistry::RegisterPrivateComponent<RayTracerCamera>(
		"RayTracerCamera");
	ClassRegistry::RegisterPrivateComponent<PointCloudScanner>(
		"PointCloudScanner");
	ClassRegistry::RegisterAsset<CompressedBTF>(
		"CompressedBTF", { ".cbtf" });

	m_sceneCamera = Serialization::ProduceSerializable<RayTracerCamera>();
	m_sceneCamera->OnCreate();

	Application::RegisterPostAttachSceneFunction([&](const std::shared_ptr<Scene>& scene) {
		m_rayTracerCamera.reset();
		});
}


void RayTracerLayer::LateUpdate() {
	UpdateScene();
	if (!CudaModule::GetRayTracer()->m_instances.empty()) {
		auto editorLayer = Application::GetLayer<EditorLayer>();
		if (m_showSceneWindow && editorLayer && m_renderingEnabled) {
			m_sceneCamera->Ready(editorLayer->m_sceneCameraPosition, editorLayer->m_sceneCameraRotation);
			m_sceneCamera->m_rendered = CudaModule::GetRayTracer()->RenderToCamera(m_environmentProperties,
				m_sceneCamera->m_cameraProperties,
				m_sceneCamera->m_rayProperties);
		}
		auto scene = GetScene();
		auto* entities = scene->UnsafeGetPrivateComponentOwnersList<RayTracerCamera>();
		m_rayTracerCamera.reset();
		if (entities) {
			bool check = false;
			for (const auto& entity : *entities) {
				if (!scene->IsEntityEnabled(entity)) continue;
				auto rayTracerCamera = scene->GetOrSetPrivateComponent<RayTracerCamera>(entity).lock();
				if (!rayTracerCamera->IsEnabled()) continue;
				auto globalTransform = scene->GetDataComponent<GlobalTransform>(rayTracerCamera->GetOwner()).m_value;
				rayTracerCamera->Ready(globalTransform[3], glm::quat_cast(globalTransform));
				rayTracerCamera->m_rendered = CudaModule::GetRayTracer()->RenderToCamera(m_environmentProperties,
					rayTracerCamera->m_cameraProperties,
					rayTracerCamera->m_rayProperties);
				if (!check) {
					if (rayTracerCamera->m_mainCamera) {
						m_rayTracerCamera = rayTracerCamera;
						check = true;
					}
				}
				else {
					rayTracerCamera->m_mainCamera = false;
				}
			}
		}
	}

}

void RayTracerLayer::OnInspect() {
	if (ImGui::BeginMainMenuBar()) {
		if (ImGui::BeginMenu("View")) {
			ImGui::Checkbox("Ray Tracer", &m_showRayTracerWindow);
			ImGui::Checkbox("Scene (Ray)", &m_showSceneWindow);
			ImGui::Checkbox("Camera (Ray)", &m_showCameraWindow);
			ImGui::EndMenu();
		}
		ImGui::EndMainMenuBar();
	}
	if (m_showRayTracerWindow) {
		if (ImGui::Begin("Ray Tracer")) {
			ImGui::Checkbox("Mesh Renderer", &m_renderMeshRenderer);
			ImGui::Checkbox("Strand Renderer", &m_renderStrandsRenderer);
			ImGui::Checkbox("Particles", &m_renderParticles);
			ImGui::Checkbox("Skinned Mesh Renderer", &m_renderSkinnedMeshRenderer);
			ImGui::Checkbox("BTF Mesh Renderer", &m_renderBTFMeshRenderer);

			if (ImGui::TreeNode("Scene Camera Settings")) {
				m_sceneCamera->OnInspect();
				ImGui::TreePop();
			}
			if (ImGui::TreeNodeEx("Environment Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
				m_environmentProperties.OnInspect();
				ImGui::TreePop();
			}
		}
		ImGui::End();
	}
	if (m_showCameraWindow) RayCameraWindow();
	if (m_showSceneWindow) SceneCameraWindow();
}

void RayTracerLayer::OnDestroy() { CudaModule::Terminate(); }

void RayTracerLayer::SceneCameraWindow() {
	auto scene = GetScene();
	auto editorLayer = Application::GetLayer<EditorLayer>();
	if (!editorLayer) return;
	if (m_leftMouseButtonHold && !Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_LEFT, Windows::GetWindow())) {
		m_leftMouseButtonHold = false;
	}
	if (m_rightMouseButtonHold &&
		!Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT, Windows::GetWindow())) {
		m_rightMouseButtonHold = false;
		m_startMouse = false;
	}
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	if (ImGui::Begin("Scene (Ray)")) {
		if (ImGui::BeginChild("RaySceneRenderer", ImVec2(0, 0), false,
			ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
			ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 5, 5 });
			if (ImGui::BeginMenuBar()) {
				if (ImGui::BeginMenu("Settings")) {
					ImGui::DragFloat("Resolution multiplier", &m_resolutionMultiplier,
						0.01f, 0.1f, 1.0f);
					m_sceneCamera->m_cameraProperties.OnInspect();
					m_sceneCamera->m_rayProperties.OnInspect();
					ImGui::EndMenu();
				}
				ImGui::EndMenuBar();
			}
			ImGui::PopStyleVar();
			ImVec2 viewPortSize = ImGui::GetWindowSize();
			if (m_sceneCamera->m_allowAutoResize)
				m_sceneCamera->m_frameSize =
				glm::vec2(viewPortSize.x, viewPortSize.y - 20) *
				m_resolutionMultiplier;
			if (m_sceneCamera->m_rendered) {
				ImGui::Image(reinterpret_cast<ImTextureID>(m_sceneCamera->m_cameraProperties.m_outputTextureId),
					ImVec2(viewPortSize.x, viewPortSize.y - 20), ImVec2(0, 1), ImVec2(1, 0));
				editorLayer->CameraWindowDragAndDrop();
			}
			else
				ImGui::Text("No mesh in the scene!");
			auto mousePosition = glm::vec2(FLT_MAX, FLT_MIN);
			if (ImGui::IsWindowFocused()) {
				const bool valid = true;
				auto mp = ImGui::GetMousePos();
				auto wp = ImGui::GetWindowPos();
				mousePosition = glm::vec2(mp.x - wp.x, mp.y - wp.y - 20);
				if (valid) {
					if (!m_startMouse) {
						m_lastX = mousePosition.x;
						m_lastY = mousePosition.y;
						m_startMouse = true;
					}
					const float xOffset = mousePosition.x - m_lastX;
					const float yOffset = -mousePosition.y + m_lastY;
					m_lastX = mousePosition.x;
					m_lastY = mousePosition.y;
#pragma region Scene Camera Controller
					if (!m_rightMouseButtonHold &&
						!(mousePosition.x < 0 || mousePosition.y < 0 || mousePosition.x > viewPortSize.x ||
							mousePosition.y > viewPortSize.y) &&
						Inputs::GetMouseInternal(GLFW_MOUSE_BUTTON_RIGHT,
							Windows::GetWindow())) {
						m_rightMouseButtonHold = true;
					}
					if (m_rightMouseButtonHold && !editorLayer->m_lockCamera) {
						const glm::vec3 front =
							editorLayer->m_sceneCameraRotation *
							glm::vec3(0, 0, -1);
						const glm::vec3 right =
							editorLayer->m_sceneCameraRotation *
							glm::vec3(1, 0, 0);
						if (Inputs::GetKeyInternal(GLFW_KEY_W,
							Windows::GetWindow())) {
							editorLayer->m_sceneCameraPosition +=
								front * static_cast<float>(Application::Time().DeltaTime()) *
								editorLayer->m_velocity;
						}
						if (Inputs::GetKeyInternal(GLFW_KEY_S,
							Windows::GetWindow())) {
							editorLayer->m_sceneCameraPosition -=
								front * static_cast<float>(Application::Time().DeltaTime()) *
								editorLayer->m_velocity;
						}
						if (Inputs::GetKeyInternal(GLFW_KEY_A,
							Windows::GetWindow())) {
							editorLayer->m_sceneCameraPosition -=
								right * static_cast<float>(Application::Time().DeltaTime()) *
								editorLayer->m_velocity;
						}
						if (Inputs::GetKeyInternal(GLFW_KEY_D,
							Windows::GetWindow())) {
							editorLayer->m_sceneCameraPosition +=
								right * static_cast<float>(Application::Time().DeltaTime()) *
								editorLayer->m_velocity;
						}
						if (Inputs::GetKeyInternal(GLFW_KEY_LEFT_SHIFT,
							Windows::GetWindow())) {
							editorLayer->m_sceneCameraPosition.y +=
								editorLayer->m_velocity *
								static_cast<float>(Application::Time().DeltaTime());
						}
						if (Inputs::GetKeyInternal(GLFW_KEY_LEFT_CONTROL,
							Windows::GetWindow())) {
							editorLayer->m_sceneCameraPosition.y -=
								editorLayer->m_velocity *
								static_cast<float>(Application::Time().DeltaTime());
						}
						if (xOffset != 0.0f || yOffset != 0.0f) {
							editorLayer->m_sceneCameraYawAngle +=
								xOffset * editorLayer->m_sensitivity;
							editorLayer->m_sceneCameraPitchAngle +=
								yOffset * editorLayer->m_sensitivity;
							if (editorLayer->m_sceneCameraPitchAngle > 89.0f)
								editorLayer->m_sceneCameraPitchAngle = 89.0f;
							if (editorLayer->m_sceneCameraPitchAngle < -89.0f)
								editorLayer->m_sceneCameraPitchAngle = -89.0f;

							editorLayer->m_sceneCameraRotation =
								UniEngine::Camera::ProcessMouseMovement(
									editorLayer->m_sceneCameraYawAngle,
									editorLayer->m_sceneCameraPitchAngle,
									false);
						}
					}
#pragma endregion
				}
			}

#pragma region Gizmos and Entity Selection
			bool mouseSelectEntity = true;
			auto selectedEntity = editorLayer->GetSelectedEntity();
			if (scene->IsEntityValid(selectedEntity)) {
				ImGuizmo::SetOrthographic(false);
				ImGuizmo::SetDrawlist();
				ImGuizmo::SetRect(ImGui::GetWindowPos().x, ImGui::GetWindowPos().y, viewPortSize.x, viewPortSize.y - 20);
				glm::mat4 cameraView =
					glm::inverse(glm::translate(editorLayer->m_sceneCameraPosition) *
						glm::mat4_cast(editorLayer->m_sceneCameraRotation));
				glm::mat4 cameraProjection = m_sceneCamera->GetProjection();
				const auto op = editorLayer->LocalPositionSelected() ? ImGuizmo::OPERATION::TRANSLATE
					: editorLayer->LocalRotationSelected()
					? ImGuizmo::OPERATION::ROTATE
					: ImGuizmo::OPERATION::SCALE;

				auto transform = scene->GetDataComponent<Transform>(selectedEntity);
				GlobalTransform parentGlobalTransform;
				Entity parentEntity = scene->GetParent(selectedEntity);
				if (parentEntity.GetIndex() != 0) {
					parentGlobalTransform = scene->GetDataComponent<GlobalTransform>(
						scene->GetParent(selectedEntity));
				}
				auto globalTransform = scene->GetDataComponent<GlobalTransform>(selectedEntity);

				ImGuizmo::Manipulate(
					glm::value_ptr(cameraView),
					glm::value_ptr(cameraProjection),
					op,
					ImGuizmo::LOCAL,
					glm::value_ptr(globalTransform.m_value));
				if (ImGuizmo::IsUsing()) {
					transform.m_value = glm::inverse(parentGlobalTransform.m_value) * globalTransform.m_value;
					scene->SetDataComponent(selectedEntity, transform);
					transform.Decompose(
						editorLayer->UnsafeGetPreviouslyStoredPosition(),
						editorLayer->UnsafeGetPreviouslyStoredRotation(),
						editorLayer->UnsafeGetPreviouslyStoredScale());
					mouseSelectEntity = false;
				}
			}
#pragma endregion
		}
		ImGui::EndChild();
		auto* window = ImGui::FindWindowByName("Scene (Ray)");
		m_renderingEnabled = !(window->Hidden && !window->Collapsed);
	}
	ImGui::End();
	ImGui::PopStyleVar();
}

void RayTracerLayer::RayCameraWindow() {
	auto editorLayer = Application::GetLayer<EditorLayer>();
	if (!editorLayer) return;
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
	if (ImGui::Begin("Camera (Ray)")) {
		if (ImGui::BeginChild("RayCameraRenderer", ImVec2(0, 0), false,
			ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
			ImVec2 viewPortSize = ImGui::GetWindowSize();
			if (m_rayTracerCamera) {
				if (m_rayTracerCamera->m_allowAutoResize)
					m_rayTracerCamera->m_frameSize = glm::vec2(viewPortSize.x, viewPortSize.y - 20);
				if (m_rayTracerCamera->m_rendered) {
					ImGui::Image(reinterpret_cast<ImTextureID>(m_rayTracerCamera->m_cameraProperties.m_outputTextureId),
						ImVec2(viewPortSize.x, viewPortSize.y - 20), ImVec2(0, 1), ImVec2(1, 0));
					editorLayer->CameraWindowDragAndDrop();
				}
				else
					ImGui::Text("No mesh in the scene!");
			}
			else {
				ImGui::Text("No camera attached!");
			}
		}
		ImGui::EndChild();
	}
	ImGui::End();
	ImGui::PopStyleVar();
}

void RayTracerLayer::Update() {
	if (m_showCameraWindow) {
		ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{ 0, 0 });
		if (ImGui::Begin("Camera (Ray)")) {
			if (ImGui::BeginChild("RayCameraRenderer", ImVec2(0, 0), false,
				ImGuiWindowFlags_None | ImGuiWindowFlags_MenuBar)) {
				if (m_rayTracerCamera && m_rayTracerCamera->m_rendered && ImGui::IsWindowFocused()) {
					Application::GetLayer<EditorLayer>()->m_mainCameraFocusOverride = true;
				}
			}
			ImGui::EndChild();
		}
		ImGui::End();
		ImGui::PopStyleVar();
	}
}

bool
RayTracerLayer::CheckMaterial(RayTracedMaterial& rayTracerMaterial, const std::shared_ptr<Material>& material) const {
	bool changed = false;
	if (rayTracerMaterial.m_materialType == MaterialType::Default && material->m_vertexColorOnly) {
		changed = true;
		rayTracerMaterial.m_materialType = MaterialType::VertexColor;
	}
	else if (rayTracerMaterial.m_materialType == MaterialType::VertexColor && !material->m_vertexColorOnly) {
		changed = true;
		rayTracerMaterial.m_materialType = MaterialType::Default;
	}

	if (changed || rayTracerMaterial.m_version != material->m_version) {
		rayTracerMaterial.m_handle = material->GetHandle();
		rayTracerMaterial.m_version = material->m_version;
		rayTracerMaterial.m_materialProperties = material->m_materialProperties;

		auto albedoTexture = material->m_albedoTexture.Get<Texture2D>();
		if (albedoTexture &&
			albedoTexture->UnsafeGetGLTexture()) {
			if (albedoTexture
				->UnsafeGetGLTexture()
				->Id() != rayTracerMaterial.m_albedoTexture.m_textureId) {
				rayTracerMaterial.m_albedoTexture.m_textureId =
					albedoTexture
					->UnsafeGetGLTexture()
					->Id();
			}
		}
		else if (rayTracerMaterial.m_albedoTexture.m_textureId != 0) {
			rayTracerMaterial.m_albedoTexture.m_textureId = 0;
		}
		auto normalTexture = material->m_normalTexture.Get<Texture2D>();
		if (normalTexture &&
			normalTexture->UnsafeGetGLTexture()) {
			if (normalTexture
				->UnsafeGetGLTexture()
				->Id() != rayTracerMaterial.m_normalTexture.m_textureId) {
				rayTracerMaterial.m_normalTexture.m_textureId =
					normalTexture
					->UnsafeGetGLTexture()
					->Id();
			}
		}
		else if (rayTracerMaterial.m_normalTexture.m_textureId != 0) {
			rayTracerMaterial.m_normalTexture.m_textureId = 0;
		}
		auto roughnessTexture = material->m_normalTexture.Get<Texture2D>();
		if (roughnessTexture &&
			roughnessTexture->UnsafeGetGLTexture()) {
			if (roughnessTexture
				->UnsafeGetGLTexture()
				->Id() != rayTracerMaterial.m_roughnessTexture.m_textureId) {
				rayTracerMaterial.m_roughnessTexture.m_textureId =
					normalTexture
					->UnsafeGetGLTexture()
					->Id();
			}
		}
		else if (rayTracerMaterial.m_roughnessTexture.m_textureId != 0) {
			rayTracerMaterial.m_roughnessTexture.m_textureId = 0;
		}
		auto metallicTexture = material->m_metallicTexture.Get<Texture2D>();
		if (metallicTexture &&
			metallicTexture->UnsafeGetGLTexture()) {
			if (metallicTexture
				->UnsafeGetGLTexture()
				->Id() != rayTracerMaterial.m_metallicTexture.m_textureId) {
				rayTracerMaterial.m_metallicTexture.m_textureId =
					metallicTexture
					->UnsafeGetGLTexture()
					->Id();
			}
		}
		else if (rayTracerMaterial.m_metallicTexture.m_textureId != 0) {
			rayTracerMaterial.m_metallicTexture.m_textureId = 0;
		}
		changed = true;
	}

	return changed;
}

bool RayTracerLayer::CheckCompressedBTF(RayTracedMaterial& rayTracerMaterial,
	const std::shared_ptr<CompressedBTF>& compressedBtf) const {
	bool changed = false;
	if (rayTracerMaterial.m_materialType != MaterialType::CompressedBTF) {
		changed = true;
		rayTracerMaterial.m_materialType = MaterialType::CompressedBTF;
	}
	if (rayTracerMaterial.m_version != compressedBtf->m_version) {
		changed = true;
		rayTracerMaterial.m_version = compressedBtf->m_version;
		rayTracerMaterial.m_btfBase = &compressedBtf->m_bTFBase;
	}
	return changed;
}

