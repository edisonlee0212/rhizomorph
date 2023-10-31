#include "Soil.hpp"

#include <cassert>

#include "EcoSysLabLayer.hpp"
#include "EditorLayer.hpp"
#include "Graphics.hpp"
#include "HeightField.hpp"
using namespace EcoSysLab;


bool OnInspectSoilParameters(SoilParameters& soilParameters)
{
	bool changed = false;
	if (ImGui::TreeNodeEx("Soil Parameters", ImGuiTreeNodeFlags_DefaultOpen))
	{
		if (ImGui::InputInt3("VoxelGrid Resolution", (int*)&soilParameters.m_voxelResolution))
		{
			changed = true;
		}
		if (ImGui::DragFloat("Delta X", &soilParameters.m_deltaX, 0.01f, 0.01f, 1.0f))
		{
			changed = true;
		}
		if (ImGui::DragFloat("Delta time", &soilParameters.m_deltaTime, 0.01f, 0.0f, 10.0f))
		{
			changed = true;
		}
		if (ImGui::InputFloat3("Bounding Box Min", (float*)&soilParameters.m_boundingBoxMin))
		{
			changed = true;
		}
		// TODO: boundaries
		if (ImGui::DragFloat("Diffusion Force", &soilParameters.m_diffusionForce, 0.01f, 0.0f, 999.0f))
		{
			changed = true;
		}
		if (ImGui::DragFloat3("Gravity Force", &soilParameters.m_gravityForce.x, 0.01f, 0.0f, 999.0f))
		{
			changed = true;
		}
		ImGui::TreePop();
	}
	return changed;
}

void SetSoilPhysicalMaterial(Noises3D& c, Noises3D& p, float sandRatio, float siltRatio, float clayRatio, float compactness)
{
	assert(compactness <= 1.0f && compactness >= 0.0f);

	const float weight = sandRatio + siltRatio + clayRatio;
	sandRatio = sandRatio * compactness / weight;
	siltRatio = siltRatio * compactness / weight;
	clayRatio = clayRatio * compactness / weight;
	const float airRatio = 1.f - compactness;

	static glm::vec2 sandMaterialProperties = glm::vec2(0.9f, 15.0f);
	static glm::vec2 siltMaterialProperties = glm::vec2(1.9f, 1.5f);
	static glm::vec2 clayMaterialProperties = glm::vec2(2.1f, 0.05f);
	static glm::vec2 airMaterialProperties = glm::vec2(5.0f, 30.0f);

	c.m_noiseDescriptors.resize(1);
	p.m_noiseDescriptors.resize(1);
	c.m_noiseDescriptors[0].m_type = 0;
	c.m_noiseDescriptors[1].m_type = 0;
	c.m_noiseDescriptors[0].m_offset = sandRatio * sandMaterialProperties.x + siltRatio * siltMaterialProperties.x + clayRatio * clayMaterialProperties.x + airRatio * airMaterialProperties.x;
	p.m_noiseDescriptors[0].m_offset = sandRatio * sandMaterialProperties.y + siltRatio * siltMaterialProperties.y + clayRatio * clayMaterialProperties.y + airRatio * airMaterialProperties.y;
}

void NoiseSoilLayerDescriptor::OnInspect()
{
	bool changed = false;
	if (ImGui::TreeNodeEx("Generate from preset soil ratio")) {
		static float sandRatio = 0.1f;
		static float siltRatio = 0.1f;
		static float clayRatio = 0.8f;
		static float compactness = 1.0f;
		ImGui::SliderFloat("Sand ratio", &sandRatio, 0.0f, 1.0f);
		ImGui::SliderFloat("Silt ratio", &siltRatio, 0.0f, 1.0f);
		ImGui::SliderFloat("Clay ratio", &clayRatio, 0.0f, 1.0f);
		ImGui::SliderFloat("Compactness", &compactness, 0.0f, 1.0f);
		if (ImGui::Button("Generate soil"))
		{
			SetSoilPhysicalMaterial(m_capacity, m_permeability, sandRatio, siltRatio, clayRatio, compactness);
			changed = true;
		}
		if (ImGui::TreeNode("Generate from preset combination")) {
			static unsigned soilTypePreset = 0;
			ImGui::Combo({ "Select soil combination preset" }, { "Clay", "Silty Clay", "Loam", "Sand", "Loamy Sand" }, soilTypePreset);
			if (ImGui::Button("Apply combination"))
			{
				switch (static_cast<SoilMaterialType>(soilTypePreset))
				{
				case SoilMaterialType::Clay:
					sandRatio = 0.1f;
					siltRatio = 0.1f;
					clayRatio = 0.8f;
					compactness = 1.f;
					break;
				case SoilMaterialType::SiltyClay:
					sandRatio = 0.1f;
					siltRatio = 0.4f;
					clayRatio = 0.5f;
					compactness = 1.f;
					break;
				case SoilMaterialType::Loam:
					sandRatio = 0.4f;
					siltRatio = 0.4f;
					clayRatio = 0.2f;
					compactness = 1.f;
					break;
				case SoilMaterialType::Sand:
					sandRatio = 1.f;
					siltRatio = 0.f;
					clayRatio = 0.f;
					compactness = 1.f;
					break;
				case SoilMaterialType::LoamySand:
					sandRatio = 0.8f;
					siltRatio = 0.1f;
					clayRatio = 0.1f;
					compactness = 1.f;
					break;
				}
			}
			ImGui::TreePop();
		}
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Capacity")) {
		changed = m_capacity.OnInspect() || changed;
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Permeability")) {
		changed = m_permeability.OnInspect() || changed;
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Density")) {
		changed = m_density.OnInspect() || changed;
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Initial nutrients")) {
		changed = m_initialNutrients.OnInspect() || changed;
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Initial water")) {
		changed = m_initialWater.OnInspect() || changed;
		ImGui::TreePop();
	}

	if (ImGui::TreeNode("Thickness")) {
		changed = m_thickness.OnInspect() || changed;
		ImGui::TreePop();
	}
	if (ImGui::TreeNode("Textures")) {
		if (Editor::DragAndDropButton<Texture2D>(m_albedoTexture, "Albedo")) changed = true;
		if (Editor::DragAndDropButton<Texture2D>(m_roughnessTexture, "Roughness")) changed = true;
		if (Editor::DragAndDropButton<Texture2D>(m_metallicTexture, "Metallic")) changed = true;
		if (Editor::DragAndDropButton<Texture2D>(m_normalTexture, "Normal")) changed = true;
		if (Editor::DragAndDropButton<Texture2D>(m_heightTexture, "Height")) changed = true;
		ImGui::TreePop();
	}
	if (changed) m_saved = false;
}

void NoiseSoilLayerDescriptor::Serialize(YAML::Emitter& out)
{
	m_capacity.Save("m_capacity", out);
	m_permeability.Save("m_permeability", out);
	m_density.Save("m_density", out);
	m_initialNutrients.Save("m_initialNutrients", out);
	m_initialWater.Save("m_initialWater", out);

	m_thickness.Save("m_thickness", out);

	m_albedoTexture.Save("m_albedoTexture", out);
	m_roughnessTexture.Save("m_roughnessTexture", out);
	m_metallicTexture.Save("m_metallicTexture", out);
	m_normalTexture.Save("m_normalTexture", out);
	m_heightTexture.Save("m_heightTexture", out);
}

void NoiseSoilLayerDescriptor::Deserialize(const YAML::Node& in)
{
	m_capacity.Load("m_capacity", in);
	m_permeability.Load("m_permeability", in);
	m_density.Load("m_density", in);
	m_initialNutrients.Load("m_initialNutrients", in);
	m_initialWater.Load("m_initialWater", in);
	m_thickness.Load("m_thickness", in);

	m_albedoTexture.Load("m_albedoTexture", in);
	m_roughnessTexture.Load("m_roughnessTexture", in);
	m_metallicTexture.Load("m_metallicTexture", in);
	m_normalTexture.Load("m_normalTexture", in);
	m_heightTexture.Load("m_heightTexture", in);
}

void NoiseSoilLayerDescriptor::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_albedoTexture);
	list.push_back(m_roughnessTexture);
	list.push_back(m_metallicTexture);
	list.push_back(m_normalTexture);
	list.push_back(m_heightTexture);
}

void SoilDescriptor::OnInspect()
{
	bool changed = false;
	if (Editor::DragAndDropButton<HeightField>(m_heightField, "Height Field", true))
	{
		changed = true;
	}

	/*
	glm::ivec3 resolution = m_voxelResolution;
	if (ImGui::DragInt3("VoxelGrid Resolution", &resolution.x, 1, 1, 100))
	{
		m_voxelResolution = resolution;
		changed = true;
	}
	if (ImGui::DragFloat3("VoxelGrid Bounding box min", &m_boundingBoxMin.x, 0.01f))
	{
		changed = true;
	}
	*/

	if (ImGui::Button("Instantiate")) {
		auto scene = Application::GetActiveScene();
		auto soilEntity = scene->CreateEntity(GetTitle());
		auto soil = scene->GetOrSetPrivateComponent<Soil>(soilEntity).lock();
		soil->m_soilDescriptor = ProjectManager::GetAsset(GetHandle());
		soil->InitializeSoilModel();
	}


	if (OnInspectSoilParameters(m_soilParameters))
	{
		changed = true;
	}
	AssetRef tempSoilLayerDescriptorHolder;
	if (Editor::DragAndDropButton<NoiseSoilLayerDescriptor>(tempSoilLayerDescriptorHolder, "Drop new SoilLayerDescriptor here...")) {
		auto sld = tempSoilLayerDescriptorHolder.Get<NoiseSoilLayerDescriptor>();
		if (sld) {
			m_soilLayerDescriptors.emplace_back(sld);
			changed = true;
		}
		tempSoilLayerDescriptorHolder.Clear();
	}
	for (int i = 0; i < m_soilLayerDescriptors.size(); i++)
	{
		if (auto soilLayerDescriptor = m_soilLayerDescriptors[i].Get<NoiseSoilLayerDescriptor>())
		{
			if (ImGui::TreeNodeEx(("No." + std::to_string(i + 1)).c_str(), ImGuiTreeNodeFlags_DefaultOpen))
			{
				ImGui::Text(("Name: " + soilLayerDescriptor->GetTitle()).c_str());

				if (ImGui::Button("Remove"))
				{
					m_soilLayerDescriptors.erase(m_soilLayerDescriptors.begin() + i);
					changed = true;
					ImGui::TreePop();
					continue;
				}
				if (!soilLayerDescriptor->Saved())
				{
					ImGui::SameLine();
					if (ImGui::Button("Save"))
					{
						soilLayerDescriptor->Save();
					}
				}
				if (i < m_soilLayerDescriptors.size() - 1) {
					ImGui::SameLine();
					if (ImGui::Button("Move down"))
					{
						changed = true;
						const auto temp = m_soilLayerDescriptors[i];
						m_soilLayerDescriptors[i] = m_soilLayerDescriptors[i + 1];
						m_soilLayerDescriptors[i + 1] = temp;
					}
				}
				if (i > 0) {
					ImGui::SameLine();
					if (ImGui::Button("Move up"))
					{
						changed = true;
						const auto temp = m_soilLayerDescriptors[i - 1];
						m_soilLayerDescriptors[i - 1] = m_soilLayerDescriptors[i];
						m_soilLayerDescriptors[i] = temp;
					}
				}
				if (ImGui::TreeNode("Settings")) {
					soilLayerDescriptor->OnInspect();
					ImGui::TreePop();
				}
				ImGui::TreePop();
			}
		}
		else
		{
			m_soilLayerDescriptors.erase(m_soilLayerDescriptors.begin() + i);
			i--;
		}
	}


	if (changed) m_saved = false;
}



void Soil::OnInspect()
{
	if (Editor::DragAndDropButton<SoilDescriptor>(m_soilDescriptor, "SoilDescriptor", true)) {
		InitializeSoilModel();
	}
	auto soilDescriptor = m_soilDescriptor.Get<SoilDescriptor>();
	if (soilDescriptor)
	{
		if (ImGui::Button("Generate surface mesh")) {
			GenerateMesh();
		}
		// Show some general properties:

		static float xDepth = 1;
		static float zDepth = 1;
		static float waterFactor = 20.f;
		static float nutrientFactor = 1.f;
		static bool groundSurface = false;
		ImGui::DragFloat("Cutout X Depth", &xDepth, 0.01f, 0.0f, 1.0f, "%.2f");
		ImGui::DragFloat("Cutout Z Depth", &zDepth, 0.01f, 0.0f, 1.0f, "%.2f");
		ImGui::DragFloat("Water factor", &waterFactor, 0.0001f, 0.0f, 1.0f, "%.4f");
		ImGui::DragFloat("Nutrient factor", &nutrientFactor, 0.0001f, 0.0f, 1.0f, "%.4f");
		ImGui::Checkbox("Ground surface", &groundSurface);
		if (ImGui::Button("Generate Cutout"))
		{
			auto scene = Application::GetActiveScene();
			auto owner = GetOwner();
			for (const auto& child : scene->GetChildren(owner))
			{
				if (scene->GetEntityName(child) == "CutOut")
				{
					scene->DeleteEntity(child);
					break;
				}
			}

			auto cutOutEntity = GenerateCutOut(xDepth, zDepth, waterFactor, nutrientFactor, groundSurface);

			scene->SetParent(cutOutEntity, owner);
		}
		if (ImGui::Button("Generate Cube"))
		{
			auto scene = Application::GetActiveScene();
			auto owner = GetOwner();
			for (const auto& child : scene->GetChildren(owner))
			{
				if (scene->GetEntityName(child) == "Cube")
				{
					scene->DeleteEntity(child);
					break;
				}
			}

			auto cutOutEntity = GenerateFullBox(waterFactor, nutrientFactor, groundSurface);

			scene->SetParent(cutOutEntity, owner);
		}

		if(ImGui::Button("Temporal Progression"))
		{
			m_temporalProgressionProgress = 0;
			m_temporalProgression = true;
		}

		//auto soilDescriptor = m_soilDescriptor.Get<SoilDescriptor>();
		//if (!m_soilModel.m_initialized) m_soilModel.Initialize(soilDescriptor->m_soilParameters);
		assert(m_soilModel.m_initialized);
		if (ImGui::Button("Initialize"))
		{
			InitializeSoilModel();
		}
		if (ImGui::Button("Reset"))
		{
			m_soilModel.Reset();
		}

		if (ImGui::Button("Split root test"))
		{
			SplitRootTestSetup();
		}
		static AssetRef soilAlbedoTexture;
		static AssetRef soilNormalTexture;
		static AssetRef soilRoughnessTexture;
		static AssetRef soilHeightTexture;
		static AssetRef soilMetallicTexture;
		Editor::DragAndDropButton<Texture2D>(soilAlbedoTexture, "Albedo", true);
		Editor::DragAndDropButton<Texture2D>(soilNormalTexture, "Normal", true);
		Editor::DragAndDropButton<Texture2D>(soilRoughnessTexture, "Roughness", true);
		Editor::DragAndDropButton<Texture2D>(soilHeightTexture, "Height", true);
		Editor::DragAndDropButton<Texture2D>(soilMetallicTexture, "Metallic", true);
		if(ImGui::Button("Nutrient Transport: Sand"))
		{
			auto albedo = soilAlbedoTexture.Get<Texture2D>();
			auto normal = soilNormalTexture.Get<Texture2D>();
			auto roughness = soilRoughnessTexture.Get<Texture2D>();
			auto height = soilHeightTexture.Get<Texture2D>();
			auto metallic = soilMetallicTexture.Get<Texture2D>();
			std::shared_ptr<SoilMaterialTexture> soilMaterialTexture = std::make_shared<SoilMaterialTexture>();
			{
				if (albedo)
				{
					albedo->GetRgbaChannelData(soilMaterialTexture->m_color_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_color_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_color_map.begin(), soilMaterialTexture->m_color_map.end(), glm::vec4(1));
				}
				if (height) {
					height->GetRedChannelData(soilMaterialTexture->m_height_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_height_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_height_map.begin(), soilMaterialTexture->m_height_map.end(), 1.0f);
				}
				if (metallic) {
					metallic->GetRedChannelData(soilMaterialTexture->m_metallic_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_metallic_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_metallic_map.begin(), soilMaterialTexture->m_metallic_map.end(), 0.2f);
				}
				if (roughness) {
					roughness->GetRedChannelData(soilMaterialTexture->m_roughness_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_roughness_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_roughness_map.begin(), soilMaterialTexture->m_roughness_map.end(), 0.8f);
				}
				if (normal) {
					normal->GetRgbChannelData(soilMaterialTexture->m_normal_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_normal_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_normal_map.begin(), soilMaterialTexture->m_normal_map.end(), glm::vec3(0, 0, 1));
				}
			}
			m_soilModel.Test_NutrientTransport_Sand(soilMaterialTexture);
		}
		if (ImGui::Button("Nutrient Transport: Loam"))
		{
			auto albedo = soilAlbedoTexture.Get<Texture2D>();
			auto normal = soilNormalTexture.Get<Texture2D>();
			auto roughness = soilRoughnessTexture.Get<Texture2D>();
			auto height = soilHeightTexture.Get<Texture2D>();
			auto metallic = soilMetallicTexture.Get<Texture2D>();
			std::shared_ptr<SoilMaterialTexture> soilMaterialTexture = std::make_shared<SoilMaterialTexture>();
			{
				if (albedo)
				{
					albedo->GetRgbaChannelData(soilMaterialTexture->m_color_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_color_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_color_map.begin(), soilMaterialTexture->m_color_map.end(), glm::vec4(1));
				}
				if (height) {
					height->GetRedChannelData(soilMaterialTexture->m_height_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_height_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_height_map.begin(), soilMaterialTexture->m_height_map.end(), 1.0f);
				}
				if (metallic) {
					metallic->GetRedChannelData(soilMaterialTexture->m_metallic_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_metallic_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_metallic_map.begin(), soilMaterialTexture->m_metallic_map.end(), 0.2f);
				}
				if (roughness) {
					roughness->GetRedChannelData(soilMaterialTexture->m_roughness_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_roughness_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_roughness_map.begin(), soilMaterialTexture->m_roughness_map.end(), 0.8f);
				}
				if (normal) {
					normal->GetRgbChannelData(soilMaterialTexture->m_normal_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_normal_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_normal_map.begin(), soilMaterialTexture->m_normal_map.end(), glm::vec3(0, 0, 1));
				}
			}
			m_soilModel.Test_NutrientTransport_Loam(soilMaterialTexture);
		}
		if (ImGui::Button("Nutrient Transport: Silt"))
		{
			auto albedo = soilAlbedoTexture.Get<Texture2D>();
			auto normal = soilNormalTexture.Get<Texture2D>();
			auto roughness = soilRoughnessTexture.Get<Texture2D>();
			auto height = soilHeightTexture.Get<Texture2D>();
			auto metallic = soilMetallicTexture.Get<Texture2D>();
			std::shared_ptr<SoilMaterialTexture> soilMaterialTexture = std::make_shared<SoilMaterialTexture>();
			{
				if (albedo)
				{
					albedo->GetRgbaChannelData(soilMaterialTexture->m_color_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_color_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_color_map.begin(), soilMaterialTexture->m_color_map.end(), glm::vec4(1));
				}
				if (height) {
					height->GetRedChannelData(soilMaterialTexture->m_height_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_height_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_height_map.begin(), soilMaterialTexture->m_height_map.end(), 1.0f);
				}
				if (metallic) {
					metallic->GetRedChannelData(soilMaterialTexture->m_metallic_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_metallic_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_metallic_map.begin(), soilMaterialTexture->m_metallic_map.end(), 0.2f);
				}
				if (roughness) {
					roughness->GetRedChannelData(soilMaterialTexture->m_roughness_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_roughness_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_roughness_map.begin(), soilMaterialTexture->m_roughness_map.end(), 0.8f);
				}
				if (normal) {
					normal->GetRgbChannelData(soilMaterialTexture->m_normal_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilMaterialTexture->m_normal_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilMaterialTexture->m_normal_map.begin(), soilMaterialTexture->m_normal_map.end(), glm::vec3(0, 0, 1));
				}
			}
			m_soilModel.Test_NutrientTransport_Silt(soilMaterialTexture);
		}
		ImGui::InputFloat("Diffusion Force", &m_soilModel.m_diffusionForce);
		ImGui::InputFloat3("Gravity Force", &m_soilModel.m_gravityForce.x);

		ImGui::Checkbox("Auto step", &m_autoStep);
		if (ImGui::Button("Step") || m_autoStep)
		{
			if (m_irrigation)
				m_soilModel.Irrigation();
			m_soilModel.Step();
		}
		ImGui::SliderFloat("Irrigation amount", &m_soilModel.m_irrigationAmount, 0.01, 100, "%.2f", ImGuiSliderFlags_Logarithmic);
		ImGui::Checkbox("apply Irrigation", &m_irrigation);

		ImGui::InputFloat3("Source position", (float*)&m_sourcePositon);
		ImGui::SliderFloat("Source amount", &m_sourceAmount, 1, 10000, "%.4f", ImGuiSliderFlags_Logarithmic);
		ImGui::InputFloat("Source width", &m_sourceWidth, 0.1, 100, "%.4f", ImGuiSliderFlags_Logarithmic);
		if (ImGui::Button("Apply Source"))
		{
			m_soilModel.ChangeWater(m_sourcePositon, m_sourceAmount, m_sourceWidth);
		}


		
	}
}
Entity Soil::GenerateSurfaceQuadX(float depth, const glm::vec2& minXY, const glm::vec2 maxXY, float waterFactor, float nutrientFactor)
{
	auto scene = Application::GetActiveScene();
	auto quadEntity = scene->CreateEntity("Slice");
	auto material = ProjectManager::CreateTemporaryAsset<Material>();
	material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
	auto albedoTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	auto normalTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	auto metallicTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	auto roughnessTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	std::vector<glm::vec4> albedoData;
	std::vector<glm::vec3> normalData;
	std::vector<float> metallicData;
	std::vector<float> roughnessData;
	glm::ivec2 textureResolution;
	m_soilModel.GetSoilTextureSlideX(depth, minXY, maxXY, albedoData, normalData, roughnessData, metallicData, textureResolution, waterFactor, nutrientFactor);
	albedoTex->SetRgbaChannelData(albedoData, textureResolution);
	normalTex->SetRgbChannelData(normalData, textureResolution);
	metallicTex->SetRedChannelData(metallicData, textureResolution);
	roughnessTex->SetRedChannelData(roughnessData, textureResolution);
	material->m_albedoTexture = albedoTex;
	material->m_normalTexture = normalTex;
	material->m_metallicTexture = metallicTex;
	material->m_roughnessTexture = roughnessTex;
	const auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(quadEntity).lock();
	meshRenderer->m_material = material;
	material->m_drawSettings.m_cullFace = false;
	meshRenderer->m_mesh = DefaultResources::Primitives::Quad;

	GlobalTransform globalTransform;
	glm::vec3 scale;
	glm::vec3 position;
	glm::vec3 rotation;
	auto soilModelSize = glm::vec3(m_soilModel.m_resolution) * m_soilModel.m_dx;

	scale = glm::vec3(soilModelSize.z * (maxXY.x - minXY.x), 1.0f, soilModelSize.y * (maxXY.y - minXY.y));
	rotation = glm::vec3(glm::radians(90.0f), -glm::radians(90.0f), 0.0f);
	position = m_soilModel.m_boundingBoxMin + glm::vec3(soilModelSize.x * depth, soilModelSize.y * (minXY.y + maxXY.y) * 0.5f, soilModelSize.z * (minXY.x + maxXY.x) * 0.5f);
	globalTransform.SetPosition(position);
	globalTransform.SetEulerRotation(rotation);
	globalTransform.SetScale(scale);
	scene->SetDataComponent(quadEntity, globalTransform);
	return quadEntity;
}

Entity Soil::GenerateSurfaceQuadZ(float depth, const glm::vec2& minXY, const glm::vec2 maxXY, float waterFactor, float nutrientFactor)
{
	auto scene = Application::GetActiveScene();
	auto quadEntity = scene->CreateEntity("Slice");

	const auto meshRenderer = scene->GetOrSetPrivateComponent<MeshRenderer>(quadEntity).lock();
	auto material = ProjectManager::CreateTemporaryAsset<Material>();
	material->SetProgram(DefaultResources::GLPrograms::StandardProgram);
	auto albedoTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	auto normalTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	auto metallicTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	auto roughnessTex = ProjectManager::CreateTemporaryAsset<Texture2D>();
	std::vector<glm::vec4> albedoData;
	std::vector<glm::vec3> normalData;
	std::vector<float> metallicData;
	std::vector<float> roughnessData;
	glm::ivec2 textureResolution;
	m_soilModel.GetSoilTextureSlideZ(depth, minXY, maxXY, albedoData, normalData, roughnessData, metallicData, textureResolution, waterFactor, nutrientFactor);
	albedoTex->SetRgbaChannelData(albedoData, textureResolution);
	normalTex->SetRgbChannelData(normalData, textureResolution);
	metallicTex->SetRedChannelData(metallicData, textureResolution);
	roughnessTex->SetRedChannelData(roughnessData, textureResolution);
	material->m_albedoTexture = albedoTex;
	material->m_normalTexture = normalTex;
	material->m_metallicTexture = metallicTex;
	material->m_roughnessTexture = roughnessTex;


	meshRenderer->m_material = material;
	material->m_drawSettings.m_cullFace = false;
	meshRenderer->m_mesh = DefaultResources::Primitives::Quad;

	GlobalTransform globalTransform;
	glm::vec3 scale;
	glm::vec3 position;
	glm::vec3 rotation;
	auto soilModelSize = glm::vec3(m_soilModel.m_resolution) * m_soilModel.m_dx;

	scale = glm::vec3(soilModelSize.x * (maxXY.x - minXY.x), 1.0f, soilModelSize.y * (maxXY.y - minXY.y));
	rotation = glm::vec3(glm::radians(90.0f), 0.0f, 0.0f);
	position = m_soilModel.m_boundingBoxMin + glm::vec3(soilModelSize.x * (minXY.x + maxXY.x) * 0.5f, soilModelSize.y * (minXY.y + maxXY.y) * 0.5f, soilModelSize.z * depth);

	globalTransform.SetPosition(position);
	globalTransform.SetEulerRotation(rotation);
	globalTransform.SetScale(scale);
	scene->SetDataComponent(quadEntity, globalTransform);
	return quadEntity;
}

Entity Soil::GenerateCutOut(float xDepth, float zDepth, float waterFactor, float nutrientFactor, bool groundSurface)
{
	auto scene = Application::GetActiveScene();
	auto combinedEntity = scene->CreateEntity("CutOut");

	if (zDepth <= 0.99f) {
		auto quad1 = GenerateSurfaceQuadX(0, { 0, 0 }, { 1.0 - zDepth , 1 }, waterFactor, nutrientFactor);
		scene->SetParent(quad1, combinedEntity);
	}
	if (zDepth >= 0.01f && xDepth <= 0.99f) {
		auto quad2 = GenerateSurfaceQuadX(xDepth, { 1.0 - zDepth, 0 }, { 1 , 1 }, waterFactor, nutrientFactor);
		scene->SetParent(quad2, combinedEntity);
	}
	if (xDepth >= 0.01f) {
		auto quad3 = GenerateSurfaceQuadZ(1.0 - zDepth, { 0, 0 }, { xDepth , 1 }, waterFactor, nutrientFactor);
		scene->SetParent(quad3, combinedEntity);
	}
	if (xDepth <= 0.99f) {
		auto quad4 = GenerateSurfaceQuadZ(1.0, { xDepth, 0 }, { 1 , 1 }, waterFactor, nutrientFactor);
		scene->SetParent(quad4, combinedEntity);
	}
	
	
	
	if (groundSurface) {
		auto groundSurface = GenerateMesh(xDepth, zDepth);
		auto soilDescriptor = m_soilDescriptor.Get<SoilDescriptor>();
		if (soilDescriptor)
		{
			auto& soilLayerDescriptors = soilDescriptor->m_soilLayerDescriptors;

			if (!soilLayerDescriptors.empty())
			{
				auto firstDescriptor = soilLayerDescriptors[0].Get<NoiseSoilLayerDescriptor>();
				if (firstDescriptor)
				{
					auto mmr = scene->GetOrSetPrivateComponent<MeshRenderer>(groundSurface).lock();
					auto mat = mmr->m_material.Get<Material>();
					mat->m_albedoTexture = firstDescriptor->m_albedoTexture;
					mat->m_normalTexture = firstDescriptor->m_normalTexture;
					mat->m_roughnessTexture = firstDescriptor->m_roughnessTexture;
					mat->m_metallicTexture = firstDescriptor->m_metallicTexture;
				}
			}
		}
	}
	return combinedEntity;
}

Entity Soil::GenerateFullBox(float waterFactor, float nutrientFactor, bool groundSurface)
{
	auto scene = Application::GetActiveScene();
	auto combinedEntity = scene->CreateEntity("Cube");

	
	auto quad1 = GenerateSurfaceQuadX(0, { 0, 0 }, { 1 , 1 }, waterFactor, nutrientFactor);
	scene->SetParent(quad1, combinedEntity);
	
	
	auto quad2 = GenerateSurfaceQuadX(1, { 0, 0 }, { 1 , 1 }, waterFactor, nutrientFactor);
	scene->SetParent(quad2, combinedEntity);
	
	
	auto quad3 = GenerateSurfaceQuadZ(0, { 0, 0 }, { 1 , 1 }, waterFactor, nutrientFactor);
	scene->SetParent(quad3, combinedEntity);

	auto quad4 = GenerateSurfaceQuadZ(1, { 0, 0 }, { 1 , 1 }, waterFactor, nutrientFactor);
	scene->SetParent(quad4, combinedEntity);
	



	if (groundSurface) {
		auto groundSurface = GenerateMesh(0, 0);
		auto soilDescriptor = m_soilDescriptor.Get<SoilDescriptor>();
		if (soilDescriptor)
		{
			auto& soilLayerDescriptors = soilDescriptor->m_soilLayerDescriptors;

			if (!soilLayerDescriptors.empty())
			{
				auto firstDescriptor = soilLayerDescriptors[0].Get<NoiseSoilLayerDescriptor>();
				if (firstDescriptor)
				{
					auto mmr = scene->GetOrSetPrivateComponent<MeshRenderer>(groundSurface).lock();
					auto mat = mmr->m_material.Get<Material>();
					mat->m_albedoTexture = firstDescriptor->m_albedoTexture;
					mat->m_normalTexture = firstDescriptor->m_normalTexture;
					mat->m_roughnessTexture = firstDescriptor->m_roughnessTexture;
					mat->m_metallicTexture = firstDescriptor->m_metallicTexture;
				}
			}
		}
	}
	return combinedEntity;
}

void Soil::Serialize(YAML::Emitter& out)
{
	m_soilDescriptor.Save("m_soilDescriptor", out);
}

void Soil::Deserialize(const YAML::Node& in)
{
	m_soilDescriptor.Load("m_soilDescriptor", in);
	InitializeSoilModel();
}

void Soil::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_soilDescriptor);
}

Entity Soil::GenerateMesh(float xDepth, float zDepth)
{
	const auto soilDescriptor = m_soilDescriptor.Get<SoilDescriptor>();
	if (!soilDescriptor)
	{
		UNIENGINE_ERROR("No soil descriptor!");
		return Entity();
	}
	const auto heightField = soilDescriptor->m_heightField.Get<HeightField>();
	if (!heightField)
	{
		UNIENGINE_ERROR("No height field!");
		return Entity();
	}
	std::vector<Vertex> vertices;
	std::vector<glm::uvec3> triangles;
	heightField->GenerateMesh(glm::vec2(soilDescriptor->m_soilParameters.m_boundingBoxMin.x, soilDescriptor->m_soilParameters.m_boundingBoxMin.z),
		glm::uvec2(soilDescriptor->m_soilParameters.m_voxelResolution.x, soilDescriptor->m_soilParameters.m_voxelResolution.z), soilDescriptor->m_soilParameters.m_deltaX, vertices, triangles
	, xDepth, zDepth);

	const auto scene = Application::GetActiveScene();
	const auto self = GetOwner();
	Entity groundSurfaceEntity;
	const auto children = scene->GetChildren(self);

	for (const auto& child : children) {
		auto name = scene->GetEntityName(child);
		if (name == "Ground surface") {
			groundSurfaceEntity = child;
			break;
		}
	}
	if (groundSurfaceEntity.GetIndex() == 0)
	{
		groundSurfaceEntity = scene->CreateEntity("Ground surface");
		scene->SetParent(groundSurfaceEntity, self);
	}

	const auto meshRenderer =
		scene->GetOrSetPrivateComponent<MeshRenderer>(groundSurfaceEntity).lock();
	const auto mesh = ProjectManager::CreateTemporaryAsset<Mesh>();
	const auto material = ProjectManager::CreateTemporaryAsset<Material>();
	mesh->SetVertices(17, vertices, triangles);
	meshRenderer->m_mesh = mesh;
	meshRenderer->m_material = material;

	return groundSurfaceEntity;
}


void Soil::InitializeSoilModel()
{
	auto soilDescriptor = m_soilDescriptor.Get<SoilDescriptor>();
	if (soilDescriptor)
	{
		auto heightField = soilDescriptor->m_heightField.Get<HeightField>();

		auto params = soilDescriptor->m_soilParameters;
		params.m_boundary_x = SoilModel::Boundary::wrap;
		params.m_boundary_y = SoilModel::Boundary::absorb;
		params.m_boundary_z = SoilModel::Boundary::wrap;

		SoilSurface soilSurface;
		std::vector<SoilLayer> soilLayers;


		if (heightField)
		{
			soilSurface.m_height = [heightField](const glm::vec2& position)
			{
				return heightField->GetValue(glm::vec2(position.x, position.y));
			};
		}
		else {

			soilSurface.m_height = [&](const glm::vec2& position)
			{
				return 0.0f;
			};

		}

		m_soilModel.m_materialTextureResolution = soilDescriptor->m_textureResolution;
		//Add top air layer
		int materialIndex = 0;

		soilLayers.emplace_back();
		auto& firstLayer = soilLayers.back();
		firstLayer.m_mat = SoilPhysicalMaterial({ materialIndex,
					[](const glm::vec3& pos) { return 1.0f; },
					[](const glm::vec3& pos) { return 0.0f; },
					[](const glm::vec3& pos) { return 0.0f; },
					[](const glm::vec3& pos) { return 0.0f; },
					[](const glm::vec3& pos) { return 0.0f; } });
		firstLayer.m_thickness = [](const glm::vec2& position) {return 0.f; };
		firstLayer.m_mat.m_soilMaterialTexture = std::make_shared<SoilMaterialTexture>();
		firstLayer.m_mat.m_soilMaterialTexture->m_color_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
		std::fill(firstLayer.m_mat.m_soilMaterialTexture->m_color_map.begin(), firstLayer.m_mat.m_soilMaterialTexture->m_color_map.end(), glm::vec4(62.0f / 255, 49.0f / 255, 23.0f / 255, 0.0f));
		firstLayer.m_mat.m_soilMaterialTexture->m_height_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
		std::fill(firstLayer.m_mat.m_soilMaterialTexture->m_height_map.begin(), firstLayer.m_mat.m_soilMaterialTexture->m_height_map.end(), 0.1f);

		firstLayer.m_mat.m_soilMaterialTexture->m_metallic_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
		std::fill(firstLayer.m_mat.m_soilMaterialTexture->m_metallic_map.begin(), firstLayer.m_mat.m_soilMaterialTexture->m_metallic_map.end(), 0.2f);

		firstLayer.m_mat.m_soilMaterialTexture->m_roughness_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
		std::fill(firstLayer.m_mat.m_soilMaterialTexture->m_roughness_map.begin(), firstLayer.m_mat.m_soilMaterialTexture->m_roughness_map.end(), 0.8f);

		firstLayer.m_mat.m_soilMaterialTexture->m_normal_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
		std::fill(firstLayer.m_mat.m_soilMaterialTexture->m_normal_map.begin(), firstLayer.m_mat.m_soilMaterialTexture->m_normal_map.end(), glm::vec3(0.0f, 0.0f, 1.0f));

		materialIndex++;
		//Add user defined layers
		auto& soilLayerDescriptors = soilDescriptor->m_soilLayerDescriptors;
		for (int i = 0; i < soilDescriptor->m_soilLayerDescriptors.size(); i++)
		{

			if (auto soilLayerDescriptor = soilLayerDescriptors[i].Get<NoiseSoilLayerDescriptor>())
			{
				soilLayers.emplace_back();
				auto& soilLayer = soilLayers.back();
				soilLayer.m_mat.m_c = [=](const glm::vec3& position) { return soilLayerDescriptor->m_capacity.GetValue(position); };
				soilLayer.m_mat.m_p = [=](const glm::vec3& position) { return soilLayerDescriptor->m_permeability.GetValue(position); };
				soilLayer.m_mat.m_d = [=](const glm::vec3& position) { return soilLayerDescriptor->m_density.GetValue(position); };
				soilLayer.m_mat.m_n = [=](const glm::vec3& position) { return soilLayerDescriptor->m_initialNutrients.GetValue(position); };
				soilLayer.m_mat.m_w = [=](const glm::vec3& position) { return soilLayerDescriptor->m_initialWater.GetValue(position); };
				soilLayer.m_mat.m_id = materialIndex;
				soilLayer.m_thickness = [soilLayerDescriptor](const glm::vec2& position)
				{
					return soilLayerDescriptor->m_thickness.GetValue(position);
				};
				const auto albedo = soilLayerDescriptor->m_albedoTexture.Get<Texture2D>();
				const auto height = soilLayerDescriptor->m_heightTexture.Get<Texture2D>();
				const auto metallic = soilLayerDescriptor->m_metallicTexture.Get<Texture2D>();
				const auto normal = soilLayerDescriptor->m_normalTexture.Get<Texture2D>();
				const auto roughness = soilLayerDescriptor->m_roughnessTexture.Get<Texture2D>();
				soilLayer.m_mat.m_soilMaterialTexture = std::make_shared<SoilMaterialTexture>();
				if (albedo)
				{
					albedo->GetRgbaChannelData(soilLayer.m_mat.m_soilMaterialTexture->m_color_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
					if (i == 0)
					{
						albedo->GetRgbaChannelData(soilLayers[0].m_mat.m_soilMaterialTexture->m_color_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
						for (auto& value : soilLayers[0].m_mat.m_soilMaterialTexture->m_color_map) value.w = 0.0f;
					}
				}else
				{
					soilLayer.m_mat.m_soilMaterialTexture->m_color_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilLayer.m_mat.m_soilMaterialTexture->m_color_map.begin(), soilLayer.m_mat.m_soilMaterialTexture->m_color_map.end(), Application::GetLayer<EcoSysLabLayer>()->m_soilLayerColors[materialIndex]);
				}
				if (height) {
					height->GetRedChannelData(soilLayer.m_mat.m_soilMaterialTexture->m_height_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}else
				{
					soilLayer.m_mat.m_soilMaterialTexture->m_height_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilLayer.m_mat.m_soilMaterialTexture->m_height_map.begin(), soilLayer.m_mat.m_soilMaterialTexture->m_height_map.end(), 1.0f);
				}
				if (metallic) {
					metallic->GetRedChannelData(soilLayer.m_mat.m_soilMaterialTexture->m_metallic_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilLayer.m_mat.m_soilMaterialTexture->m_metallic_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilLayer.m_mat.m_soilMaterialTexture->m_metallic_map.begin(), soilLayer.m_mat.m_soilMaterialTexture->m_metallic_map.end(), 0.2f);
				}
				if (roughness) {
					roughness->GetRedChannelData(soilLayer.m_mat.m_soilMaterialTexture->m_roughness_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilLayer.m_mat.m_soilMaterialTexture->m_roughness_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilLayer.m_mat.m_soilMaterialTexture->m_roughness_map.begin(), soilLayer.m_mat.m_soilMaterialTexture->m_roughness_map.end(), 0.8f);
				}
				if (normal) {
					normal->GetRgbChannelData(soilLayer.m_mat.m_soilMaterialTexture->m_normal_map, soilDescriptor->m_textureResolution.x, soilDescriptor->m_textureResolution.y);
				}
				else
				{
					soilLayer.m_mat.m_soilMaterialTexture->m_normal_map.resize(soilDescriptor->m_textureResolution.x * soilDescriptor->m_textureResolution.y);
					std::fill(soilLayer.m_mat.m_soilMaterialTexture->m_normal_map.begin(), soilLayer.m_mat.m_soilMaterialTexture->m_normal_map.end(), glm::vec3(0, 0, 1));
				}
				materialIndex++;
			}
			else
			{
				soilLayerDescriptors.erase(soilLayerDescriptors.begin() + i);
				i--;
			}
		}

		//Add bottom layer
		soilLayers.emplace_back();
		soilLayers.back().m_thickness = [](const glm::vec2& position) {return 1000.f; };
		soilLayers.back().m_mat.m_id = materialIndex;
		soilLayers.back().m_mat.m_c = [](const glm::vec3& position) {return 1000.f; };
		soilLayers.back().m_mat.m_p = [](const glm::vec3& position) {return 0.0f; };
		soilLayers.back().m_mat.m_d = [](const glm::vec3& position) {return 1000.f; };
		soilLayers.back().m_mat.m_n = [](const glm::vec3& position) {return 0.0f; };
		soilLayers.back().m_mat.m_w = [](const glm::vec3& position) {return 0.0f; };
		m_soilModel.Initialize(params, soilSurface, soilLayers);
	}
}

void Soil::SplitRootTestSetup()
{
	InitializeSoilModel();
	auto soilDescriptor = m_soilDescriptor.Get<SoilDescriptor>();
	if (soilDescriptor) {
		auto heightField = soilDescriptor->m_heightField.Get<HeightField>();
		for (int i = 0; i < m_soilModel.m_n.size(); i++)
		{
			auto position = m_soilModel.GetPositionFromCoordinate(m_soilModel.GetCoordinateFromIndex(i));
			bool underGround = true;
			if (heightField)
			{
				auto height = heightField->GetValue(glm::vec2(position.x, position.z));
				if (position.y >= height) underGround = false;
			}
			if (underGround) {
				if (position.x > m_soilModel.m_boundingBoxMin.x && position.x < m_soilModel.GetVoxelResolution().x * m_soilModel.m_dx * 0.25f + m_soilModel.m_boundingBoxMin.x)
				{
					m_soilModel.m_n[i] = 0.75f;
				}
				else if (position.x < m_soilModel.GetVoxelResolution().x * m_soilModel.m_dx * 0.5f + m_soilModel.m_boundingBoxMin.x)
				{
					m_soilModel.m_n[i] = 0.75f;
				}
				else if (position.x < m_soilModel.GetVoxelResolution().x * m_soilModel.m_dx * 0.75f + m_soilModel.m_boundingBoxMin.x)
				{
					m_soilModel.m_n[i] = 1.25f;
				}
				else
				{
					m_soilModel.m_n[i] = 1.25f;
				}
			}
			else
			{
				m_soilModel.m_n[i] = 0.0f;
			}
		}
	}
}

void Soil::FixedUpdate()
{
	if (m_temporalProgression) {
		if (m_temporalProgressionProgress < 1.0f) {
			auto scene = Application::GetActiveScene();
			auto owner = GetOwner();
			for (const auto& child : scene->GetChildren(owner))
			{
				if (scene->GetEntityName(child) == "CutOut")
				{
					scene->DeleteEntity(child);
					break;
				}
			}
			auto cutOutEntity = GenerateCutOut(m_temporalProgressionProgress, 0.99f, 0, 0, true);
			scene->SetParent(cutOutEntity, owner);
			m_temporalProgressionProgress += 0.01f;
		}
		else
		{
			m_temporalProgressionProgress = 0;
			m_temporalProgression = false;
		}
	}
}


void SerializeSoilParameters(const std::string& name, const SoilParameters& soilParameters, YAML::Emitter& out) {
	out << YAML::Key << name << YAML::BeginMap;
	out << YAML::Key << "m_voxelResolution" << YAML::Value << soilParameters.m_voxelResolution;
	out << YAML::Key << "m_deltaX" << YAML::Value << soilParameters.m_deltaX;
	out << YAML::Key << "m_deltaTime" << YAML::Value << soilParameters.m_deltaTime;
	out << YAML::Key << "m_boundingBoxMin" << YAML::Value << soilParameters.m_boundingBoxMin;

	out << YAML::Key << "m_boundary_x" << YAML::Value << static_cast<int>(soilParameters.m_boundary_x);
	out << YAML::Key << "m_boundary_y" << YAML::Value << static_cast<int>(soilParameters.m_boundary_y);
	out << YAML::Key << "m_boundary_z" << YAML::Value << static_cast<int>(soilParameters.m_boundary_z);

	out << YAML::Key << "m_diffusionForce" << YAML::Value << soilParameters.m_diffusionForce;
	out << YAML::Key << "m_gravityForce" << YAML::Value << soilParameters.m_gravityForce;
	out << YAML::EndMap;
}

void DeserializeSoilParameters(const std::string& name, SoilParameters& soilParameters, const YAML::Node& in) {
	if (in[name]) {
		auto& param = in[name];
		if (param["m_voxelResolution"]) soilParameters.m_voxelResolution = param["m_voxelResolution"].as<glm::uvec3>();
		else {
			UNIENGINE_WARNING("DeserializeSoilParameters: m_voxelResolution not found!");
			//UNIENGINE_ERROR("DeserializeSoilParameters: m_voxelResolution not found!");
			//UNIENGINE_LOG("DeserializeSoilParameters: m_voxelResolution not found!");
		}
		if (param["m_deltaX"]) soilParameters.m_deltaX = param["m_deltaX"].as<float>();
		if (param["m_deltaTime"]) soilParameters.m_deltaTime = param["m_deltaTime"].as<float>();
		if (param["m_boundingBoxMin"]) soilParameters.m_boundingBoxMin = param["m_boundingBoxMin"].as<glm::vec3>();

		if (param["m_boundary_x"]) soilParameters.m_boundary_x = static_cast<SoilModel::Boundary>(param["m_boundary_x"].as<int>());
		if (param["m_boundary_y"]) soilParameters.m_boundary_y = static_cast<SoilModel::Boundary>(param["m_boundary_y"].as<int>());
		if (param["m_boundary_z"]) soilParameters.m_boundary_z = static_cast<SoilModel::Boundary>(param["m_boundary_z"].as<int>());

		if (param["m_diffusionForce"]) soilParameters.m_diffusionForce = param["m_diffusionForce"].as<float>();
		if (param["m_gravityForce"]) soilParameters.m_gravityForce = param["m_gravityForce"].as<glm::vec3>();
	}
}

void SoilDescriptor::Serialize(YAML::Emitter& out)
{
	m_heightField.Save("m_heightField", out);
	SerializeSoilParameters("m_soilParameters", m_soilParameters, out);

	out << YAML::Key << "m_soilLayerDescriptors" << YAML::Value << YAML::BeginSeq;
	for (int i = 0; i < m_soilLayerDescriptors.size(); i++)
	{
		if (auto soilLayerDescriptor = m_soilLayerDescriptors[i].Get<ISoilLayerDescriptor>())
		{
			out << YAML::BeginMap;
			m_soilLayerDescriptors[i].Serialize(out);
			out << YAML::EndMap;
		}
		else
		{
			m_soilLayerDescriptors.erase(m_soilLayerDescriptors.begin() + i);
			i--;
		}
	}
	out << YAML::EndSeq;
}

void SoilDescriptor::Deserialize(const YAML::Node& in)
{
	m_heightField.Load("m_heightField", in);
	DeserializeSoilParameters("m_soilParameters", m_soilParameters, in);
	m_soilLayerDescriptors.clear();
	if (in["m_soilLayerDescriptors"])
	{
		for (const auto& i : in["m_soilLayerDescriptors"])
		{
			m_soilLayerDescriptors.emplace_back();
			m_soilLayerDescriptors.back().Deserialize(i);
		}
	}
}

void SoilDescriptor::CollectAssetRef(std::vector<AssetRef>& list)
{
	list.push_back(m_heightField);

	for (int i = 0; i < m_soilLayerDescriptors.size(); i++)
	{
		if (auto soilLayerDescriptor = m_soilLayerDescriptors[i].Get<ISoilLayerDescriptor>())
		{
			list.push_back(m_soilLayerDescriptors[i]);
		}
		else
		{
			m_soilLayerDescriptors.erase(m_soilLayerDescriptors.begin() + i);
			i--;
		}
	}
}

