#pragma once

#include "ecosyslab_export.h"
#include "SoilModel.hpp"
#include "HeightField.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
	enum class SoilMaterialType {
		Clay,
		SiltyClay,
		Loam,
		Sand,
		LoamySand,
		Air
	};

	class ISoilLayerDescriptor : public IAsset {};

	class NoiseSoilLayerDescriptor : public ISoilLayerDescriptor
	{
	public:
		AssetRef m_albedoTexture;
		AssetRef m_roughnessTexture;
		AssetRef m_metallicTexture;
		AssetRef m_normalTexture;
		AssetRef m_heightTexture;

		Noises3D m_capacity;
		Noises3D m_permeability;
		Noises3D m_density;
		Noises3D m_initialNutrients;
		Noises3D m_initialWater;
		Noises2D m_thickness;
		void OnInspect() override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
		void CollectAssetRef(std::vector<AssetRef>& list) override;
	};
	
	/**
	 * \brief The soil descriptor contains the procedural parameters for soil model.
	 * It helps provide the user's control menu and serialization outside the portable soil model
	 */
	class SoilDescriptor : public IAsset {
	public:
		SoilParameters m_soilParameters;
		glm::ivec2 m_textureResolution = { 512, 512 };
		std::vector<AssetRef> m_soilLayerDescriptors;
		AssetRef m_heightField;
		/**ImGui menu goes to here. Also you can take care you visualization with Gizmos here.
		 * Note that the visualization will only be activated while you are inspecting the soil private component in the entity inspector.
		 */
		void OnInspect() override;
		void Serialize(YAML::Emitter& out) override;
		void Deserialize(const YAML::Node& in) override;
		void CollectAssetRef(std::vector<AssetRef>& list) override;
	};
	enum class SoilProperty
	{
		Blank,

		WaterDensity,
		WaterDensityGradient,
		DiffusionDivergence,
		GravityDivergence,

		NutrientDensity,

		SoilDensity,

		SoilLayer
	};
	/**
	 * \brief The soil is designed to be a private component of an entity.
	 * It holds the soil model and can be referenced by multiple trees.
	 * The soil will also take care of visualization and menu for soil model.
	 */
	class Soil : public IPrivateComponent {

	public:
		SoilModel m_soilModel;
		AssetRef m_soilDescriptor;
		/**ImGui menu goes to here.Also you can take care you visualization with Gizmos here.
		 * Note that the visualization will only be activated while you are inspecting the soil private component in the entity inspector.
		 */
		void OnInspect() override;

		void Serialize(YAML::Emitter& out) override;

		void Deserialize(const YAML::Node& in) override;

		void CollectAssetRef(std::vector<AssetRef>& list) override;

		Entity GenerateMesh(float xDepth = 0.0f, float zDepth = 0.0f);

		void InitializeSoilModel();

		void SplitRootTestSetup();

		void FixedUpdate() override;
		Entity GenerateSurfaceQuadX(float depth, const glm::vec2& minXY, const glm::vec2 maxXY, float waterFactor, float nutrientFactor);
		Entity GenerateSurfaceQuadZ(float depth, const glm::vec2& minXY, const glm::vec2 maxXY, float waterFactor, float nutrientFactor);

		Entity GenerateCutOut(float xDepth, float zDepth, float waterFactor, float nutrientFactor, bool groundSurface);
		Entity GenerateFullBox(float waterFactor, float nutrientFactor, bool groundSurface);
	private:
		// member variables to avoid static variables (in case of multiple Soil instances?)
		bool m_autoStep = false;
		bool m_irrigation = true;
		float m_temporalProgressionProgress = 0;
		bool m_temporalProgression = false;
		// for user specified sources:
		glm::vec3 m_sourcePositon = glm::vec3(0, 0, 0);
		float m_sourceAmount = 50.f;
		float m_sourceWidth = 1.0f;
	};
}