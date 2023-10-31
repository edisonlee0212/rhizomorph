#pragma once
#include "VoxelGrid.hpp"
using namespace UniEngine;
namespace EcoSysLab
{
	struct IlluminationEstimationSettings
	{
		float m_voxelSize = 0.1f;
		float m_minShadowIntensity = 0.02f;
		float m_maxShadowIntensity = 1.0f;
		float m_distancePowerFactor = 2.0f;
		float m_distanceMultiplier = 2.0f;

		float m_shadowIntensityMultiplier = 10.0f;
	};
	struct ShadowVolume
	{
		glm::vec3 m_position;
		float m_size;
	};

	struct ShadowVoxel
	{
		glm::vec3 m_shadowDirection = glm::vec3(0.0f);
		float m_shadowIntensity = 0.0f;
	};

	class TreeIlluminationEstimator
	{
	public:
		IlluminationEstimationSettings m_settings;
		VoxelGrid<ShadowVoxel> m_voxel;
		[[nodiscard]] float IlluminationEstimation(const glm::vec3& position, glm::vec3& lightDirection) const;
		void AddShadowVolume(const ShadowVolume& shadowVolume);
	};
}