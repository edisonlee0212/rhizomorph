#include "TreeIlluminationEstimator.hpp"
using namespace EcoSysLab;

float TreeIlluminationEstimator::IlluminationEstimation(const glm::vec3& position, glm::vec3& lightDirection) const
{
	const auto& data = m_voxel.Peek(position);
	const float lightIntensity = glm::max(0.0f, 1.0f - data.m_shadowIntensity);
	if (lightIntensity == 0.0f)
	{
		lightDirection = glm::vec3(0.0f);
	}
	else
	{
		lightDirection = glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f) + glm::normalize(data.m_shadowDirection) * data.m_shadowIntensity);
	}

	return lightIntensity;
}

void TreeIlluminationEstimator::AddShadowVolume(const ShadowVolume& shadowVolume)
{
	const auto voxelMinBound = m_voxel.GetMinBound();
	const auto dx = m_voxel.GetVoxelDiameter();
	const auto voxelResolution = m_voxel.GetResolution();
	if (m_settings.m_distanceMultiplier == 0.0f) return;
	const float maxRadius = glm::pow(shadowVolume.m_size * m_settings.m_shadowIntensityMultiplier / m_settings.m_minShadowIntensity, 1.0f / m_settings.m_distancePowerFactor) / m_settings.m_distanceMultiplier;
	const int xCenter = (shadowVolume.m_position.x - voxelMinBound.x) / dx;
	const int yCenter = (shadowVolume.m_position.y - voxelMinBound.y) / dx;
	const int zCenter = (shadowVolume.m_position.z - voxelMinBound.z) / dx;
	for (int y = glm::clamp(yCenter - static_cast<int>(maxRadius / dx), 0, voxelResolution.y - 1); y <= glm::clamp(yCenter - 1, 0, voxelResolution.y - 1); y++)
	{
		for (int x = glm::clamp(xCenter - static_cast<int>(maxRadius / dx), 0, voxelResolution.x - 1); x <= glm::clamp(xCenter + static_cast<int>(maxRadius / dx), 0, voxelResolution.x - 1); x++)
		{
			for (int z = glm::clamp(zCenter - static_cast<int>(maxRadius / dx), 0, voxelResolution.z - 1); z <= glm::clamp(zCenter + static_cast<int>(maxRadius / dx), 0, voxelResolution.z - 1); z++)
			{
				const auto positionDiff = m_voxel.GetPosition({ x, y, z }) - shadowVolume.m_position;
				const auto angle = glm::atan(glm::sqrt(positionDiff.x * positionDiff.x + positionDiff.z * positionDiff.z) / positionDiff.y);
				const auto distance = glm::length(positionDiff);
				const float shadowIntensity = glm::cos(angle) * glm::min(m_settings.m_maxShadowIntensity, shadowVolume.m_size * m_settings.m_shadowIntensityMultiplier / glm::pow(glm::max(1.0f, distance * m_settings.m_distanceMultiplier), m_settings.m_distancePowerFactor));
				if (shadowIntensity < m_settings.m_minShadowIntensity) continue;
				const auto direction = glm::normalize(positionDiff);
				auto& data = m_voxel.Ref(glm::ivec3(x, y, z));
				data.m_shadowIntensity += shadowIntensity;
				data.m_shadowDirection += direction * data.m_shadowIntensity;
			}
		}
	}
}

